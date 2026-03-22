"""
NRTL 模型参数拟合优化模块

使用多种优化算法（差分进化 + Nelder-Mead 局部优化）拟合 NRTL 模型参数，
最小化计算溶解度与实验溶解度之间的偏差。
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.data.solubility import SolubilityDataset, ThermodynamicConstants
from src.models.nrtl import NRTLModel, NRTLParameters
from src.utils.logger import get_logger

logger = get_logger("fitting.optimizer")


@dataclass
class FittingResult:
    """拟合结果数据类"""

    parameters: NRTLParameters
    objective_value: float
    rmsd: float
    aard_percent: float
    r_squared: float
    x_calculated: np.ndarray
    x_experimental: np.ndarray
    temperatures: np.ndarray
    residuals: np.ndarray
    relative_deviations: np.ndarray
    tau_12_values: np.ndarray = None
    tau_21_values: np.ndarray = None
    G_12_values: np.ndarray = None
    G_21_values: np.ndarray = None
    convergence_message: str = ""
    iterations: int = 0

    def summary(self) -> str:
        """生成拟合结果的文本摘要"""
        lines = [
            "",
            "=" * 70,
            "NRTL 模型参数拟合结果",
            "壬二酸 (Azelaic Acid) / 水 (Water) 二元体系",
            "=" * 70,
            "",
            "━━━ 拟合参数 (能量交互参数) ━━━",
            f"  Δg₁₂ (dg_12)  = {self.parameters.dg_12:>12.2f} J/mol",
            f"  Δg₂₁ (dg_21)  = {self.parameters.dg_21:>12.2f} J/mol",
            f"  α             = 0.3 (fixed)",
            "",
            "━━━ 拟合质量指标 ━━━",
            f"  目标函数值 (OF)        = {self.objective_value:.6e}",
            f"  均方根偏差 (RMSD)      = {self.rmsd:.6e}",
            f"  平均绝对相对偏差 (AARD) = {self.aard_percent:.4f} %",
            f"  决定系数 (R²)          = {self.r_squared:.8f}",
            "",
            "━━━ 逐点对比 ━━━",
            f"  {'T (K)':>10s}  {'x_exp':>12s}  {'x_calc':>12s}  {'RD (%)':>10s}  {'τ₁₂':>8s}  {'τ₂₁':>8s}",
            "  " + "-" * 70,
        ]

        for i in range(len(self.temperatures)):
            tau12_str = f"{self.tau_12_values[i]:>8.4f}" if self.tau_12_values is not None else "    N/A"
            tau21_str = f"{self.tau_21_values[i]:>8.4f}" if self.tau_21_values is not None else "    N/A"
            lines.append(
                f"  {self.temperatures[i]:>10.2f}"
                f"  {self.x_experimental[i]:>12.4e}"
                f"  {self.x_calculated[i]:>12.4e}"
                f"  {self.relative_deviations[i] * 100:>10.4f}"
                f"  {tau12_str}"
                f"  {tau21_str}"
            )

        lines.extend([
            "  " + "-" * 70,
            f"  {'最大相对偏差':>20s} = {np.max(np.abs(self.relative_deviations)) * 100:.4f} %",
            f"  {'最小相对偏差':>20s} = {np.min(np.abs(self.relative_deviations)) * 100:.4f} %",
            "",
            f"  收敛信息: {self.convergence_message}",
            f"  迭代次数: {self.iterations}",
            "=" * 70,
        ])
        return "\n".join(lines)


class NRTLOptimizer:
    """
    NRTL 模型参数优化器

    使用两阶段优化策略:
    1. 差分进化算法 (DE) - 全局搜索，避免局部最优
    2. Nelder-Mead 单纯形法 - 局部精细优化
    """

    def __init__(
        self,
        dataset: SolubilityDataset,
        constants: ThermodynamicConstants,
        dg_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (-50000.0, 50000.0),
            (-50000.0, 50000.0),
        ),
    ) -> None:
        """
        初始化优化器。

        Args:
            dataset: 溶解度实验数据集
            constants: 热力学常数
            dg_bounds: Δg₁₂ 和 Δg₂₁ 的搜索范围 (J/mol)
        """
        if dataset.size < 2:
            raise ValueError(
                f"数据点数量不足，至少需要 2 个点用于拟合 2 个参数，当前: {dataset.size}"
            )

        self.dataset = dataset
        self.constants = constants
        self.model = NRTLModel(constants)
        self.dg_bounds = dg_bounds
        self._eval_count = 0

        logger.info(
            "优化器初始化: %d 个数据点, Δg₁₂ 范围 %s, Δg₂₁ 范围 %s",
            dataset.size, dg_bounds[0], dg_bounds[1],
        )

    def _objective_function(self, params: np.ndarray) -> float:
        """
        目标函数: 最小化相对偏差的平方和。

        OF = Σ [(x_calc - x_exp) / x_exp]²

        Args:
            params: [Δg₁₂, Δg₂₁]

        Returns:
            目标函数值
        """
        dg_12, dg_21 = params
        self._eval_count += 1

        try:
            x_calc = self.model.calculate_solubility_batch(
                self.dataset.temperatures, dg_12, dg_21
            )

            relative_errors = (x_calc - self.dataset.mole_fractions) / self.dataset.mole_fractions
            obj_value = np.sum(relative_errors ** 2)

            if np.isnan(obj_value) or np.isinf(obj_value):
                return 1e20

            return obj_value

        except (ValueError, OverflowError, FloatingPointError) as e:
            logger.debug("目标函数计算异常 (Δg₁₂=%.1f, Δg₂₁=%.1f): %s", dg_12, dg_21, e)
            return 1e20

    def fit(
        self,
        de_seed: int = 42,
        de_maxiter: int = 1000,
        de_tol: float = 1e-12,
        de_popsize: int = 25,
        local_maxiter: int = 5000,
        local_tol: float = 1e-14,
    ) -> FittingResult:
        """
        执行两阶段参数拟合。

        阶段1: 差分进化全局搜索
        阶段2: L-BFGS-B 有界局部精细优化

        Args:
            de_seed: 差分进化随机种子
            de_maxiter: DE 最大迭代次数
            de_tol: DE 收敛容差
            de_popsize: DE 种群大小
            local_maxiter: 局部优化最大迭代次数
            local_tol: 局部优化收敛容差

        Returns:
            FittingResult 拟合结果
        """
        logger.info("━━━ 阶段 1: 差分进化全局搜索 ━━━")
        self._eval_count = 0

        nrtl_logger = logging.getLogger("nrtl_fitting.models.nrtl")
        original_level = nrtl_logger.level
        nrtl_logger.setLevel(logging.ERROR)

        de_result = differential_evolution(
            self._objective_function,
            bounds=self.dg_bounds,
            seed=de_seed,
            maxiter=de_maxiter,
            tol=de_tol,
            popsize=de_popsize,
            strategy="best1bin",
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False,
        )

        logger.info(
            "DE 完成: Δg₁₂=%.2f, Δg₂₁=%.2f, OF=%.6e, 函数评估=%d",
            de_result.x[0], de_result.x[1], de_result.fun, self._eval_count,
        )

        logger.info("━━━ 阶段 2: L-BFGS-B 局部精细优化 ━━━")
        eval_before_local = self._eval_count

        local_result = minimize(
            self._objective_function,
            x0=de_result.x,
            method="L-BFGS-B",
            bounds=self.dg_bounds,
            options={
                "maxiter": local_maxiter,
                "ftol": local_tol,
                "gtol": 1e-10,
            },
        )

        nrtl_logger.setLevel(original_level)

        logger.info(
            "局部优化完成: Δg₁₂=%.2f, Δg₂₁=%.2f, OF=%.6e, 额外评估=%d",
            local_result.x[0], local_result.x[1],
            local_result.fun, self._eval_count - eval_before_local,
        )

        dg_12_opt, dg_21_opt = local_result.x
        params = NRTLParameters(dg_12=dg_12_opt, dg_21=dg_21_opt)

        result = self._build_result(params, local_result)
        return result

    def _build_result(self, params: NRTLParameters, opt_result) -> FittingResult:
        """
        构建完整的拟合结果。

        Args:
            params: 优化后的 NRTL 参数
            opt_result: scipy 优化结果对象

        Returns:
            FittingResult 实例
        """
        x_calc = self.model.calculate_solubility_batch(
            self.dataset.temperatures, params.dg_12, params.dg_21
        )
        x_exp = self.dataset.mole_fractions

        residuals = x_calc - x_exp
        relative_devs = residuals / x_exp

        rmsd = np.mean(residuals ** 2)
        aard = np.mean(relative_devs) * 100

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((x_exp - np.mean(x_exp)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        tau_12_arr = np.zeros(len(self.dataset.temperatures))
        tau_21_arr = np.zeros(len(self.dataset.temperatures))
        G_12_arr = np.zeros(len(self.dataset.temperatures))
        G_21_arr = np.zeros(len(self.dataset.temperatures))
        for i, T in enumerate(self.dataset.temperatures):
            t12, t21 = self.model.compute_tau(T, params.dg_12, params.dg_21)
            g12, g21 = self.model.compute_G_parameters(t12, t21)
            tau_12_arr[i], tau_21_arr[i] = t12, t21
            G_12_arr[i], G_21_arr[i] = g12, g21

        result = FittingResult(
            parameters=params,
            objective_value=opt_result.fun,
            rmsd=rmsd,
            aard_percent=aard,
            r_squared=r_squared,
            x_calculated=x_calc,
            x_experimental=x_exp,
            temperatures=self.dataset.temperatures,
            residuals=residuals,
            relative_deviations=relative_devs,
            tau_12_values=tau_12_arr,
            tau_21_values=tau_21_arr,
            G_12_values=G_12_arr,
            G_21_values=G_21_arr,
            convergence_message=str(opt_result.message) if hasattr(opt_result, "message") else "完成",
            iterations=opt_result.nit if hasattr(opt_result, "nit") else 0,
        )

        logger.info("拟合质量: RMSD=%.4e, AARD=%.4f%%, R²=%.8f", rmsd, aard, r_squared)
        return result

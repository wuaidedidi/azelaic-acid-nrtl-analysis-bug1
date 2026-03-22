"""
溶解度数据分析模块

基于 Pandas 对壬二酸/水体系溶解度实验数据与 NRTL 拟合结果进行深度分析，包括:
1. 描述性统计 — 实验数据的分布特征
2. 热力学推导 — van't Hoff 线性回归计算溶解焓、溶解熵、吉布斯自由能
3. 误差与异常值分析 — 残差统计、Z-score 异常检测
4. 参数灵敏度分析 — Δg₁₂/Δg₂₁ 微扰对 AARD 的影响
5. 温度区间分组分析 — 低/中/高温段的拟合表现差异
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats

from src.data.solubility import SolubilityDataset, ThermodynamicConstants
from src.fitting.optimizer import FittingResult
from src.models.nrtl import NRTLModel
from src.utils.logger import get_logger

logger = get_logger("analysis.analyzer")


@dataclass
class ThermodynamicQuantities:
    """van't Hoff 线性回归推导的热力学量"""

    delta_H_sol: float       # 表观溶解焓 (J/mol)
    delta_S_sol: float       # 表观溶解熵 (J/(mol·K))
    delta_G_sol_298: float   # 298.15 K 下吉布斯自由能 (J/mol)
    r_squared: float         # 线性回归 R²
    slope: float             # ln(x) vs 1/T 斜率
    intercept: float         # ln(x) vs 1/T 截距


@dataclass
class SensitivityResult:
    """单参数灵敏度分析结果"""

    parameter_name: str
    base_value: float
    perturbations: np.ndarray       # 扰动比例 (如 -0.10 ~ +0.10)
    perturbed_values: np.ndarray    # 扰动后的参数值
    aard_values: np.ndarray         # 对应的 AARD (%)
    gradient: float                 # AARD 对参数值的局部梯度 (% per J/mol)


@dataclass
class AnalysisReport:
    """完整数据分析报告"""

    master_df: pd.DataFrame
    descriptive_stats: pd.DataFrame
    thermodynamic: ThermodynamicQuantities
    residual_stats: Dict[str, float]
    outlier_indices: List[int]
    sensitivity_dg12: SensitivityResult
    sensitivity_dg21: SensitivityResult
    group_analysis: pd.DataFrame
    correlation_matrix: pd.DataFrame


class SolubilityAnalyzer:
    """
    壬二酸溶解度数据分析器

    整合实验数据与 NRTL 拟合结果，执行多维度数据分析并输出 Pandas DataFrame。
    """

    R = 8.314  # J/(mol·K)

    def __init__(
        self,
        dataset: SolubilityDataset,
        result: FittingResult,
        constants: ThermodynamicConstants,
    ) -> None:
        self.dataset = dataset
        self.result = result
        self.constants = constants
        self.model = NRTLModel(constants)
        self._master_df: pd.DataFrame | None = None
        logger.info("分析器初始化完成，共 %d 个数据点", dataset.size)

    @property
    def master_df(self) -> pd.DataFrame:
        """构建主数据表，合并实验值、计算值与各类推导量"""
        if self._master_df is not None:
            return self._master_df

        r = self.result
        inv_T = 1.0 / r.temperatures
        ln_x_exp = np.log(r.x_experimental)
        ln_x_calc = np.log(r.x_calculated)

        gamma_values = np.zeros(len(r.temperatures))
        for i, T in enumerate(r.temperatures):
            tau_12, tau_21 = self.model.compute_tau(
                T, r.parameters.dg_12, r.parameters.dg_21
            )
            gamma_values[i] = self.model.activity_coefficient(
                r.x_experimental[i], tau_12, tau_21
            )

        df = pd.DataFrame({
            "T_K": r.temperatures,
            "T_C": r.temperatures - 273.15,
            "inv_T": inv_T,
            "x_exp": r.x_experimental,
            "x_calc": r.x_calculated,
            "ln_x_exp": ln_x_exp,
            "ln_x_calc": ln_x_calc,
            "residual": r.residuals,
            "abs_residual": np.abs(r.residuals),
            "relative_deviation": r.relative_deviations,
            "rd_percent": r.relative_deviations,
            "abs_rd_percent": np.abs(r.relative_deviations) * 100,
            "tau_12": r.tau_12_values,
            "tau_21": r.tau_21_values,
            "G_12": r.G_12_values,
            "G_21": r.G_21_values,
            "gamma_1": gamma_values,
            "ln_gamma_1": np.log(gamma_values),
        })

        bins = pd.cut(
            df["T_K"],
            bins=[280, 295, 310, 325],
            labels=["低温(280-295K)", "高温(310-325K)", "中温(295-310K)"],
        )
        df["temp_group"] = bins

        self._master_df = df
        logger.info("主数据表构建完成: %d 行 × %d 列", len(df), len(df.columns))
        return df

    def descriptive_statistics(self) -> pd.DataFrame:
        """对核心数值列进行描述性统计"""
        cols = ["T_K", "x_exp", "x_calc", "rd_percent", "gamma_1", "tau_12"]
        desc = self.master_df[cols].describe()
        desc.loc["range"] = desc.loc["max"] - desc.loc["min"]
        desc.loc["cv_%"] = (desc.loc["std"] / desc.loc["mean"]).abs() * 100
        logger.info("描述性统计完成")
        return desc

    def vant_hoff_regression(self) -> ThermodynamicQuantities:
        """
        van't Hoff 线性回归: ln(x) = -ΔH_sol/(R·T) + ΔS_sol/R

        对实验数据拟合 ln(x_exp) vs 1/T，提取表观溶解焓和溶解熵。
        """
        df = self.master_df
        slope, intercept, r_value, _, _ = stats.linregress(df["inv_T"], df["ln_x_exp"])

        delta_H = -slope * self.R
        delta_S = intercept * self.R
        delta_G_298 = delta_H - 298.15 * delta_S

        result = ThermodynamicQuantities(
            delta_H_sol=delta_H,
            delta_S_sol=delta_S,
            delta_G_sol_298=delta_G_298,
            r_squared=r_value ** 2,
            slope=slope,
            intercept=intercept,
        )

        logger.info(
            "van't Hoff 回归: ΔH_sol=%.1f J/mol, ΔS_sol=%.2f J/(mol·K), R²=%.6f",
            delta_H, delta_S, r_value ** 2,
        )
        return result

    def residual_analysis(self) -> Dict[str, float]:
        """残差统计分析"""
        rd = self.master_df["rd_percent"]
        residuals = self.master_df["residual"]

        _, shapiro_p = stats.shapiro(rd) if len(rd) >= 3 else (0, 0)

        result = {
            "mean_rd_percent": float(rd.mean()),
            "std_rd_percent": float(rd.std()),
            "median_rd_percent": float(rd.median()),
            "max_abs_rd_percent": float(rd.abs().max()),
            "skewness": float(rd.skew()),
            "kurtosis": float(rd.kurtosis()),
            "shapiro_p_value": float(shapiro_p),
            "mean_abs_residual": float(residuals.abs().mean()),
            "max_abs_residual": float(residuals.abs().max()),
            "rmsd": float(np.sqrt((residuals ** 2).mean())),
        }

        logger.info(
            "残差分析: 均值偏差=%.4f%%, 标准差=%.4f%%, 偏度=%.4f",
            result["mean_rd_percent"], result["std_rd_percent"], result["skewness"],
        )
        return result

    def detect_outliers(self, z_threshold: float = 0.5) -> List[int]:
        """基于 Z-score 的异常值检测"""
        rd = self.master_df["rd_percent"]
        z_scores = np.abs(stats.zscore(rd))
        outlier_mask = z_scores > z_threshold
        outlier_indices = list(self.master_df.index[outlier_mask])

        if outlier_indices:
            logger.warning(
                "检测到 %d 个潜在异常点 (Z>%.1f): 索引 %s",
                len(outlier_indices), z_threshold, outlier_indices,
            )
        else:
            logger.info("未检测到异常点 (Z 阈值=%.1f)", z_threshold)

        return outlier_indices

    def parameter_sensitivity(
        self,
        perturbation_range: float = 0.10,
        n_points: int = 21,
    ) -> tuple[SensitivityResult, SensitivityResult]:
        """
        参数灵敏度分析

        对 Δg₁₂ 和 Δg₂₁ 分别施加 ±perturbation_range 的扰动，
        计算每个扰动水平下的 AARD，评估模型对参数的敏感程度。
        """
        base_dg12 = self.result.parameters.dg_12
        base_dg21 = self.result.parameters.dg_21
        temperatures = self.result.temperatures
        x_exp = self.result.x_experimental

        perturbations = np.linspace(0, perturbation_range, n_points)

        def compute_aard(dg12: float, dg21: float) -> float:
            x_calc = self.model.calculate_solubility_batch(temperatures, dg12, dg21)
            return float(np.mean(np.abs((x_calc - x_exp) / x_exp)) * 100)

        aard_dg12 = np.array([
            compute_aard(base_dg12 * (1 + p), base_dg21) for p in perturbations
        ])
        aard_dg21 = np.array([
            compute_aard(base_dg12, base_dg21 * (1 + p)) for p in perturbations
        ])

        grad_dg12 = float(np.gradient(aard_dg12, base_dg12 * perturbations).mean()) if base_dg12 != 0 else 0.0
        grad_dg21 = float(np.gradient(aard_dg21, base_dg21 * perturbations).mean()) if base_dg21 != 0 else 0.0

        s_dg12 = SensitivityResult(
            parameter_name="Δg₁₂",
            base_value=base_dg12,
            perturbations=perturbations,
            perturbed_values=base_dg12 * (1 + perturbations),
            aard_values=aard_dg12,
            gradient=grad_dg12,
        )
        s_dg21 = SensitivityResult(
            parameter_name="Δg₂₁",
            base_value=base_dg21,
            perturbations=perturbations,
            perturbed_values=base_dg21 * (1 + perturbations),
            aard_values=aard_dg21,
            gradient=grad_dg21,
        )

        logger.info(
            "灵敏度分析完成: Δg₁₂ 梯度=%.6f, Δg₂₁ 梯度=%.6f",
            grad_dg12, grad_dg21,
        )
        return s_dg12, s_dg21

    def group_analysis(self) -> pd.DataFrame:
        """按温度区间分组统计拟合表现"""
        df = self.master_df.dropna(subset=["temp_group"])
        grouped = df.groupby("temp_group", observed=True).agg(
            n_points=("T_K", "count"),
            mean_x_exp=("x_exp", "mean"),
            mean_x_calc=("x_calc", "mean"),
            mean_abs_rd=("abs_rd_percent", "mean"),
            max_abs_rd=("abs_rd_percent", "max"),
            mean_gamma=("gamma_1", "mean"),
            mean_tau_12=("tau_12", "mean"),
            mean_tau_21=("tau_21", "mean"),
        ).round(6)
        logger.info("温度分组分析完成: %d 个分组", len(grouped))
        return grouped

    def correlation_analysis(self) -> pd.DataFrame:
        """计算核心变量间的 Pearson 相关系数矩阵"""
        cols = ["T_K", "x_exp", "x_calc", "gamma_1", "tau_12", "tau_21", "rd_percent"]
        corr = self.master_df[cols].corr().round(4)
        logger.info("相关性分析完成: %d × %d 矩阵", *corr.shape)
        return corr

    def run_full_analysis(self) -> AnalysisReport:
        """执行完整分析流程，返回汇总报告"""
        logger.info("=" * 50)
        logger.info("开始执行完整数据分析...")
        logger.info("=" * 50)

        desc = self.descriptive_statistics()
        thermo = self.vant_hoff_regression()
        resid = self.residual_analysis()
        outliers = self.detect_outliers()
        s_dg12, s_dg21 = self.parameter_sensitivity()
        groups = self.group_analysis()
        corr = self.correlation_analysis()

        report = AnalysisReport(
            master_df=self.master_df,
            descriptive_stats=desc,
            thermodynamic=thermo,
            residual_stats=resid,
            outlier_indices=outliers,
            sensitivity_dg12=s_dg12,
            sensitivity_dg21=s_dg21,
            group_analysis=groups,
            correlation_matrix=corr,
        )

        logger.info("完整数据分析已完成")
        return report

    def export_master_csv(self, output_dir: str = "output") -> str:
        """将主数据表导出为 CSV"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / "analysis_master_data.csv"
        self.master_df.to_csv(filepath, index=False, float_format="%.1e")
        logger.info("主数据表已导出: %s", filepath)
        return str(filepath)

    def export_analysis_report(self, output_dir: str = "output") -> str:
        """将分析报告导出为 JSON"""
        import json

        report = self.run_full_analysis()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "descriptive_statistics": report.descriptive_stats.to_dict(),
            "thermodynamic_quantities": {
                "delta_H_sol_J_per_mol": round(report.thermodynamic.delta_H_sol, -2),
                "delta_S_sol_J_per_mol_K": round(report.thermodynamic.delta_S_sol, 4),
                "delta_G_sol_298K_J_per_mol": round(report.thermodynamic.delta_G_sol_298, 2),
                "vant_hoff_R_squared": round(report.thermodynamic.r_squared, 6),
                "slope": round(report.thermodynamic.slope, 4),
                "intercept": round(report.thermodynamic.intercept, 4),
            },
            "residual_analysis": report.residual_stats,
            "outlier_indices": report.outlier_indices,
            "group_analysis": report.group_analysis.to_dict(),
            "correlation_matrix": report.correlation_matrix.to_dict(),
            "sensitivity": {
                "dg12": {
                    "base_value": report.sensitivity_dg12.base_value,
                    "gradient": report.sensitivity_dg12.gradient,
                    "aard_at_perturbations": {
                        f"{p:+.2%}": round(a, 4)
                        for p, a in zip(
                            report.sensitivity_dg12.perturbations,
                            report.sensitivity_dg12.aard_values,
                        )
                    },
                },
                "dg21": {
                    "base_value": report.sensitivity_dg21.base_value,
                    "gradient": report.sensitivity_dg21.gradient,
                    "aard_at_perturbations": {
                        f"{p:+.2%}": round(a, 4)
                        for p, a in zip(
                            report.sensitivity_dg21.perturbations,
                            report.sensitivity_dg21.aard_values,
                        )
                    },
                },
            },
        }

        filepath = path / "analysis_report.json"
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("分析报告 JSON 已导出: %s", filepath)
        return str(filepath)

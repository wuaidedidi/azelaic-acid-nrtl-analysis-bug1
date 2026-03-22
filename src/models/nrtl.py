"""
NRTL (Non-Random Two-Liquid) 热力学模型模块

实现 NRTL 活度系数模型，用于固-液平衡 (SLE) 计算。
模型基于 Renon & Prausnitz (1968) 提出的局部组成概念。

固-液平衡基本方程:
    ln(x₁) = -(ΔH_fus / R) × (1/T - 1/T_m) - ln(γ₁)

NRTL 活度系数方程 (二元体系，组分1为溶质，组分2为溶剂):
    ln(γ₁) = x₂² × [τ₂₁ × (G₂₁ / (x₁ + x₂·G₂₁))²
                    + τ₁₂ × G₁₂ / (x₂ + x₁·G₁₂)²]

其中 (标准温度依赖参数化):
    τ₁₂(T) = Δg₁₂ / (R·T)
    τ₂₁(T) = Δg₂₁ / (R·T)
    G₁₂ = exp(-α × τ₁₂)
    G₂₁ = exp(-α × τ₂₁)

拟合参数为能量交互参数 Δg₁₂ 和 Δg₂₁ (J/mol)，
τ 在每个温度点动态计算，确保活度系数具有正确的温度依赖性。
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.data.solubility import ThermodynamicConstants
from src.utils.logger import get_logger

logger = get_logger("models.nrtl")


@dataclass
class NRTLParameters:
    """NRTL 模型能量交互参数 (J/mol)"""

    dg_12: float  # 溶质-溶剂能量交互参数 Δg₁₂ (J/mol)
    dg_21: float  # 溶剂-溶质能量交互参数 Δg₂₁ (J/mol)

    def __repr__(self) -> str:
        return f"NRTLParameters(Δg₁₂={self.dg_12:.2f} J/mol, Δg₂₁={self.dg_21:.2f} J/mol)"


class NRTLModel:
    """
    NRTL 活度系数模型

    用于计算二元体系中溶质的活度系数，
    进而通过固-液平衡方程计算溶解度。
    """

    def __init__(self, constants: ThermodynamicConstants) -> None:
        """
        初始化 NRTL 模型。

        Args:
            constants: 热力学常数（熔点、熔化焓、气体常数、α参数）
        """
        constants.validate()
        self.constants = constants
        logger.info("NRTL 模型初始化完成，α = %.2f", constants.alpha)

    def compute_tau(
        self, T: float, dg_12: float, dg_21: float
    ) -> Tuple[float, float]:
        """
        计算温度依赖的 τ 参数。

        τ₁₂(T) = Δg₁₂ / (R·T)
        τ₂₁(T) = Δg₂₁ / (R·T)

        Args:
            T: 温度 (K)
            dg_12: Δg₁₂ 能量参数 (J/mol)
            dg_21: Δg₂₁ 能量参数 (J/mol)

        Returns:
            (τ₁₂, τ₂₁) 元组
        """
        RT = self.constants.R * T
        return dg_12 / RT, dg_21 / RT

    def compute_G_parameters(
        self, tau_12: float, tau_21: float
    ) -> Tuple[float, float]:
        """
        计算 NRTL 模型的 G 参数。

        Args:
            tau_12: τ₁₂ 交互参数
            tau_21: τ₂₁ 交互参数

        Returns:
            (G₁₂, G₂₁) 元组
        """
        alpha = self.constants.alpha
        G_12 = np.exp(-alpha * tau_12)
        G_21 = np.exp(-alpha * tau_21)
        return G_12, G_21

    def activity_coefficient(
        self, x1: float, tau_12: float, tau_21: float
    ) -> float:
        """
        计算溶质（组分1）的活度系数 γ₁。

        NRTL 方程 (二元体系):
        ln(γ₁) = x₂² × [τ₂₁ × (G₂₁/(x₁ + x₂·G₂₁))²
                        + τ₁₂·G₁₂/(x₂ + x₁·G₁₂)²]

        Args:
            x1: 溶质的摩尔分数
            tau_12: τ₁₂ 参数
            tau_21: τ₂₁ 参数

        Returns:
            活度系数 γ₁

        Raises:
            ValueError: 当摩尔分数不在有效范围时
        """
        if x1 <= 0 or x1 >= 1:
            raise ValueError(f"溶质摩尔分数必须在 (0, 1) 范围内，当前值: {x1}")

        x2 = 1.0 - x1
        G_12, G_21 = self.compute_G_parameters(tau_12, tau_21)

        denom_21 = x1 + x2 * G_21
        denom_12 = x2 + x1 * G_12

        if abs(denom_21) < 1e-15 or abs(denom_12) < 1e-15:
            raise ValueError("NRTL 计算中出现分母趋近于零的情况")

        term1 = tau_21 * (G_21 / denom_21) ** 2
        term2 = tau_12 * G_12 / (denom_12 ** 2)

        ln_gamma_1 = x2 ** 2 * (term1 + term2)
        gamma_1 = np.exp(ln_gamma_1)

        return gamma_1

    def ln_activity_coefficient(
        self, x1: float, tau_12: float, tau_21: float
    ) -> float:
        """
        计算溶质活度系数的自然对数 ln(γ₁)。

        Args:
            x1: 溶质的摩尔分数
            tau_12: τ₁₂ 参数
            tau_21: τ₂₁ 参数

        Returns:
            ln(γ₁)
        """
        x2 = 1.0 - x1
        G_12, G_21 = self.compute_G_parameters(tau_12, tau_21)

        denom_21 = x1 + x2 * G_21
        denom_12 = x2 + x1 * G_12

        if abs(denom_21) < 1e-15 or abs(denom_12) < 1e-15:
            return 1e10

        term1 = tau_21 * (G_21 / denom_21) ** 2
        term2 = tau_12 * G_12 / (denom_12 ** 2)

        return x2 ** 2 * (term1 + term2)

    def ideal_solubility_ln(self, T: float) -> float:
        """
        计算理想溶解度的自然对数 ln(x_ideal)。

        基于 van't Hoff 方程:
            ln(x_ideal) = -(ΔH_fus / R) × (1/T - 1/T_m)

        Args:
            T: 温度 (K)

        Returns:
            ln(x_ideal)

        Raises:
            ValueError: 温度超出合理范围
        """
        if T <= 0:
            raise ValueError(f"温度必须为正值: {T} K")
        if T > self.constants.T_m:
            logger.warning(
                "温度 %.2f K 超过熔点 %.2f K，结果可能不可靠",
                T, self.constants.T_m,
            )

        return -(self.constants.delta_H_fus / self.constants.R) * (
            1.0 / T - 1.0 / self.constants.T_m
        )

    def ln_activity_coefficient_inf_dilution(
        self, tau_12: float, tau_21: float
    ) -> float:
        """
        计算无限稀释下溶质的活度系数对数 ln(γ₁∞)。

        当 x₁ → 0, x₂ → 1 时:
            ln(γ₁∞) = τ₂₁ + τ₁₂ × G₁₂

        Args:
            tau_12: τ₁₂ 参数
            tau_21: τ₂₁ 参数

        Returns:
            ln(γ₁∞)
        """
        G_12, _ = self.compute_G_parameters(tau_12, tau_21)
        return tau_21 + tau_12 * G_12

    def calculate_solubility(
        self, T: float, dg_12: float, dg_21: float,
        max_iter: int = 50, tol: float = 1e-10,
    ) -> float:
        """
        求解固-液平衡方程，计算给定温度下的溶解度。

        固-液平衡方程:
            ln(x₁) = -(ΔH_fus/R) × (1/T - 1/T_m) - ln(γ₁)

        求解策略:
            1. 计算温度依赖的 τ(T) = Δg/(RT)
            2. 用无限稀释活度系数做初始估计
            3. 迭代精修直至收敛

        Args:
            T: 温度 (K)
            dg_12: Δg₁₂ 能量参数 (J/mol)
            dg_21: Δg₂₁ 能量参数 (J/mol)
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            溶质摩尔分数 x₁
        """
        tau_12, tau_21 = self.compute_tau(T, dg_12, dg_21)
        ln_x_ideal = self.ideal_solubility_ln(T)

        ln_gamma_inf = self.ln_activity_coefficient_inf_dilution(tau_12, tau_21)
        ln_x1 = ln_x_ideal - ln_gamma_inf
        ln_x1 = max(ln_x1, -700.0)
        x1 = np.exp(ln_x1)
        x1 = np.clip(x1, 1e-30, 0.999)

        for _ in range(max_iter):
            ln_gamma = self.ln_activity_coefficient(x1, tau_12, tau_21)
            ln_x_new = ln_x_ideal - ln_gamma
            ln_x_new = max(ln_x_new, -700.0)
            x1_new = np.exp(ln_x_new)
            x1_new = np.clip(x1_new, 1e-30, 0.999)

            if abs(x1_new - x1) / max(abs(x1), 1e-30) < tol:
                return float(x1_new)

            x1 = x1_new

        return float(x1)

    def calculate_solubility_batch(
        self, temperatures: np.ndarray, dg_12: float, dg_21: float
    ) -> np.ndarray:
        """
        批量计算多个温度下的溶解度。

        Args:
            temperatures: 温度数组 (K)
            dg_12: Δg₁₂ 能量参数 (J/mol)
            dg_21: Δg₂₁ 能量参数 (J/mol)

        Returns:
            各温度对应的摩尔分数数组
        """
        results = np.zeros(len(temperatures))
        for i, T in enumerate(temperatures):
            results[i] = self.calculate_solubility(T, dg_12, dg_21)
        return results

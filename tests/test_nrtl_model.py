"""
测试模块: NRTL 热力学模型计算正确性
"""

import unittest
import math

import numpy as np

from src.data.solubility import ThermodynamicConstants
from src.models.nrtl import NRTLModel, NRTLParameters


class TestNRTLParameters(unittest.TestCase):
    """测试 NRTL 参数数据类"""

    def test_create_parameters(self):
        params = NRTLParameters(dg_12=15000.0, dg_21=12000.0)
        self.assertAlmostEqual(params.dg_12, 15000.0)
        self.assertAlmostEqual(params.dg_21, 12000.0)

    def test_repr(self):
        params = NRTLParameters(dg_12=15000.0, dg_21=12000.0)
        r = repr(params)
        self.assertIn("Δg₁₂", r)
        self.assertIn("J/mol", r)


class TestNRTLModel(unittest.TestCase):
    """测试 NRTL 模型核心计算"""

    def setUp(self):
        self.constants = ThermodynamicConstants()
        self.model = NRTLModel(self.constants)

    def test_compute_tau(self):
        """验证 τ(T) = Δg / (R·T)"""
        T = 300.0
        dg_12, dg_21 = 10000.0, 8000.0
        tau_12, tau_21 = self.model.compute_tau(T, dg_12, dg_21)
        expected_tau_12 = 10000.0 / (8.314 * 300.0)
        expected_tau_21 = 8000.0 / (8.314 * 300.0)
        self.assertAlmostEqual(tau_12, expected_tau_12, places=6)
        self.assertAlmostEqual(tau_21, expected_tau_21, places=6)

    def test_tau_temperature_dependence(self):
        """验证 τ 随温度变化（高温时 τ 更小）"""
        dg_12, dg_21 = 10000.0, 8000.0
        tau_12_low, _ = self.model.compute_tau(280.0, dg_12, dg_21)
        tau_12_high, _ = self.model.compute_tau(320.0, dg_12, dg_21)
        self.assertGreater(tau_12_low, tau_12_high)

    def test_compute_G_parameters(self):
        """验证 G = exp(-α·τ)"""
        tau_12, tau_21 = 5.0, 4.0
        G_12, G_21 = self.model.compute_G_parameters(tau_12, tau_21)
        expected_G_12 = math.exp(-0.3 * 5.0)
        expected_G_21 = math.exp(-0.3 * 4.0)
        self.assertAlmostEqual(G_12, expected_G_12, places=10)
        self.assertAlmostEqual(G_21, expected_G_21, places=10)

    def test_activity_coefficient_dilute(self):
        """稀溶液活度系数应 > 1（溶质在水中溶解度低）"""
        gamma = self.model.activity_coefficient(1e-4, 5.0, 4.0)
        self.assertGreater(gamma, 1.0)

    def test_activity_coefficient_invalid_x(self):
        """无效摩尔分数应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.model.activity_coefficient(0.0, 5.0, 4.0)
        with self.assertRaises(ValueError):
            self.model.activity_coefficient(1.0, 5.0, 4.0)

    def test_ln_activity_coefficient(self):
        """ln(γ₁) 应与 log(activity_coefficient) 一致"""
        x1 = 1e-4
        tau_12, tau_21 = 5.0, 4.0
        ln_gamma = self.model.ln_activity_coefficient(x1, tau_12, tau_21)
        gamma = self.model.activity_coefficient(x1, tau_12, tau_21)
        self.assertAlmostEqual(ln_gamma, math.log(gamma), places=8)

    def test_ln_activity_coefficient_inf_dilution(self):
        """无限稀释: ln(γ₁∞) = τ₂₁ + τ₁₂·G₁₂"""
        tau_12, tau_21 = 5.0, 4.0
        G_12, _ = self.model.compute_G_parameters(tau_12, tau_21)
        expected = tau_21 + tau_12 * G_12
        result = self.model.ln_activity_coefficient_inf_dilution(tau_12, tau_21)
        self.assertAlmostEqual(result, expected, places=10)

    def test_ideal_solubility_ln(self):
        """验证 van't Hoff 方程: ln(x_ideal) = -(ΔH/R)(1/T - 1/T_m)"""
        T = 300.0
        result = self.model.ideal_solubility_ln(T)
        expected = -(34500.0 / 8.314) * (1.0 / 300.0 - 1.0 / 379.65)
        self.assertAlmostEqual(result, expected, places=8)

    def test_ideal_solubility_at_melting_point(self):
        """熔点处理想溶解度应为 1 (ln=0)"""
        result = self.model.ideal_solubility_ln(379.65)
        self.assertAlmostEqual(result, 0.0, places=8)

    def test_calculate_solubility_positive(self):
        """溶解度计算结果应为正值"""
        x = self.model.calculate_solubility(300.0, 15000.0, 12000.0)
        self.assertGreater(x, 0.0)
        self.assertLess(x, 1.0)

    def test_calculate_solubility_increases_with_temperature(self):
        """溶解度应随温度升高而增大"""
        x_low = self.model.calculate_solubility(283.15, 15000.0, 12000.0)
        x_high = self.model.calculate_solubility(323.15, 15000.0, 12000.0)
        self.assertGreater(x_high, x_low)

    def test_calculate_solubility_batch(self):
        """批量计算应与逐个计算一致"""
        temps = np.array([283.15, 303.15, 323.15])
        dg_12, dg_21 = 15000.0, 12000.0
        batch = self.model.calculate_solubility_batch(temps, dg_12, dg_21)
        for i, T in enumerate(temps):
            single = self.model.calculate_solubility(T, dg_12, dg_21)
            self.assertAlmostEqual(batch[i], single, places=12)


if __name__ == "__main__":
    unittest.main()

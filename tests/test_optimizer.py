"""
测试模块: NRTL 参数拟合优化器
"""

import unittest

import numpy as np

from src.data.solubility import SolubilityDataset, ThermodynamicConstants
from src.fitting.optimizer import NRTLOptimizer, FittingResult


class TestNRTLOptimizer(unittest.TestCase):
    """测试优化器初始化与拟合"""

    def setUp(self):
        self.dataset = SolubilityDataset.load_default()
        self.constants = ThermodynamicConstants()

    def test_optimizer_init(self):
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        self.assertIsNotNone(optimizer.model)
        self.assertEqual(len(self.dataset.temperatures), 9)

    def test_optimizer_init_insufficient_data(self):
        """数据不足时应抛出异常"""
        single_point = SolubilityDataset(
            data_points=self.dataset.data_points[:1]
        )
        with self.assertRaises(ValueError):
            NRTLOptimizer(dataset=single_point, constants=self.constants)

    def test_fit_returns_result(self):
        """拟合应返回完整 FittingResult"""
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        result = optimizer.fit(
            de_seed=42, de_maxiter=200, de_popsize=10,
            local_maxiter=500,
        )
        self.assertIsInstance(result, FittingResult)

    def test_fit_quality_aard(self):
        """AARD 应 < 15%"""
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        result = optimizer.fit(
            de_seed=42, de_maxiter=500, de_popsize=15,
            local_maxiter=2000,
        )
        self.assertLess(result.aard_percent, 15.0,
                        f"AARD {result.aard_percent:.2f}% 超过 15% 阈值")

    def test_fit_quality_r_squared(self):
        """R² 应 > 0.8"""
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        result = optimizer.fit(
            de_seed=42, de_maxiter=500, de_popsize=15,
            local_maxiter=2000,
        )
        self.assertGreater(result.r_squared, 0.8,
                           f"R² {result.r_squared:.4f} 低于 0.8 阈值")

    def test_fit_parameters_physical(self):
        """拟合参数应在物理合理范围内"""
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        result = optimizer.fit(de_seed=42, de_maxiter=500, de_popsize=15)
        self.assertGreater(result.parameters.dg_12, -50000)
        self.assertLess(result.parameters.dg_12, 50000)
        self.assertGreater(result.parameters.dg_21, -50000)
        self.assertLess(result.parameters.dg_21, 50000)

    def test_fit_arrays_correct_length(self):
        """结果数组长度应与数据点数一致"""
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        result = optimizer.fit(de_seed=42, de_maxiter=200, de_popsize=10)
        self.assertEqual(len(result.x_calculated), 9)
        self.assertEqual(len(result.x_experimental), 9)
        self.assertEqual(len(result.temperatures), 9)
        self.assertEqual(len(result.residuals), 9)
        self.assertEqual(len(result.relative_deviations), 9)
        self.assertEqual(len(result.tau_12_values), 9)
        self.assertEqual(len(result.tau_21_values), 9)

    def test_fit_summary_not_empty(self):
        """summary() 应返回可读文本"""
        optimizer = NRTLOptimizer(
            dataset=self.dataset,
            constants=self.constants,
        )
        result = optimizer.fit(de_seed=42, de_maxiter=200, de_popsize=10)
        summary = result.summary()
        self.assertIn("NRTL", summary)
        self.assertIn("Δg₁₂", summary)
        self.assertIn("AARD", summary)


if __name__ == "__main__":
    unittest.main()

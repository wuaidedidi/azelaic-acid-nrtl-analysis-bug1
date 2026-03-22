"""
测试模块: 溶解度实验数据加载与校验
"""

import unittest

import numpy as np
from pydantic import ValidationError

from src.data.solubility import (
    SolubilityDataPoint,
    SolubilityDataset,
    ThermodynamicConstants,
)


class TestSolubilityDataPoint(unittest.TestCase):
    """测试单个数据点校验"""

    def test_valid_data_point(self):
        point = SolubilityDataPoint(temperature_k=298.15, mole_fraction=1.703e-4)
        self.assertAlmostEqual(point.temperature_k, 298.15)
        self.assertAlmostEqual(point.mole_fraction, 1.703e-4)

    def test_invalid_temperature_negative(self):
        with self.assertRaises(ValidationError):
            SolubilityDataPoint(temperature_k=-10.0, mole_fraction=0.001)

    def test_invalid_temperature_out_of_range(self):
        with self.assertRaises(ValidationError):
            SolubilityDataPoint(temperature_k=600.0, mole_fraction=0.001)

    def test_invalid_mole_fraction_zero(self):
        with self.assertRaises(ValidationError):
            SolubilityDataPoint(temperature_k=300.0, mole_fraction=0.0)

    def test_invalid_mole_fraction_one(self):
        with self.assertRaises(ValidationError):
            SolubilityDataPoint(temperature_k=300.0, mole_fraction=1.0)


class TestThermodynamicConstants(unittest.TestCase):
    """测试热力学常数校验"""

    def test_default_constants(self):
        c = ThermodynamicConstants()
        self.assertAlmostEqual(c.T_m, 379.65)
        self.assertAlmostEqual(c.delta_H_fus, 34500.0)
        self.assertAlmostEqual(c.R, 8.314)
        self.assertAlmostEqual(c.alpha, 0.3)

    def test_validate_passes(self):
        c = ThermodynamicConstants()
        c.validate()

    def test_invalid_melting_point(self):
        c = ThermodynamicConstants(T_m=-1.0)
        with self.assertRaises(ValueError):
            c.validate()

    def test_invalid_alpha(self):
        c = ThermodynamicConstants(alpha=1.5)
        with self.assertRaises(ValueError):
            c.validate()


class TestSolubilityDataset(unittest.TestCase):
    """测试数据集加载"""

    def test_load_default_size(self):
        dataset = SolubilityDataset.load_default()
        self.assertEqual(dataset.size, 9)

    def test_load_default_temperatures(self):
        dataset = SolubilityDataset.load_default()
        expected_temps = [283.15, 288.15, 293.15, 298.15, 303.15,
                          308.15, 313.15, 318.15, 323.15]
        np.testing.assert_array_almost_equal(dataset.temperatures, expected_temps)

    def test_load_default_mole_fractions(self):
        dataset = SolubilityDataset.load_default()
        expected_x = [7.140e-05, 9.189e-05, 1.476e-04, 1.703e-04,
                      2.186e-04, 2.891e-04, 3.502e-04, 4.974e-04, 9.849e-04]
        np.testing.assert_array_almost_equal(dataset.mole_fractions, expected_x, decimal=8)

    def test_temperatures_monotonically_increasing(self):
        dataset = SolubilityDataset.load_default()
        diffs = np.diff(dataset.temperatures)
        self.assertTrue(np.all(diffs > 0))

    def test_summary_not_empty(self):
        dataset = SolubilityDataset.load_default()
        summary = dataset.summary()
        self.assertIn("壬二酸", summary)
        self.assertIn("283.15", summary)


if __name__ == "__main__":
    unittest.main()

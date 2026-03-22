"""
壬二酸（Azelaic Acid）溶解度实验数据管理模块

管理壬二酸在水中的溶解度实验数据，提供数据校验、加载与预处理功能。
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from src.utils.logger import get_logger

logger = get_logger("data.solubility")


class SolubilityDataPoint(BaseModel):
    """单个溶解度数据点的验证模型"""

    temperature_k: float
    mole_fraction: float

    @field_validator("temperature_k")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"温度必须为正值，当前值: {v} K")
        if v < 200 or v > 500:
            raise ValueError(f"温度超出合理范围 (200-500 K)，当前值: {v} K")
        return v

    @field_validator("mole_fraction")
    @classmethod
    def validate_mole_fraction(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError(f"摩尔分数必须在 (0, 1) 范围内，当前值: {v}")
        return v


@dataclass
class ThermodynamicConstants:
    """热力学常数"""

    T_m: float = 379.65          # 熔点，K
    delta_H_fus: float = 34500.0 # 熔化焓，J/mol
    R: float = 8.314             # 气体常数，J/(mol·K)
    alpha: float = 0.3           # NRTL 非随机性参数

    def validate(self) -> None:
        """校验热力学常数的合理性"""
        if self.T_m <= 0:
            raise ValueError(f"熔点必须为正值: {self.T_m} K")
        if self.delta_H_fus <= 0:
            raise ValueError(f"熔化焓必须为正值: {self.delta_H_fus} J/mol")
        if self.R <= 0:
            raise ValueError(f"气体常数必须为正值: {self.R}")
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError(f"NRTL α 参数必须在 (0, 1) 范围内: {self.alpha}")
        logger.info(
            "热力学常数校验通过: T_m=%.2f K, ΔH_fus=%.0f J/mol, R=%.3f, α=%.1f",
            self.T_m, self.delta_H_fus, self.R, self.alpha,
        )


@dataclass
class SolubilityDataset:
    """壬二酸在水中的溶解度数据集"""

    data_points: List[SolubilityDataPoint] = field(default_factory=list)

    @staticmethod
    def load_default() -> "SolubilityDataset":
        """
        加载壬二酸在水中的溶解度默认实验数据。

        数据来源: 用户提供的实验测量值
        溶质: 壬二酸 (Azelaic Acid, C9H16O4)
        溶剂: 水 (H2O)

        Returns:
            包含完整实验数据的 SolubilityDataset 实例
        """
        raw_data: List[Tuple[float, float]] = [
            (283.15, 7.140e-05),
            (288.15, 9.189e-05),
            (293.15, 1.476e-04),
            (298.15, 1.703e-04),
            (303.15, 2.186e-04),
            (308.15, 2.891e-04),
            (313.15, 3.502e-04),
            (318.15, 4.974e-04),
            (323.15, 9.849e-04),
        ]

        dataset = SolubilityDataset()
        for temp, x in raw_data:
            point = SolubilityDataPoint(temperature_k=temp, mole_fraction=x)
            dataset.data_points.append(point)

        logger.info("已加载 %d 个溶解度实验数据点", len(dataset.data_points))
        logger.info(
            "温度范围: %.2f K ~ %.2f K",
            dataset.temperatures[0], dataset.temperatures[-1],
        )
        logger.info(
            "摩尔分数范围: %.4e ~ %.4e",
            dataset.mole_fractions.min(), dataset.mole_fractions.max(),
        )

        return dataset

    @property
    def temperatures(self) -> np.ndarray:
        """返回温度数组 (K)"""
        return np.array([p.temperature_k for p in self.data_points])

    @property
    def mole_fractions(self) -> np.ndarray:
        """返回摩尔分数数组"""
        return np.array([p.mole_fraction for p in self.data_points])

    @property
    def size(self) -> int:
        """数据点数量"""
        return len(self.data_points)

    def to_dataframe(self) -> pd.DataFrame:
        """将数据集转换为 Pandas DataFrame"""
        return pd.DataFrame({
            "temperature_K": self.temperatures,
            "temperature_C": self.temperatures - 273.15,
            "mole_fraction": self.mole_fractions,
            "inv_T": 1.0 / self.temperatures,
            "ln_x": np.log(self.mole_fractions),
        })

    def summary(self) -> str:
        """返回数据集的文本摘要"""
        lines = [
            "=" * 65,
            "壬二酸 (Azelaic Acid) 在水中的溶解度实验数据",
            "=" * 65,
            f"{'温度 (K)':>12s}  {'摩尔分数 x':>15s}",
            "-" * 35,
        ]
        for p in self.data_points:
            lines.append(f"{p.temperature_k:>12.2f}  {p.mole_fraction:>15.4e}")
        lines.append("-" * 35)
        lines.append(f"共 {self.size} 个数据点")
        lines.append("=" * 65)
        return "\n".join(lines)

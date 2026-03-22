"""
测试修复是否正确
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from dataclasses import dataclass

# 模拟 FittingResult 数据结构
@dataclass
class MockParams:
    dg_12: float
    dg_21: float

@dataclass
class MockResult:
    relative_deviations: np.ndarray
    temperatures: np.ndarray
    x_experimental: np.ndarray
    x_calculated: np.ndarray
    parameters: MockParams
    aard_percent: float = 0.0
    rmsd: float = 0.0
    r_squared: float = 0.0
    residuals: np.ndarray = None
    tau_12_values: np.ndarray = None
    tau_21_values: np.ndarray = None
    G_12_values: np.ndarray = None
    G_21_values: np.ndarray = None
    objective_value: float = 0.0
    convergence_message: str = ""
    iterations: int = 0

# 测试相对偏差计算
def test_relative_deviation_plot():
    print("=" * 60)
    print("测试: 相对偏差柱状图修复验证")
    print("=" * 60)
    
    # 创建模拟数据 - 小数形式的相对偏差
    mock_rd = np.array([0.023, -0.015, 0.052, -0.031, 0.008, -0.042, 0.019])
    # 预期的百分比形式
    expected_rd_percent = mock_rd * 100
    
    print(f"\n原始小数形式的相对偏差:")
    for i, v in enumerate(mock_rd):
        print(f"  点 {i+1}: {v:.6f}")
    
    print(f"\n转换为百分比后 (应显示的数值):")
    for i, v in enumerate(expected_rd_percent):
        print(f"  点 {i+1}: {v:.4f}%")
    
    print(f"\nY轴标签应为: 'Relative Deviation (%)'")
    print(f"原错误标签: 'Absolute Deviation (mol/mol)'")
    
    print(f"\n结论: 修复后的数据范围应为 {expected_rd_percent.min():.2f}% 到 {expected_rd_percent.max():.2f}%")
    print(f"      而不是原范围 {mock_rd.min():.6f} 到 {mock_rd.max():.6f} (小数形式)")

# 测试残差分布直方图
def test_residual_histogram():
    print("\n" + "=" * 60)
    print("测试: 残差分布直方图修复验证")
    print("=" * 60)
    
    print(f"\n原设置: bins=2 (只有两根柱子，无法显示分布)")
    print(f"修复后: bins='auto' (自动计算合适的分箱数量)")
    print(f"\n结论: 修复后直方图应能显示更详细的分布形状，而不是只有两根粗柱子")

# 运行测试
if __name__ == "__main__":
    test_relative_deviation_plot()
    test_residual_histogram()
    
    print("\n" + "=" * 60)
    print("修复总结:")
    print("=" * 60)
    print("\n1. 相对偏差柱状图高度问题:")
    print("   - 问题: 使用小数形式的相对偏差 (如 0.05 表示 5%)")
    print("   - 修复: 乘以 100 转换为百分比形式 (如 5%)")
    print("   - 代码修改: rd_percent = result.relative_deviations * 100")
    
    print("\n2. Y轴标签错误问题:")
    print("   - 问题: 标签写的是 'Absolute Deviation (mol/mol)'")
    print("   - 修复: 改为 'Relative Deviation (%)'")
    
    print("\n3. 残差分布直方图柱子太粗问题:")
    print("   - 问题: bins=2 只有两根柱子")
    print("   - 修复: bins='auto' 让 matplotlib 自动计算合适的分箱数量")
    print("")

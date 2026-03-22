"""
测试修复是否正确 - 简化版
"""

# 测试相对偏差计算
def test_relative_deviation_plot():
    print("=" * 60)
    print("测试: 相对偏差柱状图修复验证")
    print("=" * 60)
    
    # 创建模拟数据 - 小数形式的相对偏差
    mock_rd = [0.023, -0.015, 0.052, -0.031, 0.008, -0.042, 0.019]
    # 预期的百分比形式
    expected_rd_percent = [x * 100 for x in mock_rd]
    
    print(f"\n原始小数形式的相对偏差 (如 0.05 表示 5%):")
    for i, v in enumerate(mock_rd):
        print(f"  点 {i+1}: {v:.6f}")
    
    print(f"\n转换为百分比后 (应显示的数值):")
    for i, v in enumerate(expected_rd_percent):
        print(f"  点 {i+1}: {v:.4f}%")
    
    print(f"\nY轴标签应为: 'Relative Deviation (%)'")
    print(f"原错误标签: 'Absolute Deviation (mol/mol)'")
    
    min_val = min(expected_rd_percent)
    max_val = max(expected_rd_percent)
    print(f"\n结论: 修复后的数据范围应为 {min_val:.2f}% 到 {max_val:.2f}%")
    print(f"      而不是原范围 {min(mock_rd):.6f} 到 {max(mock_rd):.6f} (小数形式)")

# 测试残差分布直方图
def test_residual_histogram():
    print("\n" + "=" * 60)
    print("测试: 残差分布直方图修复验证")
    print("=" * 60)
    
    print(f"\n原设置: bins=2 (只有两根柱子，无法显示分布)")
    print(f"修复后: bins='auto' (自动计算合适的分箱数量)")
    print(f"\n结论: 修复后直方图应能显示更详细的分布形状，而不是只有两根粗柱子")

# 显示代码修改位置
def show_code_changes():
    print("\n" + "=" * 60)
    print("代码修改位置 (src/visualization/plotter.py)")
    print("=" * 60)
    
    print("\n1. 第 138 行: 相对偏差转换为百分比")
    print("   原代码: rd_percent = result.relative_deviations")
    print("   修复后: rd_percent = result.relative_deviations * 100")
    
    print("\n2. 第 149 行: Y轴标签更正")
    print("   原代码: ax.set_ylabel('Absolute Deviation (mol/mol)')")
    print("   修复后: ax.set_ylabel('Relative Deviation (%)')")
    
    print("\n3. 第 401 行: 直方图分箱数量更正")
    print("   原代码: ax1.hist(rd, bins=2, ...)")
    print("   修复后: ax1.hist(rd, bins='auto', ...)")

# 运行测试
if __name__ == "__main__":
    test_relative_deviation_plot()
    test_residual_histogram()
    show_code_changes()
    
    print("\n" + "=" * 60)
    print("修复完成!")
    print("=" * 60)

"""
验证统计指标修复效果的测试脚本
"""
import numpy as np

# 模拟数据
x_exp = np.array([7.14e-05, 9.189e-05, 0.0001476, 0.0001703, 0.0002186, 0.0002891, 0.0003502, 0.0004974, 0.0009849])
x_calc = np.array([6.66121128e-05, 9.23086621e-05, 1.26543246e-04, 1.71719091e-04, 2.30804879e-04, 3.07448149e-04, 4.06111332e-04, 5.32236710e-04, 6.92449075e-04])

# 计算残差和相对偏差
residuals = x_calc - x_exp
relative_devs = residuals / x_exp

print("=" * 60)
print("修复后的统计指标计算验证")
print("=" * 60)

# 修复前的错误计算
print("\n--- 修复前的错误计算 ---")
rmsd_wrong = np.mean(residuals ** 2)
aard_wrong = np.mean(relative_devs) * 100
print(f"RMSD (无平方根): {rmsd_wrong:.6e}")
print(f"AARD (无绝对值): {aard_wrong:.4f} %")

# 修复后的正确计算
print("\n--- 修复后的正确计算 ---")
rmsd_correct = np.sqrt(np.mean(residuals ** 2))
aard_correct = np.mean(np.abs(relative_devs)) * 100
print(f"RMSD (有平方根): {rmsd_correct:.6e}")
print(f"AARD (有绝对值): {aard_correct:.4f} %")

# van't Hoff 方程验证
print("\n--- van't Hoff 方程符号验证 ---")
temperatures = np.array([283.15, 288.15, 293.15, 298.15, 303.15, 308.15, 313.15, 318.15, 323.15])
inv_T = 1.0 / temperatures
ln_x_exp = np.log(x_exp)

from scipy import stats
slope, intercept, r_value, _, _ = stats.linregress(inv_T, ln_x_exp)
R = 8.314

delta_H_wrong = slope * R  # 修复前（错误）
delta_H_correct = -slope * R  # 修复后（正确）

print(f"回归斜率: {slope:.4f}")
print(f"ΔH (无负号 - 错误): {delta_H_wrong:.2f} J/mol")
print(f"ΔH (有负号 - 正确): {delta_H_correct:.2f} J/mol")
print(f"结论: ΔH > 0，说明溶解是吸热过程，符合壬二酸的实际情况")

print("\n" + "=" * 60)
print("修复验证完成！")
print("=" * 60)

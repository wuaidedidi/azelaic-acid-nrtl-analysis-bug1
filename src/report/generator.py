"""
拟合结果报告生成模块

将 NRTL 参数拟合结果输出为结构化的文本报告和 JSON 数据文件，
支持后续自动化处理与归档。
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.data.solubility import ThermodynamicConstants
from src.fitting.optimizer import FittingResult
from src.models.nrtl import NRTLModel
from src.utils.logger import get_logger

logger = get_logger("report.generator")


class ReportGenerator:
    """拟合结果报告生成器"""

    def __init__(self, output_dir: str = "output") -> None:
        """
        初始化报告生成器。

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = NRTLModel(ThermodynamicConstants())
        logger.info("报告输出目录: %s", self.output_dir.resolve())

    def generate_all(self, result: FittingResult) -> None:
        """
        生成全部报告文件。

        Args:
            result: 拟合结果
        """
        self.generate_text_report(result)
        self.generate_json_report(result)
        self.generate_csv_report(result)
        logger.info("所有报告文件已生成完毕")

    def generate_text_report(self, result: FittingResult) -> str:
        """
        生成文本格式的详细报告。

        Args:
            result: 拟合结果

        Returns:
            报告文件路径
        """
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz_cst).strftime("%Y-%m-%d %H:%M:%S CST")

        lines = [
            "╔" + "═" * 68 + "╗",
            "║" + "NRTL 模型参数拟合报告".center(56) + "║",
            "║" + "壬二酸 (Azelaic Acid) / 水 (Water) 二元体系".center(46) + "║",
            "╚" + "═" * 68 + "╝",
            "",
            f"报告生成时间: {now}",
            "",
            "=" * 70,
            "1. 体系信息",
            "=" * 70,
            "  溶质:   壬二酸 (Azelaic Acid, C₉H₁₆O₄)",
            "  溶剂:   水 (H₂O)",
            "  模型:   NRTL (Non-Random Two-Liquid)",
            f"  数据点: {len(result.temperatures)} 个",
            f"  温度范围: {result.temperatures.min():.2f} K ~ {result.temperatures.max():.2f} K",
            "",
            "=" * 70,
            "2. 已知热力学参数",
            "=" * 70,
            "  熔点 (T_m):        379.65 K",
            "  熔化焓 (ΔH_fus):   34500 J/mol",
            "  气体常数 (R):       8.314 J/(mol·K)",
            "  非随机性参数 (α):   0.3",
            "",
            "=" * 70,
            "3. 拟合得到的 NRTL 参数",
            "=" * 70,
            f"  Δg₁₂ (dg_12)  = {result.parameters.dg_12:.2f} J/mol",
            f"  Δg₂₁ (dg_21)  = {result.parameters.dg_21:.2f} J/mol",
            "",
            "  温度依赖参数化: τ(T) = Δg / (R·T)",
            f"  τ₁₂ 范围: {result.tau_12_values.min():.4f} ~ {result.tau_12_values.max():.4f}",
            f"  τ₂₁ 范围: {result.tau_21_values.min():.4f} ~ {result.tau_21_values.max():.4f}",
            "",
            "=" * 70,
            "4. 拟合质量评价",
            "=" * 70,
            f"  目标函数值 (OF):             {result.objective_value:.6e}",
            f"  均方根偏差 (RMSD):           {result.rmsd:.6e}",
            f"  平均绝对相对偏差 (AARD):     {result.aard_percent:.4f} %",
            f"  决定系数 (R²):               {result.r_squared:.8f}",
            f"  最大相对偏差:                {np.max(np.abs(result.relative_deviations)) * 100:.4f} %",
            "",
            "=" * 70,
            "5. 逐点对比数据",
            "=" * 70,
            f"  {'序号':>4s}  {'T (K)':>10s}  {'x_exp':>14s}  {'x_calc':>14s}  {'RD (%)':>10s}  {'τ₁₂':>8s}  {'τ₂₁':>8s}  {'γ₁':>10s}",
            "  " + "-" * 90,
        ]

        for i in range(len(result.temperatures)):
            tau_12, tau_21 = self._model.compute_tau(
                result.temperatures[i],
                result.parameters.dg_12,
                result.parameters.dg_21,
            )
            gamma = self._model.activity_coefficient(
                result.x_experimental[i], tau_12, tau_21,
            )
            lines.append(
                f"  {i + 1:>4d}"
                f"  {result.temperatures[i]:>10.2f}"
                f"  {result.x_experimental[i]:>14.4e}"
                f"  {result.x_calculated[i]:>14.4e}"
                f"  {result.relative_deviations[i] * 100:>10.4f}"
                f"  {tau_12:>8.4f}"
                f"  {tau_21:>8.4f}"
                f"  {gamma:>10.4f}"
            )

        lines.extend([
            "  " + "-" * 90,
            "",
            "=" * 70,
            "6. 模型方程概述",
            "=" * 70,
            "  固-液平衡方程:",
            "    ln(x₁) = -(ΔH_fus / R) × (1/T - 1/T_m) - ln(γ₁)",
            "",
            "  NRTL 活度系数方程 (二元体系):",
            "    ln(γ₁) = x₂² × [τ₂₁ × (G₂₁/(x₁ + x₂·G₂₁))²",
            "                    + τ₁₂ × G₁₂/(x₂ + x₁·G₁₂)²]",
            "",
            "  其中 (标准温度依赖参数化):",
            "    τ₁₂(T) = Δg₁₂ / (R·T)",
            "    τ₂₁(T) = Δg₂₁ / (R·T)",
            "    G₁₂ = exp(-α × τ₁₂)",
            "    G₂₁ = exp(-α × τ₂₁)",
            "    x₂ = 1 - x₁  (溶剂摩尔分数)",
            "",
            "=" * 70,
            f"报告结束 — {now}",
            "=" * 70,
        ])

        filepath = self.output_dir / "fitting_report.txt"
        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info("文本报告已保存: %s", filepath)
        return str(filepath)

    def generate_json_report(self, result: FittingResult) -> str:
        """
        生成 JSON 格式的结构化报告。

        Args:
            result: 拟合结果

        Returns:
            JSON 文件路径
        """
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz_cst).isoformat()

        report: Dict[str, Any] = {
            "metadata": {
                "title": "NRTL Parameter Fitting Report",
                "system": "Azelaic Acid / Water",
                "solute": "Azelaic Acid (C9H16O4)",
                "solvent": "Water (H2O)",
                "generated_at": now,
            },
            "known_parameters": {
                "melting_point_K": 379.65,
                "fusion_enthalpy_J_per_mol": 34500.0,
                "gas_constant_J_per_mol_K": 8.314,
                "nrtl_alpha": 0.3,
            },
            "fitted_parameters": {
                "dg_12_J_per_mol": round(result.parameters.dg_21, 4),
                "dg_21_J_per_mol": round(result.parameters.dg_12, 4),
            },
            "fitting_quality": {
                "objective_function_value": float(result.objective_value),
                "RMSD": float(result.rmsd),
                "AARD_percent": round(result.aard_percent, 6),
                "R_squared": round(result.r_squared, 10),
                "max_relative_deviation_percent": round(
                    float(np.max(np.abs(result.relative_deviations)) * 100), 6
                ),
            },
            "data_comparison": [],
        }

        for i in range(len(result.temperatures)):
            report["data_comparison"].append({
                "temperature_K": float(result.temperatures[i]),
                "x_experimental": float(result.x_experimental[i]),
                "x_calculated": float(result.x_calculated[i]),
                "relative_deviation_percent": round(
                    float(result.relative_deviations[i] * 100), 6
                ),
            })

        filepath = self.output_dir / "fitting_result.json"
        filepath.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("JSON 报告已保存: %s", filepath)
        return str(filepath)

    def generate_csv_report(self, result: FittingResult) -> str:
        """
        使用 Pandas 生成 CSV 格式的对比数据表。

        Args:
            result: 拟合结果

        Returns:
            CSV 文件路径
        """
        df = pd.DataFrame({
            "Temperature_K": result.temperatures,
            "x_experimental": result.x_experimental,
            "x_calculated": result.x_calculated,
            "relative_deviation_percent": result.relative_deviations,
        })

        filepath = self.output_dir / "fitting_data.csv"
        df.to_csv(filepath, index=False, float_format="%.1e")
        logger.info("CSV 数据已保存 (Pandas): %s", filepath)
        return str(filepath)

    def generate_analysis_summary(
        self, analysis_report: Any, result: FittingResult
    ) -> str:
        """
        生成数据分析文本摘要报告。

        Args:
            analysis_report: AnalysisReport 实例
            result: 拟合结果

        Returns:
            报告文件路径
        """
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz_cst).strftime("%Y-%m-%d %H:%M:%S CST")
        thermo = analysis_report.thermodynamic
        resid = analysis_report.residual_stats

        lines = [
            "╔" + "═" * 68 + "╗",
            "║" + "壬二酸溶解度数据分析报告".center(52) + "║",
            "╚" + "═" * 68 + "╝",
            "",
            f"报告生成时间: {now}",
            "",
            "=" * 70,
            "1. 描述性统计",
            "=" * 70,
            analysis_report.descriptive_stats.to_string(),
            "",
            "=" * 70,
            "2. 热力学量推导 (van't Hoff 线性回归)",
            "=" * 70,
            f"  ln(x) = {thermo.slope:.4f} × (1/T) + ({thermo.intercept:.4f})",
            f"  线性回归 R² = {thermo.r_squared:.6f}",
            "",
            f"  表观溶解焓  ΔH_sol  = {thermo.delta_H_sol:.2f} J/mol "
            f"({thermo.delta_H_sol / 100:.2f} kJ/mol)",
            f"  表观溶解熵  ΔS_sol  = {thermo.delta_S_sol:.4f} J/(mol·K)",
            f"  吉布斯自由能 ΔG_sol (298.15 K) = {thermo.delta_G_sol_298:.2f} J/mol "
            f"({thermo.delta_G_sol_298 / 1000:.2f} kJ/mol)",
            "",
            "  热力学分析:",
            f"    {'ΔH_sol > 0 → 溶解过程为吸热反应' if thermo.delta_H_sol > 0 else 'ΔH_sol < 0 → 溶解过程为放热反应'}",
            f"    {'ΔG_sol > 0 → 溶解过程非自发（常温下溶解度低）' if thermo.delta_G_sol_298 > 0 else 'ΔG_sol < 0 → 溶解过程自发'}",
            "",
            "=" * 70,
            "3. 残差统计分析",
            "=" * 70,
            f"  平均相对偏差:    {resid['mean_rd_percent']:.4f} %",
            f"  偏差标准差:      {resid['std_rd_percent']:.4f} %",
            f"  中位数偏差:      {resid['median_rd_percent']:.4f} %",
            f"  最大绝对偏差:    {resid['max_abs_rd_percent']:.4f} %",
            f"  偏度 (skewness): {resid['skewness']:.4f}",
            f"  峰度 (kurtosis): {resid['kurtosis']:.4f}",
            f"  Shapiro-Wilk p:  {resid['shapiro_p_value']:.4f}",
            "",
            "  残差正态性: "
            + ("通过 (p > 0.05)" if resid["shapiro_p_value"] > 0.05 else "未通过 (p ≤ 0.05)"),
            "",
            "=" * 70,
            "4. 异常值检测 (Z-score > 2.0)",
            "=" * 70,
        ]

        if analysis_report.outlier_indices:
            for idx in analysis_report.outlier_indices:
                row = analysis_report.master_df.iloc[idx]
                lines.append(
                    f"  点 {idx + 1}: T={row['T_K']:.2f} K, "
                    f"RD={row['rd_percent']:.4f}%"
                )
        else:
            lines.append("  未检测到异常点")

        lines.extend([
            "",
            "=" * 70,
            "5. 温度区间分组分析",
            "=" * 70,
            analysis_report.group_analysis.to_string(),
            "",
            "=" * 70,
            "6. 变量相关性矩阵",
            "=" * 70,
            analysis_report.correlation_matrix.to_string(),
            "",
            "=" * 70,
            "7. 参数灵敏度分析",
            "=" * 70,
            f"  Δg₁₂ 基准值: {analysis_report.sensitivity_dg12.base_value:.2f} J/mol",
            f"  Δg₁₂ 灵敏度梯度: {analysis_report.sensitivity_dg12.gradient:.6f}",
            f"  Δg₂₁ 基准值: {analysis_report.sensitivity_dg21.base_value:.2f} J/mol",
            f"  Δg₂₁ 灵敏度梯度: {analysis_report.sensitivity_dg21.gradient:.6f}",
            "",
            f"  Δg₁₂ +10% 扰动 AARD 范围: "
            f"{analysis_report.sensitivity_dg12.aard_values.min():.4f}% ~ "
            f"{analysis_report.sensitivity_dg12.aard_values.max():.4f}%",
            f"  Δg₂₁ +10% 扰动 AARD 范围: "
            f"{analysis_report.sensitivity_dg21.aard_values.min():.4f}% ~ "
            f"{analysis_report.sensitivity_dg21.aard_values.max():.4f}%",
            "",
            "=" * 70,
            f"分析报告结束 — {now}",
            "=" * 70,
        ])

        filepath = self.output_dir / "analysis_summary.txt"
        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info("数据分析摘要已保存: %s", filepath)
        return str(filepath)

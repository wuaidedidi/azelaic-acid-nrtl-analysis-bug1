"""
结果可视化模块

生成 NRTL 模型拟合结果的专业图表，包括:
1. 溶解度拟合对比图 (实验值 vs 计算值)
2. 相对偏差分布图
3. van't Hoff 图 (ln(x) vs 1/T)
4. 综合仪表板
5. 相关性热力图 (Pandas 分析)
6. 参数灵敏度图 (Pandas 分析)
7. 残差分布直方图 (Pandas 分析)
8. 分析综合仪表板 (Pandas 分析)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.solubility import ThermodynamicConstants
from src.fitting.optimizer import FittingResult
from src.models.nrtl import NRTLModel
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.analysis.analyzer import AnalysisReport

logger = get_logger("visualization.plotter")

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


class ResultPlotter:
    """拟合结果可视化器"""

    def __init__(self, output_dir: str = "output/figures") -> None:
        """
        初始化可视化器。

        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = NRTLModel(ThermodynamicConstants())
        logger.info("图表输出目录: %s", self.output_dir.resolve())

    def plot_all(self, result: FittingResult) -> None:
        """
        生成所有图表。

        Args:
            result: 拟合结果
        """
        self.plot_solubility_comparison(result)
        self.plot_relative_deviation(result)
        self.plot_vant_hoff(result)
        self.plot_combined_dashboard(result)
        logger.info("所有图表已生成完毕")

    def plot_solubility_comparison(self, result: FittingResult) -> str:
        """
        绘制溶解度拟合对比图。

        Args:
            result: 拟合结果

        Returns:
            图表保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.scatter(
            result.temperatures, result.x_experimental,
            marker="o", s=100, c="#E74C3C", edgecolors="black",
            linewidth=1.2, zorder=5, label="Experimental Data",
        )

        T_smooth = np.linspace(
            result.temperatures.min() - 3,
            result.temperatures.max() + 3,
            200,
        )
        x_smooth = self._model.calculate_solubility_batch(
            T_smooth, result.parameters.dg_12, result.parameters.dg_21
        )

        ax.plot(
            T_smooth, x_smooth,
            "-", color="#2E86C1", linewidth=2.5, label="NRTL Model Fit",
        )

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Mole Fraction $x_1$")
        ax.set_title(
            "Azelaic Acid Solubility in Water — NRTL Model Fitting\n"
            f"$\\Delta g_{{12}}$ = {result.parameters.dg_12:.1f} J/mol, "
            f"$\\Delta g_{{21}}$ = {result.parameters.dg_21:.1f} J/mol, "
            f"AARD = {result.aard_percent:.2f}%"
        )
        ax.legend(loc="upper left", framealpha=0.9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -3))

        filepath = self.output_dir / "solubility_comparison.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("溶解度对比图已保存: %s", filepath)
        return str(filepath)

    def plot_relative_deviation(self, result: FittingResult) -> str:
        """
        绘制相对偏差分布图。

        Args:
            result: 拟合结果

        Returns:
            图表保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        rd_percent = result.relative_deviations

        colors = ["#27AE60" if abs(v) < 10 else "#E74C3C" for v in rd_percent]
        bars = ax.bar(
            range(len(result.temperatures)), rd_percent,
            color=colors, edgecolor="black", linewidth=0.8, width=0.6,
        )

        ax.set_xticks(range(len(result.temperatures)))
        ax.set_xticklabels([f"{T:.1f}" for T in result.temperatures], rotation=45)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Absolute Deviation (mol/mol)")
        ax.set_title("Relative Deviation of NRTL Model Fitting")
        ax.axhline(y=0, color="black", linewidth=1.0)
        ax.axhline(y=10, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="±10% threshold")
        ax.axhline(y=-10, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend()

        filepath = self.output_dir / "relative_deviation.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("相对偏差图已保存: %s", filepath)
        return str(filepath)

    def plot_vant_hoff(self, result: FittingResult) -> str:
        """
        绘制 van't Hoff 图 (ln(x) vs 1000/T)。

        Args:
            result: 拟合结果

        Returns:
            图表保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        inv_T = 100.0 / result.temperatures
        ln_x_exp = np.log(result.x_experimental)
        ln_x_calc = np.log(result.x_calculated)

        ax.scatter(
            inv_T, ln_x_exp,
            marker="s", s=100, c="#E74C3C", edgecolors="black",
            linewidth=1.2, zorder=5, label="Experimental",
        )

        T_smooth = np.linspace(
            result.temperatures.min() - 3,
            result.temperatures.max() + 3,
            200,
        )
        x_smooth = self._model.calculate_solubility_batch(
            T_smooth, result.parameters.dg_12, result.parameters.dg_21
        )
        inv_T_smooth = 100.0 / T_smooth
        ln_x_smooth = np.log(np.clip(x_smooth, 1e-15, None))

        ax.plot(
            inv_T_smooth, ln_x_smooth,
            "-", color="#2E86C1", linewidth=2.5, label="NRTL Model",
        )

        ax.set_xlabel("$1000 / T$ (K$^{-1}$)")
        ax.set_ylabel("$\\ln(x_1)$")
        ax.set_title("van't Hoff Plot — Azelaic Acid / Water System")
        ax.legend(loc="upper right", framealpha=0.9)

        filepath = self.output_dir / "vant_hoff_plot.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("van't Hoff 图已保存: %s", filepath)
        return str(filepath)

    def plot_combined_dashboard(self, result: FittingResult) -> str:
        """
        绘制综合仪表板 (2x2 子图)。

        Args:
            result: 拟合结果

        Returns:
            图表保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "NRTL Parameter Fitting Dashboard — Azelaic Acid / Water",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # (0,0) 溶解度对比
        ax = axes[0, 0]
        ax.scatter(
            result.temperatures, result.x_experimental,
            marker="o", s=80, c="#E74C3C", edgecolors="black",
            linewidth=1, zorder=5, label="Exp.",
        )
        ax.plot(
            result.temperatures, result.x_calculated,
            "s--", color="#2E86C1", markersize=7, linewidth=1.5,
            label="NRTL Calc.",
        )
        ax.set_xlabel("T (K)")
        ax.set_ylabel("$x_1$")
        ax.set_title("Solubility Comparison")
        ax.legend(fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -3))

        # (0,1) 相对偏差
        ax = axes[0, 1]
        rd_percent = result.relative_deviations * 100
        ax.bar(
            range(len(result.temperatures)), rd_percent,
            color="#3498DB", edgecolor="black", linewidth=0.5, width=0.6,
        )
        ax.set_xticks(range(len(result.temperatures)))
        ax.set_xticklabels([f"{T:.0f}" for T in result.temperatures], rotation=45, fontsize=9)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("RD (%)")
        ax.set_title("Relative Deviation")
        ax.axhline(y=0, color="black", linewidth=0.8)

        # (1,0) van't Hoff 图
        ax = axes[1, 0]
        inv_T = 100.0 / result.temperatures
        ax.scatter(
            inv_T, np.log(result.x_experimental),
            marker="s", s=80, c="#E74C3C", edgecolors="black",
            linewidth=1, zorder=5, label="Exp.",
        )
        ax.plot(
            inv_T, np.log(result.x_calculated),
            "o--", color="#2E86C1", markersize=6, linewidth=1.5,
            label="NRTL",
        )
        ax.set_xlabel("$1000/T$ (K$^{-1}$)")
        ax.set_ylabel("$\\ln(x_1)$")
        ax.set_title("van't Hoff Plot")
        ax.legend(fontsize=9)
        ax.invert_xaxis()

        # (1,1) 拟合指标摘要
        ax = axes[1, 1]
        ax.axis("off")
        info_text = (
            f"{'━' * 35}\n"
            f"  NRTL Fitting Summary\n"
            f"{'━' * 35}\n\n"
            f"  Solute:  Azelaic Acid (C₉H₁₆O₄)\n"
            f"  Solvent: Water (H₂O)\n\n"
            f"  Δg₁₂ = {result.parameters.dg_12:.2f} J/mol\n"
            f"  Δg₂₁ = {result.parameters.dg_21:.2f} J/mol\n"
            f"  α   = 0.3 (fixed)\n\n"
            f"  RMSD  = {result.rmsd:.4e}\n"
            f"  AARD  = {result.aard_percent:.4f} %\n"
            f"  R²    = {result.r_squared:.8f}\n\n"
            f"  T_m      = 379.65 K\n"
            f"  ΔH_fus   = 34500 J/mol\n"
            f"{'━' * 35}"
        )
        ax.text(
            0.1, 0.95, info_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#BDC3C7"),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filepath = self.output_dir / "fitting_dashboard.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("综合仪表板已保存: %s", filepath)
        return str(filepath)

    # ──────────────────────────────────────────────────
    # 数据分析图表 (基于 Pandas AnalysisReport)
    # ──────────────────────────────────────────────────

    def plot_analysis_all(self, report: AnalysisReport) -> None:
        """生成全部分析图表"""
        self.plot_correlation_heatmap(report)
        self.plot_sensitivity(report)
        self.plot_residual_distribution(report)
        self.plot_analysis_dashboard(report)
        logger.info("所有分析图表已生成完毕")

    def plot_correlation_heatmap(self, report: AnalysisReport) -> str:
        """绘制变量间 Pearson 相关性热力图"""
        corr = report.correlation_matrix
        fig, ax = plt.subplots(figsize=(9, 8))

        im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)

        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

        fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        ax.set_title("Correlation Heatmap — Solubility Analysis Variables")

        filepath = self.output_dir / "correlation_heatmap.png"
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info("相关性热力图已保存: %s", filepath)
        return str(filepath)

    def plot_sensitivity(self, report: AnalysisReport) -> str:
        """绘制参数灵敏度分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        s12 = report.sensitivity_dg12
        ax1.plot(
            s12.perturbations * 100, s12.aard_values,
            "o-", color="#E74C3C", linewidth=2, markersize=5,
        )
        ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax1.axhline(y=s12.aard_values[len(s12.aard_values) // 2], color="gray",
                     linestyle=":", alpha=0.5)
        ax1.set_xlabel("$\\Delta g_{12}$ Perturbation (%)")
        ax1.set_ylabel("AARD (%)")
        ax1.set_title(
            f"Sensitivity of $\\Delta g_{{12}}$\n"
            f"(base = {s12.base_value:.1f} J/mol)"
        )

        s21 = report.sensitivity_dg21
        ax2.plot(
            s21.perturbations * 100, s21.aard_values,
            "s-", color="#2E86C1", linewidth=2, markersize=5,
        )
        ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax2.axhline(y=s21.aard_values[len(s21.aard_values) // 2], color="gray",
                     linestyle=":", alpha=0.5)
        ax2.set_xlabel("$\\Delta g_{21}$ Perturbation (%)")
        ax2.set_ylabel("AARD (%)")
        ax2.set_title(
            f"Sensitivity of $\\Delta g_{{21}}$\n"
            f"(base = {s21.base_value:.1f} J/mol)"
        )

        fig.suptitle(
            "NRTL Parameter Sensitivity Analysis (±10%)", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filepath = self.output_dir / "parameter_sensitivity.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("灵敏度分析图已保存: %s", filepath)
        return str(filepath)

    def plot_residual_distribution(self, report: AnalysisReport) -> str:
        """绘制残差分布直方图与正态拟合"""
        rd = report.master_df["rd_percent"].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.hist(rd, bins=2, color="#3498DB", edgecolor="black",
                 alpha=0.7, density=True, label="Observed")
        mu, sigma = rd.mean(), rd.std()
        if sigma > 0:
            x_fit = np.linspace(rd.min() - 1, rd.max() + 1, 100)
            from scipy.stats import norm
            ax1.plot(x_fit, norm.pdf(x_fit, mu, sigma), "r-", linewidth=2,
                     label=f"Normal($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})")
        ax1.set_xlabel("Relative Deviation (%)")
        ax1.set_ylabel("Density")
        ax1.set_title("Residual Distribution")
        ax1.legend(fontsize=9)

        from scipy import stats as sp_stats
        sp_stats.probplot(rd, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal)")
        ax2.get_lines()[0].set(marker="o", color="#E74C3C", markersize=8)
        ax2.get_lines()[1].set(color="#2E86C1", linewidth=2)

        fig.suptitle(
            "Residual Analysis — Distribution & Normality",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filepath = self.output_dir / "residual_distribution.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("残差分布图已保存: %s", filepath)
        return str(filepath)

    def plot_analysis_dashboard(self, report: AnalysisReport) -> str:
        """绘制数据分析综合仪表板 (2x2)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Data Analysis Dashboard — Azelaic Acid / Water",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # (0,0) 温度-溶解度散点 + van't Hoff 回归线
        ax = axes[0, 0]
        df = report.master_df
        ax.scatter(df["inv_T"], df["ln_x_exp"], s=80, c="#E74C3C",
                   edgecolors="black", zorder=5, label="Experimental")
        t = report.thermodynamic
        x_line = np.linspace(df["inv_T"].min(), df["inv_T"].max(), 100)
        y_line = t.slope * x_line + t.intercept
        ax.plot(x_line, y_line, "--", color="#2E86C1", linewidth=2,
                label=f"Linear fit (R²={t.r_squared:.4f})")
        ax.set_xlabel("1/T (K$^{-1}$)")
        ax.set_ylabel("ln($x$)")
        ax.set_title("van't Hoff Regression")
        ax.legend(fontsize=9)

        # (0,1) 温度分组箱线图
        ax = axes[0, 1]
        groups_data = []
        group_labels = []
        for name, grp in df.dropna(subset=["temp_group"]).groupby("temp_group", observed=True):
            groups_data.append(grp["abs_rd_percent"].values)
            group_labels.append(str(name))
        if groups_data:
            bp = ax.boxplot(groups_data, labels=group_labels, patch_artist=True)
            colors_box = ["#AED6F1", "#A9DFBF", "#F9E79F"]
            for patch, c in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
                patch.set_facecolor(c)
        ax.set_xlabel("Temperature Group")
        ax.set_ylabel("|Relative Deviation| (%)")
        ax.set_title("Fitting Quality by Temperature Group")

        # (1,0) 灵敏度对比
        ax = axes[1, 0]
        s12 = report.sensitivity_dg12
        s21 = report.sensitivity_dg21
        ax.plot(s12.perturbations * 100, s12.aard_values, "o-",
                color="#E74C3C", linewidth=1.5, markersize=4, label="$\\Delta g_{12}$")
        ax.plot(s21.perturbations * 100, s21.aard_values, "s-",
                color="#2E86C1", linewidth=1.5, markersize=4, label="$\\Delta g_{21}$")
        ax.set_xlabel("Perturbation (%)")
        ax.set_ylabel("AARD (%)")
        ax.set_title("Parameter Sensitivity Comparison")
        ax.legend(fontsize=9)

        # (1,1) 分析指标摘要
        ax = axes[1, 1]
        ax.axis("off")
        resid = report.residual_stats
        info_text = (
            f"{'━' * 40}\n"
            f"  Data Analysis Summary\n"
            f"{'━' * 40}\n\n"
            f"  ▸ Thermodynamic Quantities\n"
            f"    ΔH_sol = {t.delta_H_sol / 1000:.2f} kJ/mol\n"
            f"    ΔS_sol = {t.delta_S_sol:.2f} J/(mol·K)\n"
            f"    ΔG_sol(298K) = {t.delta_G_sol_298 / 1000:.2f} kJ/mol\n\n"
            f"  ▸ Residual Statistics\n"
            f"    Mean RD = {resid['mean_rd_percent']:.4f}%\n"
            f"    Std RD  = {resid['std_rd_percent']:.4f}%\n"
            f"    Skewness = {resid['skewness']:.4f}\n"
            f"    Shapiro p = {resid['shapiro_p_value']:.4f}\n\n"
            f"  ▸ Outliers: {len(report.outlier_indices)} point(s)\n"
            f"  ▸ Data Points: {len(report.master_df)}\n"
            f"{'━' * 40}"
        )
        ax.text(
            0.05, 0.95, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#BDC3C7"),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filepath = self.output_dir / "analysis_dashboard.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("分析综合仪表板已保存: %s", filepath)
        return str(filepath)

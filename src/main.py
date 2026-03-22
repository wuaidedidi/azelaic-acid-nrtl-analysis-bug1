"""
壬二酸 (Azelaic Acid) 在水中 NRTL 模型参数拟合与数据分析 — 主入口

完整工作流程:
1. 加载并校验实验溶解度数据
2. 初始化 NRTL 模型与优化器
3. 执行两阶段参数拟合 (差分进化 + L-BFGS-B)
4. 生成拟合可视化图表
5. Pandas 数据分析 (统计/热力学推导/灵敏度/异常检测)
6. 生成分析图表 (热力图/灵敏度/残差分布)
7. 输出详细报告 (TXT / JSON / CSV)
"""

import sys
import time

from src.analysis.analyzer import SolubilityAnalyzer
from src.data.solubility import SolubilityDataset, ThermodynamicConstants
from src.fitting.optimizer import NRTLOptimizer
from src.report.generator import ReportGenerator
from src.utils.logger import setup_logger, get_logger
from src.visualization.plotter import ResultPlotter


def main() -> int:
    """
    主函数：执行壬二酸/水体系 NRTL 模型参数拟合全流程。

    Returns:
        0 表示成功，1 表示失败
    """
    root_logger = setup_logger()
    logger = get_logger("main")

    logger.info("=" * 70)
    logger.info("壬二酸 (Azelaic Acid) / 水 (Water) 体系")
    logger.info("NRTL 模型参数拟合程序启动")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # ──────────────────────────────────────────────
        # Step 1: 加载实验数据
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 1: 加载实验数据 ━━━")
        dataset = SolubilityDataset.load_default()
        logger.info("\n%s", dataset.summary())

        # ──────────────────────────────────────────────
        # Step 2: 初始化热力学常数
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 2: 初始化热力学常数 ━━━")
        constants = ThermodynamicConstants(
            T_m=379.65,
            delta_H_fus=34500.0,
            R=8.314,
            alpha=0.3,
        )
        constants.validate()

        # ──────────────────────────────────────────────
        # Step 3: 执行参数拟合
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 3: 执行 NRTL 参数拟合 ━━━")
        optimizer = NRTLOptimizer(
            dataset=dataset,
            constants=constants,
            dg_bounds=((-50000.0, 50000.0), (-50000.0, 50000.0)),
        )
        result = optimizer.fit(
            de_seed=42,
            de_maxiter=1000,
            de_tol=1e-12,
            de_popsize=25,
            local_maxiter=5000,
            local_tol=1e-14,
        )

        logger.info(result.summary())

        # ──────────────────────────────────────────────
        # Step 4: 生成可视化图表
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 4: 生成可视化图表 ━━━")
        plotter = ResultPlotter(output_dir="output/figures")
        plotter.plot_all(result)

        # ──────────────────────────────────────────────
        # Step 5: 数据分析 (Pandas)
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 5: 数据分析 (Pandas) ━━━")
        analyzer = SolubilityAnalyzer(
            dataset=dataset, result=result, constants=constants,
        )
        analysis_report = analyzer.run_full_analysis()
        analyzer.export_master_csv("output")
        analyzer.export_analysis_report("output")

        logger.info("  DataFrame 形状: %s", str(analysis_report.master_df.shape))
        logger.info(
            "  热力学推导: ΔH_sol=%.2f J/mol, ΔS_sol=%.4f J/(mol·K)",
            analysis_report.thermodynamic.delta_H_sol,
            analysis_report.thermodynamic.delta_S_sol,
        )

        # ──────────────────────────────────────────────
        # Step 6: 生成分析图表
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 6: 生成分析图表 ━━━")
        plotter.plot_analysis_all(analysis_report)

        # ──────────────────────────────────────────────
        # Step 7: 生成报告
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 7: 生成报告 ━━━")
        reporter = ReportGenerator(output_dir="output")
        reporter.generate_all(result)
        reporter.generate_analysis_summary(analysis_report, result)

        elapsed = time.time() - start_time

        logger.info("")
        logger.info("=" * 70)
        logger.info("拟合与分析完成！总耗时: %.2f 秒", elapsed)
        logger.info("=" * 70)
        logger.info("拟合结果:")
        logger.info("  Δg₁₂ = %.2f J/mol", result.parameters.dg_12)
        logger.info("  Δg₂₁ = %.2f J/mol", result.parameters.dg_21)
        logger.info("  τ₁₂ 范围: %.4f ~ %.4f", result.tau_12_values.min(), result.tau_12_values.max())
        logger.info("  τ₂₁ 范围: %.4f ~ %.4f", result.tau_21_values.min(), result.tau_21_values.max())
        logger.info("  AARD = %.4f %%", result.aard_percent)
        logger.info("  R²   = %.8f", result.r_squared)
        logger.info("")
        logger.info("数据分析结果:")
        logger.info("  ΔH_sol = %.2f kJ/mol", analysis_report.thermodynamic.delta_H_sol / 1000)
        logger.info("  ΔS_sol = %.2f J/(mol·K)", analysis_report.thermodynamic.delta_S_sol)
        logger.info("  ΔG_sol(298K) = %.2f kJ/mol", analysis_report.thermodynamic.delta_G_sol_298 / 1000)
        logger.info("  异常点数: %d", len(analysis_report.outlier_indices))
        logger.info("")
        logger.info("输出文件:")
        logger.info("  报告:     output/fitting_report.txt")
        logger.info("  分析摘要: output/analysis_summary.txt")
        logger.info("  数据:     output/fitting_result.json")
        logger.info("  分析JSON: output/analysis_report.json")
        logger.info("  CSV:      output/fitting_data.csv")
        logger.info("  主数据表: output/analysis_master_data.csv")
        logger.info("  图表:     output/figures/")
        logger.info("  日志:     output/logs/nrtl_fitting.log")
        logger.info("=" * 70)

        return 0

    except ValueError as e:
        logger.error("数据校验错误: %s", e)
        return 1
    except RuntimeError as e:
        logger.error("运行时错误: %s", e)
        return 1
    except Exception as e:
        logger.error("未预期的错误: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

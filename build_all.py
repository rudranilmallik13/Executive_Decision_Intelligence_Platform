"""
EDIP - Build Everything
==========================
Runs the full pipeline end-to-end, in order:

    1. Generate synthetic datasets              (data/generate_data.py)
    2. Build the feature store                  (pipeline/build_feature_store.py)
    3. Train revenue & demand forecast models    (models/train_revenue_demand_forecast.py)
    4. Train churn & CLV models                  (models/train_churn_clv.py)
    5. Build the RAG vector index                (rag/rag_pipeline.py)
    6. Smoke-test the decision engine            (decision_engine/optimizer.py)
    7. Smoke-test the scenario simulator          (decision_engine/scenario_simulator.py)
    8. Generate a sample executive PPTX report   (reports/generate_executive_report.py)

Run this once after cloning/unzipping the project:
    python build_all.py

Then explore interactively:
    python agent/ai_consultant.py "Why did revenue drop last quarter?"
    python agent/ai_consultant.py "Which customers are likely to churn?"
    python agent/ai_consultant.py "What pricing strategy should we use for Europe?"
    python agent/ai_consultant.py "What will happen if inflation increases by 3%?"
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DATA_FILES = [
    ROOT / "data" / "sales_transactions.csv",
    ROOT / "data" / "finance_statements.csv",
    ROOT / "data" / "crm_customers.csv",
    ROOT / "data" / "erp_inventory.csv",
    ROOT / "data" / "weather_daily.csv",
    ROOT / "data" / "competitor_prices.csv",
    ROOT / "data" / "market_news_sentiment.csv",
    ROOT / "data" / "docs" / "annual_report_2025.txt",
    ROOT / "data" / "docs" / "pricing_policy.txt",
    ROOT / "data" / "docs" / "q2_2026_board_meeting_notes.txt",
    ROOT / "data" / "docs" / "market_research_summary.txt",
    ROOT / "data" / "docs" / "supplier_policy.txt",
]
FEATURE_STORE_FILES = [
    ROOT / "feature_store" / "monthly_region_features.parquet",
    ROOT / "feature_store" / "customer_features.parquet",
    ROOT / "feature_store" / "inventory_features.parquet",
]
MODEL_ARTIFACTS = [
    ROOT / "models" / "artifacts" / "revenue_forecast_model.joblib",
    ROOT / "models" / "artifacts" / "demand_forecast_model.joblib",
    ROOT / "models" / "artifacts" / "churn_model.joblib",
    ROOT / "models" / "artifacts" / "clv_model.joblib",
    ROOT / "models" / "artifacts" / "customer_scores.csv",
]
RAG_FILES = [
    ROOT / "rag" / "index" / "faiss.index",
    ROOT / "rag" / "index" / "embedder.joblib",
    ROOT / "rag" / "index" / "chunks.json",
]
REPORT_FILE = ROOT / "outputs" / "EDIP_Executive_Report.pptx"

STEPS = [
    ("Generating synthetic datasets", [PYTHON, "data/generate_data.py"], DATA_FILES),
    ("Building feature store", [PYTHON, "pipeline/build_feature_store.py"], FEATURE_STORE_FILES),
    ("Training revenue & demand forecast models", [PYTHON, "models/train_revenue_demand_forecast.py"], MODEL_ARTIFACTS[:2]),
    ("Training churn & CLV models", [PYTHON, "models/train_churn_clv.py"], MODEL_ARTIFACTS[2:]),
    ("Building RAG vector index", [PYTHON, "rag/build_index.py"], RAG_FILES),
    ("Smoke-testing decision engine", [PYTHON, "decision_engine/optimizer.py"], []),
    ("Smoke-testing scenario simulator", [PYTHON, "decision_engine/scenario_simulator.py"], []),
    ("Generating sample executive PPTX report", [PYTHON, "reports/generate_executive_report.py"], [REPORT_FILE]),
]


def all_exist(paths):
    return all(path.exists() for path in paths)


def run_step(label, cmd):
    print(f"\n>>> {label} ...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[FAILED] step '{label}' exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)
    print(f"[OK] ({time.time() - t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Build EDIP artifacts and preserve existing generated data unless --force is used."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild everything from scratch, including raw synthetic data.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Regenerate raw synthetic data even if it already exists.",
    )
    parser.add_argument(
        "--force-feature-store",
        action="store_true",
        help="Rebuild feature store files even if they already exist.",
    )
    parser.add_argument(
        "--force-models",
        action="store_true",
        help="Retrain ML models even if artifacts already exist.",
    )
    parser.add_argument(
        "--force-rag",
        action="store_true",
        help="Rebuild the RAG vector index even if it already exists.",
    )
    parser.add_argument(
        "--force-report",
        action="store_true",
        help="Regenerate the executive PPTX report even if it already exists.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("EDIP - Executive Decision Intelligence Platform: full build")
    print("=" * 70)

    for label, cmd, outputs in STEPS:
        if args.force:
            need_step = True
        elif label == "Generating synthetic datasets":
            need_step = args.force_data or not all_exist(outputs)
        elif label == "Building feature store":
            need_step = args.force_feature_store or not all_exist(outputs)
        elif label == "Training revenue & demand forecast models":
            need_step = args.force_models or not all_exist(outputs)
        elif label == "Training churn & CLV models":
            need_step = args.force_models or not all_exist(outputs)
        elif label == "Building RAG vector index":
            need_step = args.force_rag or not all_exist(outputs)
        elif label == "Generating sample executive PPTX report":
            need_step = args.force_report or not all_exist(outputs)
        else:
            need_step = True

        if need_step:
            run_step(label, cmd)
        else:
            print(f"\n>>> Skipping {label} (already built)")

    print("\n" + "=" * 70)
    print("Build complete. If you want to launch the dashboard, run:")
    print("  streamlit run app.py")
    print("or execute the one-step runner:")
    print("  python run.py --serve")
    print("Sample executive report: outputs/EDIP_Executive_Report.pptx")
    print("=" * 70)


if __name__ == "__main__":
    main()

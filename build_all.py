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
    python3 build_all.py

Then explore interactively:
    python3 agent/ai_consultant.py "Why did revenue drop last quarter?"
    python3 agent/ai_consultant.py "Which customers are likely to churn?"
    python3 agent/ai_consultant.py "What pricing strategy should we use for Europe?"
    python3 agent/ai_consultant.py "What will happen if inflation increases by 3%?"
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("Generating synthetic datasets", [PYTHON, "data/generate_data.py"]),
    ("Building feature store", [PYTHON, "pipeline/build_feature_store.py"]),
    ("Training revenue & demand forecast models", [PYTHON, "models/train_revenue_demand_forecast.py"]),
    ("Training churn & CLV models", [PYTHON, "models/train_churn_clv.py"]),
    ("Building RAG vector index", [PYTHON, "rag/build_index.py"]),
    ("Smoke-testing decision engine", [PYTHON, "decision_engine/optimizer.py"]),
    ("Smoke-testing scenario simulator", [PYTHON, "decision_engine/scenario_simulator.py"]),
    ("Generating sample executive PPTX report", [PYTHON, "reports/generate_executive_report.py"]),
]


def main():
    print("=" * 70)
    print("EDIP - Executive Decision Intelligence Platform: full build")
    print("=" * 70)
    for label, cmd in STEPS:
        print(f"\n>>> {label} ...")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[FAILED] step '{label}' exited with code {result.returncode}. Stopping.")
            sys.exit(result.returncode)
        print(f"[OK] ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 70)
    print("Build complete. Try:")
    print('  python3 agent/ai_consultant.py "Why did revenue drop last quarter?"')
    print('  python3 agent/ai_consultant.py "Which customers are likely to churn?"')
    print('  python3 agent/ai_consultant.py "What pricing strategy should we use for Europe?"')
    print('  python3 agent/ai_consultant.py "What will happen if inflation increases by 3%?"')
    print("Sample executive report: outputs/EDIP_Executive_Report.pptx")
    print("=" * 70)


if __name__ == "__main__":
    main()

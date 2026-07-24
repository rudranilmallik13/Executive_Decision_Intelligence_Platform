# EDIP — Executive Decision Intelligence Platform

An AI consultant that analyzes company data, predicts business outcomes, recommends
strategic actions, and explains its reasoning with supporting evidence.

Ask it things like:
- *"Why did revenue drop last quarter?"*
- *"Which customers are likely to churn?"*
- *"Which products should be discontinued?"*
- *"How much inventory should we order?"*
- *"What pricing strategy should we use?"*
- *"What will happen if inflation increases by 3%?"*
- *"What should we do next?"*

Instead of a dashboard, you get a data-backed answer with numbers, model
explanations (SHAP), and citations back to the source documents.

---

## 1. Quick start

```bash
pip install -r requirements.txt
python build_all.py
```

`build_all.py` runs the entire pipeline once, in order: generates synthetic data →
builds the feature store → trains all ML models → builds the RAG index → smoke-tests
the optimizer and scenario simulator → generates a sample executive PPTX report.
Takes about 30–60 seconds on a laptop.

Then launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Use the dashboard to ask business questions and get structured, explainable answers.

For CLI users, the existing agent is still available:

```bash
python agent/ai_consultant.py "Why did revenue drop last quarter?"
python agent/ai_consultant.py "Which customers are likely to churn?"
python agent/ai_consultant.py "Which products should be discontinued?"
python agent/ai_consultant.py "How much inventory should we order for Europe?"
python agent/ai_consultant.py "What pricing strategy should we use for Europe?"
python agent/ai_consultant.py "What will happen if inflation increases by 3%?"
python agent/ai_consultant.py "What should we do next?"
```

Each command prints a structured JSON answer: quantitative findings, model
explanation (SHAP), supporting document citations, and a plain-English narrative.

A ready-made executive slide deck is generated at
`outputs/EDIP_Executive_Report.pptx`.

---

## 2. What's actually implemented (not just diagrammed)

| Layer | Implementation | Where |
|---|---|---|
| **Data** | 7 synthetic structured sources (147K+ sales transactions, 4,000 customers, ERP inventory, finance, weather, competitor prices, macro sentiment) + 5 unstructured company documents, all internally consistent with a baked-in "Q2 2026 revenue drop" storyline | `data/` |
| **Data pipeline / Feature Store** | Joins, lag features, rolling averages, seasonality features | `pipeline/build_feature_store.py` |
| **ML Models** | XGBoost regression (revenue & demand forecast), XGBoost classification (churn), XGBoost regression (CLV) | `models/` |
| **Vector DB / RAG** | FAISS flat index + TF-IDF embeddings, chunking, retrieval with relevance scores and source attribution | `rag/` |
| **Decision Engine** | EOQ inventory optimization, price-elasticity profit optimization, LP marketing budget allocation, MIP supplier selection with diversification constraints (all via PuLP) | `decision_engine/optimizer.py` |
| **Scenario Simulator** | Re-runs the *actual trained models* under perturbed macro/price/competitor inputs — not a static percentage assumption | `decision_engine/scenario_simulator.py` |
| **Explainable AI** | SHAP-based per-prediction explanations, confidence scoring from held-out model metrics | `decision_engine/explainability.py` |
| **Autonomous Agent** | Natural-language router that plans which tools to call (SQL/feature-store query, ML model, optimizer, RAG) and synthesizes a cited, structured answer | `agent/ai_consultant.py` |
| **Executive Reporting** | Auto-generated PowerPoint deck with KPI cards and prioritized action items | `reports/generate_executive_report.py` |

### Trained model performance (from the last `build_all.py` run)

| Model | Metric | Value |
|---|---|---|
| Revenue forecast (XGBoost) | R² / MAPE | 0.95 / ~4.4% |
| Demand forecast (XGBoost) | R² / MAPE | 0.95 / ~4.0% |
| Churn prediction (XGBoost) | AUC / F1 | 0.79 / 0.41 |
| CLV regression (XGBoost) | R² | 0.81 |

(Exact numbers vary slightly by random seed/environment; re-run `build_all.py` to
regenerate `models/artifacts/*_metrics.json`.)

---

## 3. Architecture

```
 ERP / CRM / Sales / Finance / Weather / Competitor Prices / Market News
                              │
                              ▼
                        Data Pipeline  (pipeline/build_feature_store.py)
                              │
                              ▼
                        Feature Store  (feature_store/*.parquet)
                              │
              ┌───────────────┼────────────────────────┐
              ▼                                         ▼
        ML Models                                Company Documents
   (models/*.py, XGBoost)                  (annual reports, policies,
   • Demand Forecast                        board notes, market research)
   • Revenue Forecast                                    │
   • Churn Prediction                                     ▼
   • CLV                                           RAG Pipeline (rag/)
              │                              embedding (TF-IDF) → FAISS → retrieval
              │                                            │
              └───────────────┬────────────────────────────┘
                               ▼
                    Autonomous AI Consultant Agent
                        (agent/ai_consultant.py)
                    routes NL question → calls the
                    right tool(s) → synthesizes answer
                               │
                               ▼
                       Decision Engine
              (decision_engine/optimizer.py,
               decision_engine/scenario_simulator.py)
              EOQ · price optimization · LP budget
              allocation · MIP supplier selection ·
              what-if scenario re-simulation
                               │
                               ▼
                       Explainable AI Layer
                  (decision_engine/explainability.py)
                  SHAP values · confidence scores
                               │
                               ▼
              Executive Dashboard / PPTX Report
              (reports/generate_executive_report.py)
```

---

## 4. Project layout

```
EDIP/
├── build_all.py                        # one-command full pipeline build
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py                # synthetic data generator (run first)
│   ├── docs/                           # unstructured docs for RAG (generated)
│   └── *.csv                           # structured sources (generated)
├── pipeline/
│   └── build_feature_store.py          # ingestion + feature engineering
├── feature_store/                      # model-ready tables (generated)
├── models/
│   ├── train_revenue_demand_forecast.py
│   ├── train_churn_clv.py
│   └── artifacts/                      # trained models + metrics (generated)
├── rag/
│   ├── rag_pipeline.py                 # RAGPipeline class (chunk/embed/retrieve)
│   ├── build_index.py                  # entrypoint to build the FAISS index
│   └── index/                          # FAISS index + chunks (generated)
├── decision_engine/
│   ├── optimizer.py                    # EOQ, pricing, budget, supplier MIP/LP
│   ├── scenario_simulator.py           # what-if simulation via trained models
│   └── explainability.py               # SHAP explanation cards
├── agent/
│   └── ai_consultant.py                # natural-language orchestrator (the "brain")
├── reports/
│   └── generate_executive_report.py    # PPTX executive summary generator
└── outputs/
    └── EDIP_Executive_Report.pptx      # generated sample report
```

---

## 5. Design notes & what would change in a real production deployment

This is a fully working reference implementation designed to run **offline, for
free, with no external API keys**, so every module is the "real" algorithm class
(XGBoost, FAISS, PuLP MIP/LP, SHAP) but two components are deliberately
lightweight stand-ins that would be swapped in production:

1. **Embeddings**: `rag/rag_pipeline.py`'s `TfidfEmbedder` is a drop-in
   placeholder for a real embedding API (OpenAI `text-embedding-3`, Voyage,
   Sentence-Transformers). The `fit`/`transform` interface and the FAISS index
   code do not need to change — only the embedder class.
2. **Intent routing**: `agent/ai_consultant.py`'s keyword-based `route()` method
   is a transparent, deterministic stand-in for an LLM tool-calling loop (GPT-4 /
   Claude / Gemini). See the `route_with_llm()` stub at the bottom of that file
   for exactly how to wire in a real LLM — the underlying tools it would call
   (forecast models, optimizer, RAG, SHAP explainer) are unchanged either way.

Also worth knowing: real production data pipelines would run in Airflow/Dagster
against a warehouse (Snowflake/BigQuery) rather than flat CSVs, and model
training/serving would run via MLflow + a feature-store service (Feast/Tecton)
rather than local Parquet files — the modeling and business logic in this
repo (`models/`, `decision_engine/`) would carry over largely unchanged.

---

## 6. Regenerating everything from scratch

```bash
rm -rf feature_store models/artifacts rag/index outputs/*.pptx data/*.csv data/docs
python3 build_all.py
```

This is fully deterministic (fixed random seeds) except for the exact PuLP/CBC
solver timing.

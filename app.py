import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from agent.ai_consultant import AIConsultant

try:
    from reports.generate_executive_report import build_presentation_bytes
    PPTX_AVAILABLE = True
except ImportError:
    build_presentation_bytes = None
    PPTX_AVAILABLE = False

st.set_page_config(
    page_title="EDIP Executive AI Dashboard",
    page_icon="📊",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "Why did revenue drop last quarter?",
    "Which customers are likely to churn?",
    "Which products should be discontinued?",
    "How much inventory should we order for Europe?",
    "What pricing strategy should we use for Europe?",
    "What will happen if inflation increases by 3%?",
    "What should we do next?",
]

@st.cache_resource(show_spinner=False)
def load_consultant():
    try:
        return AIConsultant()
    except Exception as exc:
        return exc


def render_answer(answer):
    st.subheader("Answer")
    st.write(answer.get("narrative", "No narrative available."))

    if answer.get("quantitative_findings"):
        st.subheader("Quantitative findings")
        st.json(answer["quantitative_findings"])

    if answer.get("trend"):
        st.subheader("Recent trend")
        trend_df = pd.DataFrame(answer["trend"])
        st.line_chart(
            trend_df.set_index("month")["revenue"].rename("Revenue"),
            height=300,
        )
        st.line_chart(
            trend_df.set_index("month")["units_sold"].rename("Units Sold"),
            height=300,
        )

    if answer.get("summary"):
        st.subheader("Summary")
        st.json(answer["summary"])

    if answer.get("churn_probability_distribution"):
        st.subheader("Churn probability distribution")
        churn_df = pd.DataFrame(answer["churn_probability_distribution"])
        st.bar_chart(churn_df.set_index("bucket"), height=320)

    if answer.get("recommended_actions"):
        st.subheader("Recommended actions")
        cols = st.columns(min(len(answer["recommended_actions"]), 3))
        for idx, action in enumerate(answer["recommended_actions"]):
            col = cols[idx % len(cols)]
            col.markdown(f"**Priority {action['priority']}**")
            col.markdown(f"**{action['action']}**")
            col.markdown(f"- *Owner:* {action['owner']}  \n- *Rationale:* {action['rationale']}")

        if PPTX_AVAILABLE and build_presentation_bytes is not None:
            try:
                ppt_bytes = build_presentation_bytes(answer)
                st.subheader("Download executive PPT")
                st.download_button(
                    "Download PPTX report",
                    data=ppt_bytes,
                    file_name=f"EDIP_{answer.get('answer_type','analysis')}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            except Exception as exc:
                st.warning(f"Unable to generate PPT download: {exc}")
        else:
            st.warning("PPT download is unavailable because python-pptx is not installed. Install it with 'pip install python-pptx'.")

    if answer.get("marketing_allocation"):
        st.subheader("Marketing budget allocation")
        alloc_df = pd.DataFrame(answer["marketing_allocation"]).copy()
        st.bar_chart(alloc_df.set_index("region")["allocated_budget"], height=320)
        st.dataframe(alloc_df)

    if answer.get("supplier_allocation"):
        st.subheader("Supplier allocation")
        supplier_df = pd.DataFrame(answer["supplier_allocation"]).copy()
        st.bar_chart(supplier_df.set_index("supplier")["allocated_volume"], height=320)
        st.dataframe(supplier_df)

    if answer.get("price_scenarios"):
        st.subheader("Price scenario analysis")
        price_df = pd.DataFrame(answer["price_scenarios"]).copy()
        price_df["price_change_pct"] = price_df["price_change_pct"].astype(float)
        price_df = price_df.sort_values("price_change_pct")
        if "estimated_profit" in price_df.columns and "estimated_revenue" in price_df.columns:
            st.line_chart(
                price_df.set_index("price_change_pct")[["estimated_profit", "estimated_revenue"]],
                height=320,
            )
        st.dataframe(price_df)

    for key in [
        "top_at_risk_customers",
        "discontinuation_candidates",
        "all_products_ranked",
        "recommendations",
    ]:
        if answer.get(key):
            st.subheader(key.replace("_", " ").title())
            st.dataframe(answer[key])

    if answer.get("supporting_evidence"):
        st.subheader("Supporting evidence")
        for evidence in answer["supporting_evidence"]:
            st.markdown(
                f"**{evidence.get('source', 'source unknown')}**  \n"
                f"{evidence.get('excerpt', '')}  \n"
                f"_Relevance: {evidence.get('relevance_score', evidence.get('score', 'n/a'))}_"
            )

    with st.expander("Raw JSON output"):
        st.json(answer)


def main():
    st.title("EDIP Executive AI Dashboard")
    st.markdown(
        "Use this dashboard to ask business questions of the Executive Decision Intelligence "
        "Platform. The answer combines structured model analysis, optimization, scenario simulation, "
        "and company-document retrieval into a single, explainable response."
    )

    consultant = load_consultant()
    if isinstance(consultant, Exception):
        st.error("Required artifacts are missing or incomplete. Build the project before asking questions.")
        st.markdown(
            "Use `python build_all.py` or `python run.py --serve` to generate the required data, features, models, and index. "
            "If you want to force a rebuild, run `python build_all.py --force`."
        )
        st.exception(consultant)
        return

    with st.form(key="question_form"):
        selected = st.selectbox("Choose an example question", EXAMPLE_QUESTIONS)
        question = st.text_area("Or enter your own question", value=selected, height=140)
        submitted = st.form_submit_button("Run analysis")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
            return

        with st.spinner("Generating answer..."):
            try:
                answer = consultant.route(question)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
                return

        render_answer(answer)

    st.sidebar.header("Quick start")
    st.sidebar.write("1. Build the project with `python build_all.py` or use `python run.py --serve`.")
    st.sidebar.write("2. Run this app with `streamlit run app.py`, or launch directly with `python run.py --serve`.")
    st.sidebar.write("3. Ask any executive-level business question and review the dashboard output.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualizations added")
    st.sidebar.markdown("- Revenue trend line charts for the last 6 months\n- Churn probability distribution bar chart\n- Pricing scenario profit/revenue chart\n- Marketing budget allocation chart\n- Supplier allocation chart")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from agent.ai_consultant import AIConsultant

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
    return AIConsultant()


def render_answer(answer):
    st.subheader("Answer")
    st.write(answer.get("narrative", "No narrative available."))

    if answer.get("quantitative_findings"):
        st.subheader("Quantitative findings")
        st.json(answer["quantitative_findings"])

    if answer.get("summary"):
        st.subheader("Summary")
        st.json(answer["summary"])

    if answer.get("recommended_actions"):
        st.subheader("Recommended actions")
        for action in answer["recommended_actions"]:
            st.markdown(
                f"**Priority {action['priority']}: {action['action']}**  \n"
                f"- Rationale: {action['rationale']}  \n"
                f"- Owner: {action['owner']}"
            )

    for key in [
        "price_scenarios",
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
    st.sidebar.write("1. Build the project with `python build_all.py` if you have not already done so.")
    st.sidebar.write("2. Run this app with `streamlit run streamlit_app.py`.")
    st.sidebar.write("3. Ask any executive-level business question and review the dashboard output.")


if __name__ == "__main__":
    main()

"""Streamlit front end for the AgenticRAGCrewAI fishery assistant."""

from __future__ import annotations

import streamlit as st

from process import result_to_text, run_workflow


def main() -> None:
    st.set_page_config(page_title="AgenticRAGCrewAI Fishery Assistant", layout="wide")
    st.title("AgenticRAGCrewAI Fishery Assistant")

    st.markdown(
        "Ask a question about Lake Pyhäjärvi fishery, catch trends, economic value, "
        "or environmental relationships. The assistant will run a CrewAI workflow "
        "using the local analytics tools and processed data files."
    )

    question = st.text_input(
        "Ask your question:",
        "Is crayfish more valuable than fish? When did it become the most important species?",
    )

    if st.button("Run analysis"):
        if not question.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Running agentic workflow..."):
            result = run_workflow(question.strip())

        st.subheader("Final Answer")
        st.write(result_to_text(result))

        st.subheader("Raw CrewAI Result")
        st.text(str(result))


if __name__ == "__main__":
    main()

"""The "Ask the data" tab: chat UI plus the tool-calling agent loop."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import agent_tools
import llm

# Ceiling on tool round-trips for a single question. Reaching it means the model
# is looping, so we stop and say so rather than spending the user's quota.
MAX_ITERATIONS = 8

EXAMPLE_QUESTIONS = [
    "Which vendor has the worst average downtime?",
    "Show me ticket volume trend by month",
    "What's our SLA compliance rate?",
    "Compare FCR rate across zones",
]


def _init_state() -> None:
    st.session_state.setdefault("chat_log", [])       # what is displayed
    st.session_state.setdefault("llm_history", None)  # provider-native history
    st.session_state.setdefault("pending_question", None)


def _render_artifact(artifact: dict, key: str) -> None:
    """Draw one chart, table or KPI card produced during the agent loop."""
    kind = artifact.get("kind")
    if kind == "chart":
        st.plotly_chart(
            artifact["fig"], width="stretch", key=key, config={"displaylogo": False}
        )
    elif kind == "table":
        if artifact.get("title"):
            st.caption(artifact["title"])
        st.dataframe(artifact["data"], width="stretch", hide_index=True)
    elif kind == "kpi":
        st.metric(artifact["label"], artifact["value"], delta=artifact.get("delta"))


def _render_log() -> None:
    """Replay the conversation, including the charts each answer produced."""
    for m_index, message in enumerate(st.session_state.chat_log):
        with st.chat_message(message["role"]):
            for a_index, artifact in enumerate(message.get("artifacts", [])):
                _render_artifact(artifact, key=f"artifact_{m_index}_{a_index}")
            if message.get("text"):
                st.markdown(message["text"])
            if message.get("sql"):
                with st.expander(f"SQL ({len(message['sql'])} queries)"):
                    for query in message["sql"]:
                        st.code(query, language="sql")


def _run_agent(provider, df: pd.DataFrame, con, question: str) -> dict:
    """Drive the tool-calling loop until the model answers with text only.

    Returns the assistant message to append to the chat log.
    """
    system_prompt = agent_tools.system_prompt(df)

    if st.session_state.llm_history is None:
        st.session_state.llm_history = provider.start(system_prompt)
    history = st.session_state.llm_history
    history.append({"role": "user", "content": question})

    artifacts: list[dict] = []
    executed_sql: list[str] = []
    status_area = st.empty()

    for iteration in range(MAX_ITERATIONS):
        try:
            turn = provider.call(history, agent_tools.TOOLS, system_prompt)
        except Exception as exc:
            status_area.empty()
            return {
                "role": "assistant",
                "text": f"The model call failed: `{type(exc).__name__}: {exc}`",
                "artifacts": artifacts,
                "sql": executed_sql,
            }

        provider.append_assistant(history, turn)

        if not turn.wants_tools:
            status_area.empty()
            return {
                "role": "assistant",
                "text": turn.text or "_(no answer returned)_",
                "artifacts": artifacts,
                "sql": executed_sql,
            }

        # Execute every tool the model asked for this turn.
        results = []
        for call in turn.tool_calls:
            if call.name == "query_data":
                sql = str(call.arguments.get("sql", "")).strip()
                if sql:
                    executed_sql.append(sql)
                status_area.caption(f"Querying the data… (step {iteration + 1})")
            else:
                status_area.caption(f"Building the {call.name.split('_')[-1]}…")

            content, is_error = agent_tools.execute_tool(
                call.name, call.arguments, con, artifacts
            )
            results.append((call, content, is_error))

        provider.append_tool_results(history, results)

    status_area.empty()
    return {
        "role": "assistant",
        "text": (
            f"I stopped after {MAX_ITERATIONS} tool calls without settling on an "
            "answer. Try narrowing the question."
        ),
        "artifacts": artifacts,
        "sql": executed_sql,
    }


@st.fragment
def render(df: pd.DataFrame, con) -> None:
    """Draw the chat tab. `df` and `con` are the full, unfiltered dataset."""
    _init_state()

    try:
        provider = llm.resolve_provider()
    except llm.ProviderError as exc:
        st.error(str(exc))
        st.info(
            "Copy `.env.example` to `.env`, put your key in it, and restart "
            "`streamlit run app.py`."
        )
        return

    header, reset = st.columns([5, 1])
    with header:
        st.caption(
            f"Answers come from live SQL over all {len(df):,} tickets — the "
            "sidebar filters apply to the Dashboard tab only. Ask for a "
            "filtered slice in your question instead."
        )
    with reset:
        if st.button("Clear chat", width="stretch"):
            st.session_state.chat_log = []
            st.session_state.llm_history = None
            st.rerun(scope="fragment")

    _render_log()

    # Example prompts, to get a demo moving quickly.
    if not st.session_state.chat_log:
        st.write("**Try one of these:**")
        for column, question in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
            if column.button(question, width="stretch", key=f"eg_{question}"):
                st.session_state.pending_question = question
                st.rerun(scope="fragment")

    typed = st.chat_input("Ask a question about the tickets…")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if not question:
        return

    st.session_state.chat_log.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {provider.label}…"):
            message = _run_agent(provider, df, con, question)

    st.session_state.chat_log.append(message)
    # Re-run so the new answer renders through the same path as the history,
    # which keeps the plotly chart keys stable.
    st.rerun(scope="fragment")

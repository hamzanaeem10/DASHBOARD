"""The "Ask the data" tab: chat UI plus the tool-calling agent loop."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import agent_tools
import llm

# Ceiling on tool round-trips for a single question. Reaching it means the model
# is looping, so we stop and say so rather than spending the user's quota.
MAX_ITERATIONS = 8

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "📡"

# (question, one-line hint) - the hint sets expectation for what kind of answer
# is coming, so the demo reads as intentional rather than a random pick.
EXAMPLE_QUESTIONS = [
    ("Which vendor has the worst average downtime?", "🏆 ranking"),
    ("Show me ticket volume trend by month", "📈 trend"),
    ("What's our SLA compliance rate?", "🎯 headline number"),
    ("Compare FCR rate across zones", "📊 comparison"),
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


def _render_message(message: dict, m_index: int) -> None:
    """Draw one message, including any charts/KPIs and a query/timing footer."""
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        for a_index, artifact in enumerate(message.get("artifacts", [])):
            _render_artifact(artifact, key=f"artifact_{m_index}_{a_index}")
        if message.get("text"):
            st.markdown(message["text"])

        sql = message.get("sql")
        elapsed = message.get("elapsed_s")
        if sql or elapsed:
            footer = st.container()
            cols = footer.columns([3, 1])
            if sql:
                with cols[0]:
                    label = f"🔎 {len(sql)} quer{'y' if len(sql) == 1 else 'ies'} run"
                    with st.expander(label):
                        for i, query in enumerate(sql, 1):
                            if len(sql) > 1:
                                st.caption(f"Query {i}")
                            st.code(query, language="sql")
            if elapsed:
                cols[-1].caption(f"⏱ {elapsed:.1f}s")


def _render_log() -> None:
    """Replay the conversation, including the charts each answer produced."""
    for m_index, message in enumerate(st.session_state.chat_log):
        _render_message(message, m_index)


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
    started = time.monotonic()

    def _finish(text: str) -> dict:
        status_area.empty()
        return {
            "role": "assistant",
            "text": text,
            "artifacts": artifacts,
            "sql": executed_sql,
            "elapsed_s": time.monotonic() - started,
        }

    for iteration in range(MAX_ITERATIONS):
        try:
            turn = provider.call(history, agent_tools.TOOLS, system_prompt)
        except Exception as exc:
            return _finish(f"⚠️ The model call failed: `{type(exc).__name__}: {exc}`")

        provider.append_assistant(history, turn)

        if not turn.wants_tools:
            return _finish(turn.text or "_(no answer returned)_")

        # Execute every tool the model asked for this turn.
        results = []
        for call in turn.tool_calls:
            if call.name == "query_data":
                sql = str(call.arguments.get("sql", "")).strip()
                if sql:
                    executed_sql.append(sql)
                status_area.caption(f"🔎 Querying the data… (step {iteration + 1})")
            elif call.name == "render_chart":
                status_area.caption("📊 Building the chart…")
            else:
                status_area.caption("🧮 Computing the headline number…")

            content, is_error = agent_tools.execute_tool(
                call.name, call.arguments, con, artifacts
            )
            results.append((call, content, is_error))

        provider.append_tool_results(history, results)

    return _finish(
        f"I stopped after {MAX_ITERATIONS} tool calls without settling on an "
        "answer. Try narrowing the question."
    )


def _render_empty_state(provider_label: str) -> None:
    """Welcome card shown before the first message - sets up the demo flow."""
    with st.container(border=True):
        st.markdown("**Ask a question in plain English — I'll write the SQL.**")
        st.caption(
            f"Powered by {provider_label}, querying live over the full dataset "
            "with DuckDB. Try one of these, or type your own below."
        )
        top = st.columns(2, gap="small")
        bottom = st.columns(2, gap="small")
        for column, (question, hint) in zip(top + bottom, EXAMPLE_QUESTIONS):
            with column:
                if st.button(question, width="stretch", key=f"eg_{question}"):
                    st.session_state.pending_question = question
                    st.rerun(scope="fragment")
                st.caption(hint)


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
            "filtered slice in your question instead, e.g. \"…in the Central zone\"."
        )
    with reset:
        if st.button("🗑️ Clear", width="stretch", disabled=not st.session_state.chat_log):
            st.session_state.chat_log = []
            st.session_state.llm_history = None
            st.rerun(scope="fragment")

    _render_log()

    if not st.session_state.chat_log:
        _render_empty_state(provider.label)

    typed = st.chat_input("Ask a question about the tickets…")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if not question:
        return

    user_message = {"role": "user", "text": question}
    st.session_state.chat_log.append(user_message)
    _render_message(user_message, len(st.session_state.chat_log) - 1)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner(f"Thinking with {provider.label}…"):
            message = _run_agent(provider, df, con, question)

    st.session_state.chat_log.append(message)
    # Re-run so the new answer renders through the same path as the history,
    # which keeps the plotly chart keys stable.
    st.rerun(scope="fragment")

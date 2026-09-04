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

# --- Presentation -----------------------------------------------------------
#
# Streamlit's default chat chrome is a flat, full-width row per message, which
# makes a long analytical transcript hard to skim. The CSS below turns each turn
# into a bubble - assistant left on a card, user right on a tinted one - so the
# eye can find the answers without reading. Everything is driven off the same
# tokens as theme.py, so the chat and the charts inside it agree.
#
# Role is not exposed on the DOM node when a custom avatar is used, so
# _render_message emits a hidden marker span and the rules match it with :has().

_CSS = """
<style>
:root {
  --chat-ink: #0b0b0b;
  --chat-ink-2: #52514e;
  --chat-muted: #898781;
  --chat-line: #e6e5dd;
  --chat-primary: #2a78d6;
  --chat-primary-dark: #1c5cab;
}

/* The hidden role marker (and nothing else) collapses away completely. */
.stElementContainer:has(.chat-role) { display: none !important; }

/* ---- Header -------------------------------------------------------- */
.chat-head { display: flex; align-items: center; gap: 10px; margin: 2px 0 6px; }
.chat-head h3 {
  margin: 0; font-size: 17px; font-weight: 640; letter-spacing: -0.01em;
  color: var(--chat-ink);
}
.chat-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px;
  background: rgba(42, 120, 214, 0.09);
  border: 1px solid rgba(42, 120, 214, 0.22);
  color: var(--chat-primary-dark);
  font-size: 11.5px; font-weight: 600; letter-spacing: 0.01em;
}
.chat-pill::before {
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: #0ca30c; box-shadow: 0 0 0 3px rgba(12, 163, 12, 0.16);
}
.chat-note {
  font-size: 12.5px; line-height: 1.5; color: var(--chat-muted);
  max-width: 68ch; margin: 0 0 2px;
}

/* ---- Message rows -------------------------------------------------- */
[data-testid="stChatMessage"] {
  background: transparent !important;
  padding: 0 !important;
  gap: 12px;
  margin-bottom: 16px;
  align-items: flex-start;
}
[data-testid="stChatMessage"] > [data-testid^="stChatMessageAvatar"] {
  width: 36px; height: 36px; flex: 0 0 36px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50%; font-size: 16px; line-height: 1;
  background: #ffffff; border: 1px solid var(--chat-line);
  box-shadow: 0 1px 2px rgba(11, 11, 11, 0.06);
}

/* Assistant: a quiet card that can hold a chart without feeling boxed in. */
[data-testid="stChatMessage"]:has(.chat-role-assistant)
  > [data-testid^="stChatMessageAvatar"] {
  background: linear-gradient(145deg, #3d8ae8, #1c5cab);
  border-color: var(--chat-primary-dark);
  box-shadow: 0 2px 7px rgba(42, 120, 214, 0.3);
}
[data-testid="stChatMessage"]:has(.chat-role-assistant)
  [data-testid="stChatMessageContent"] {
  background: #ffffff;
  border: 1px solid var(--chat-line);
  border-radius: 5px 18px 18px 18px;
  padding: 15px 18px 13px;
  box-shadow: 0 1px 2px rgba(11, 11, 11, 0.04),
              0 14px 30px -22px rgba(11, 11, 11, 0.45);
}

/* User: right-hand tinted bubble, sized to its text. */
[data-testid="stChatMessage"]:has(.chat-role-user) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has(.chat-role-user)
  > [data-testid^="stChatMessageAvatar"] {
  background: #f4f3ef;
}
[data-testid="stChatMessage"]:has(.chat-role-user)
  [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: min(660px, 82%); margin-left: auto;
  background: linear-gradient(140deg, #f0f6fe, #e3edfa);
  border: 1px solid #d2e2f7;
  border-radius: 18px 5px 18px 18px;
  padding: 11px 17px;
  color: var(--chat-ink); font-weight: 500;
  box-shadow: 0 1px 2px rgba(42, 120, 214, 0.06);
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }

/* ---- Answer footer: query log + timing ------------------------------ */
[data-testid="stChatMessage"] [data-testid="stExpander"] details {
  border: 1px solid var(--chat-line); border-radius: 10px;
  background: #fbfbf9; overflow: hidden;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary {
  font-size: 12px; font-weight: 550; color: var(--chat-ink-2); padding: 6px 10px;
}
[data-testid="stChatMessage"] [data-testid="stExpander"] summary:hover {
  color: var(--chat-primary-dark);
}
.chat-elapsed {
  display: inline-flex; align-items: center; gap: 5px;
  float: right; margin-top: 7px;
  padding: 3px 9px; border-radius: 999px;
  background: #f4f3ef; border: 1px solid var(--chat-line);
  color: var(--chat-muted); font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.chat-status {
  display: inline-flex; align-items: center; gap: 8px;
  color: var(--chat-ink-2); font-size: 13px;
}
.chat-status::before {
  content: ""; width: 7px; height: 7px; border-radius: 50%;
  background: var(--chat-primary);
  animation: chat-pulse 1.1s ease-in-out infinite;
}
@keyframes chat-pulse {
  0%, 100% { opacity: 0.25; transform: scale(0.8); }
  50%      { opacity: 1;    transform: scale(1.15); }
}

/* ---- Empty state ---------------------------------------------------- */
.st-key-chat_hero {
  border: 1px solid var(--chat-line) !important;
  border-radius: 20px !important;
  padding: 26px 26px 22px !important;
  background:
    radial-gradient(130% 150% at 0% 0%,
                    rgba(42, 120, 214, 0.075), rgba(42, 120, 214, 0) 58%),
    linear-gradient(180deg, #ffffff, #fbfbf9) !important;
  box-shadow: 0 1px 2px rgba(11, 11, 11, 0.04),
              0 30px 56px -46px rgba(11, 11, 11, 0.55) !important;
}
.chat-hero-title {
  font-size: 21px; font-weight: 660; letter-spacing: -0.02em;
  color: var(--chat-ink); margin: 0 0 6px;
}
.chat-hero-sub {
  font-size: 13.5px; line-height: 1.55; color: var(--chat-ink-2);
  max-width: 62ch; margin: 0 0 18px;
}

/* Suggestion cards. */
[class*="st-key-eg_"] button {
  width: 100%; min-height: 62px;
  display: flex; align-items: flex-start; justify-content: flex-start;
  text-align: left;
  padding: 13px 15px;
  border-radius: 14px; border: 1px solid var(--chat-line);
  background: #ffffff; color: var(--chat-ink);
  font-weight: 520; line-height: 1.35;
  transition: transform .13s ease, border-color .13s ease, box-shadow .13s ease;
}
[class*="st-key-eg_"] button p { text-align: left; margin: 0; }
[class*="st-key-eg_"] button:hover {
  transform: translateY(-2px);
  border-color: var(--chat-primary);
  color: var(--chat-primary-dark);
  box-shadow: 0 12px 24px -16px rgba(42, 120, 214, 0.6);
}
[class*="st-key-eg_"] button:active { transform: translateY(0); }
.eg-hint {
  font-size: 11.5px; color: var(--chat-muted);
  padding: 7px 4px 0; letter-spacing: 0.015em;
}

/* ---- Composer ------------------------------------------------------- */
[data-testid="stChatInput"] {
  border-radius: 15px;
  border: 1px solid var(--chat-line);
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(11, 11, 11, 0.04),
              0 14px 30px -24px rgba(11, 11, 11, 0.6);
  transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--chat-primary);
  box-shadow: 0 0 0 3px rgba(42, 120, 214, 0.14);
}

/* ---- Clear button --------------------------------------------------- */
.st-key-chat_clear button {
  border-radius: 10px; border: 1px solid var(--chat-line);
  background: #ffffff; color: var(--chat-ink-2);
  font-size: 13px; font-weight: 520;
  transition: border-color .13s ease, color .13s ease;
}
.st-key-chat_clear button:hover:not(:disabled) {
  border-color: #d9958c; color: #c0392b;
}
</style>
"""


def _safe_rerun() -> None:
    """Rerun just this fragment - falling back to a full rerun if Streamlit
    doesn't consider the current execution a fragment-scoped rerun.

    Streamlit only allows scope="fragment" while already inside a rerun that
    it tracks as fragment-triggered (e.g. a widget inside this @st.fragment
    changing) - never during a full script run, even from fragment code. The
    normal chat flow always qualifies, but a full-app rerun triggered from
    elsewhere (a dashboard filter, a fresh page load) would not, so this
    falls back instead of crashing the whole chat on that edge case.
    """
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        st.rerun()


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
    role = message["role"]
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR
    with st.chat_message(role, avatar=avatar):
        # Hidden hook the bubble CSS keys off - a custom avatar hides the role
        # from the DOM, so the stylesheet cannot tell the two apart without it.
        st.markdown(
            f'<span class="chat-role chat-role-{role}"></span>',
            unsafe_allow_html=True,
        )
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
                cols[-1].markdown(
                    f'<span class="chat-elapsed">⏱ {elapsed:.1f}s</span>',
                    unsafe_allow_html=True,
                )


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

    def _status(text: str) -> None:
        """A live step label with a pulsing dot, so waiting feels accounted for."""
        status_area.markdown(
            f'<div class="chat-status">{text}</div>', unsafe_allow_html=True
        )

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
                _status(f"Querying the data — step {iteration + 1}")
            elif call.name == "render_chart":
                _status("Building the chart")
            else:
                _status("Computing the headline number")

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
    with st.container(border=True, key="chat_hero"):
        st.markdown(
            '<div class="chat-hero-title">Ask a question in plain English — '
            "I'll write the SQL.</div>"
            '<div class="chat-hero-sub">'
            f"Powered by {provider_label}, querying live over the full dataset "
            "with DuckDB. Start with one of these, or type your own below."
            "</div>",
            unsafe_allow_html=True,
        )
        top = st.columns(2, gap="small")
        bottom = st.columns(2, gap="small")
        for i, (column, (question, hint)) in enumerate(
            zip(top + bottom, EXAMPLE_QUESTIONS)
        ):
            with column:
                # Index keys, not the question text: the key becomes a CSS class
                # (`st-key-eg_0`), and spaces/punctuation would not survive that.
                if st.button(question, width="stretch", key=f"eg_{i}"):
                    st.session_state.pending_question = question
                    _safe_rerun()
                st.markdown(f'<div class="eg-hint">{hint}</div>', unsafe_allow_html=True)


@st.fragment
def render(df: pd.DataFrame, con) -> None:
    """Draw the chat tab. `df` and `con` are the full, unfiltered dataset."""
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    try:
        provider = llm.resolve_provider()
    except llm.ProviderError as exc:
        st.error(str(exc))
        st.info(
            "Copy `.env.example` to `.env`, put your key in it, and restart "
            "`streamlit run app.py`."
        )
        return

    header, reset = st.columns([5, 1], vertical_alignment="center")
    with header:
        st.markdown(
            '<div class="chat-head">'
            "<h3>Ask the data</h3>"
            f'<span class="chat-pill">{provider.label}</span>'
            "</div>"
            '<p class="chat-note">'
            f"Answers come from live SQL over all {len(df):,} tickets — the "
            "sidebar filters apply to the Dashboard tab only. Ask for a "
            'filtered slice in your question instead, e.g. "…in the Central zone".'
            "</p>",
            unsafe_allow_html=True,
        )
    with reset:
        if st.button(
            "🗑️ Clear",
            width="stretch",
            key="chat_clear",
            disabled=not st.session_state.chat_log,
        ):
            st.session_state.chat_log = []
            st.session_state.llm_history = None
            _safe_rerun()

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
        st.markdown(
            '<span class="chat-role chat-role-assistant"></span>',
            unsafe_allow_html=True,
        )
        with st.spinner(f"Thinking with {provider.label}…"):
            message = _run_agent(provider, df, con, question)

    st.session_state.chat_log.append(message)
    # Re-run so the new answer renders through the same path as the history,
    # which keeps the plotly chart keys stable.
    _safe_rerun()

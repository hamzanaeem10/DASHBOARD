"""Streamlit entrypoint: page config, data loading, and the two tabs.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import chatbot
import dashboard
import data_loader

# Load .env before anything reads an API key.
load_dotenv()

st.set_page_config(
    page_title="Ticket Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolve_source() -> tuple[bytes | None, str | None, str]:
    """Decide which workbook to load: the upload if there is one, else the bundled file.

    Returns ``(file_bytes, path, label)``.
    """
    upload = st.sidebar.file_uploader(
        "Ticket workbook (.xlsx)",
        type=["xlsx"],
        help="Reads the sheet named 'Tickets'. Replaces the loaded data.",
    )
    if upload is not None:
        return upload.getvalue(), None, upload.name

    default_path = data_loader.DEFAULT_DATA_PATH
    if os.path.exists(default_path):
        return None, default_path, os.path.basename(default_path)

    return None, None, ""


def _get_connection(df, token: str):
    """One DuckDB connection per loaded dataset, kept in session state.

    A connection object is not hashable, so it cannot go through st.cache_data;
    the token tells us when the underlying frame changed.
    """
    if st.session_state.get("duckdb_token") != token:
        st.session_state.duckdb_con = data_loader.get_connection(df)
        st.session_state.duckdb_token = token
    return st.session_state.duckdb_con


def main() -> None:
    st.sidebar.title("📡 Ticket Analytics")
    file_bytes, path, label = _resolve_source()

    if file_bytes is None and path is None:
        st.title("Ticket Analytics")
        st.info(
            "Upload a ticket workbook (.xlsx) in the sidebar to begin. "
            "It must contain a sheet named **Tickets**."
        )
        st.caption(
            f"No bundled workbook found at `{data_loader.DEFAULT_DATA_PATH}`. "
            "Set the `TICKETS_XLSX` environment variable to point at one."
        )
        return

    signature = (
        f"upload:{label}:{len(file_bytes)}" if file_bytes is not None
        else data_loader.file_signature(path)
    )

    try:
        df, report = data_loader.load_data(file_bytes, path, signature)
    except Exception as exc:
        st.error(f"Could not load the workbook: {exc}")
        return

    con = _get_connection(df, signature)

    # --- Sidebar: what got loaded, and what cleaning did -------------------
    st.sidebar.caption(f"**{label}** — {len(df):,} tickets")
    with st.sidebar.expander("Load report"):
        st.write(
            f"- Rows read: **{report['rows_in']:,}**\n"
            f"- Rows kept: **{report['rows_out']:,}**\n"
            f"- Dropped (unparseable date): **{report['rows_dropped_unparseable_date']:,}**\n"
            f"- Vendor `0` placeholders nulled: **{report.get('Vendor_nulled', 0):,}**\n"
            f"- UCID `0` placeholders nulled: **{report.get('ucid_zero_nulled', 0):,}**\n"
            f"- Media Type spellings merged: **{report.get('media_types_merged', 0)}**\n"
            f"- Duplicate ticket Numbers (not deduped): **{report.get('duplicate_ticket_numbers', 0)}**\n"
            f"- closed_at earlier than Created On (not corrected): **{report.get('closed_before_created', 0)}**"
        )

    selections = dashboard.render_filters(df)
    filtered = dashboard.apply_filters(df, selections)

    # --- Main area ---------------------------------------------------------
    st.title("Ticket Analytics")
    dashboard_tab, chat_tab = st.tabs(["Dashboard", "Ask the data"])

    with dashboard_tab:
        dashboard.render(df, filtered, selections)

    with chat_tab:
        # The chatbot deliberately sees the full dataset, not the filtered one.
        chatbot.render(df, con)


if __name__ == "__main__":
    main()

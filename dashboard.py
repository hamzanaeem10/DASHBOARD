"""The fixed analytics dashboard: sidebar filters, KPI cards, and the charts.

Section numbering follows the client's own `Summary` sheet, so a chart here can
be checked against the pivot table it replaces.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from data_loader import COMPLAINT_SERVICE_RQST

# An empty multiselect means "no constraint", which is friendlier than making
# the user select all 400 clients to see everything.
_DIMENSION_FILTERS = [
    ("Zone", "Zone", "Zone"),
    ("Client_Uniq", "Client", "Client"),
    ("Vendor Names", "Vendor", "Vendor"),
    ("Media Type", "Media Type", "Media type"),
]

SCOPE_COMPLAINTS = "Complaints only"
SCOPE_ALL = "All tickets"


def render_filters(df: pd.DataFrame) -> dict:
    """Draw the sidebar controls and return the chosen values."""
    st.sidebar.subheader("Scope")
    scope = st.sidebar.radio(
        "Ticket scope",
        [SCOPE_COMPLAINTS, SCOPE_ALL],
        index=0,
        label_visibility="collapsed",
        help=(
            "Their monthly report counts complaints only. Keeping that as the "
            "default means these figures reconcile with the report they already "
            "circulate."
        ),
    )
    complaints = int(df["Service Rqst"].eq(COMPLAINT_SERVICE_RQST).sum())
    st.sidebar.caption(
        f"Complaints {complaints:,} · all tickets {len(df):,}"
    )

    st.sidebar.subheader("Filters")
    min_date = df["Created On"].min().date()
    max_date = df["Created On"].max().date()
    date_range = st.sidebar.date_input(
        "Created between",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filters on Created On.",
    )

    selections: dict = {"scope": scope, "date_range": date_range}
    for column, key, label in _DIMENSION_FILTERS:
        options = sorted(df[column].dropna().astype(str).unique())
        selections[key] = st.sidebar.multiselect(
            label, options, default=[], placeholder=f"All ({len(options)})",
        )
    return selections


def apply_filters(df: pd.DataFrame, selections: dict) -> pd.DataFrame:
    """Apply scope and filters. Every chart reads the frame this returns."""
    out = df

    if selections.get("scope", SCOPE_COMPLAINTS) == SCOPE_COMPLAINTS:
        out = out[out["Service Rqst"] == COMPLAINT_SERVICE_RQST]

    date_range = selections.get("date_range")
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range
        # `end` is inclusive of the whole day.
        out = out[
            (out["Created On"] >= pd.Timestamp(start))
            & (out["Created On"] < pd.Timestamp(end) + pd.Timedelta(days=1))
        ]

    for column, key, _label in _DIMENSION_FILTERS:
        chosen = selections.get(key)
        if chosen:
            out = out[out[column].astype(str).isin(chosen)]

    return out


def _kpi_row(df: pd.DataFrame, full: pd.DataFrame) -> None:
    """Five headline metrics for the current slice."""
    cols = st.columns(5)
    total = len(df)

    cols[0].metric("Total tickets", f"{total:,}")
    if len(full):
        cols[0].caption(f"{total / len(full) * 100:.1f}% of all tickets")

    if total == 0:
        for col, label in zip(
            cols[1:], ["MTTR", "Avg uptime", "SLA compliance", "FCR rate"]
        ):
            col.metric(label, "—")
        return

    # MTTR is the mean of Downtime Calc - the same statistic the client reports
    # monthly, shown in their H:MM:SS format rather than as a decimal.
    mttr = df["downtime_hours"].mean()
    cols[1].metric("MTTR", charts.format_hours(mttr))
    cols[1].caption("Mean time to repair")

    avg_uptime = df["Uptime %"].mean()
    cols[2].metric(
        "Avg uptime", f"{avg_uptime * 100:.2f}%" if pd.notna(avg_uptime) else "—"
    )

    sla = df["SLA Status"].eq("SLA Met").mean() * 100
    cols[3].metric("SLA compliance", f"{sla:.1f}%")
    cols[3].caption("SLA Status = SLA Met")

    fcr = df["FCR"].eq("yes").mean() * 100
    cols[4].metric("FCR rate", f"{fcr:.1f}%")
    cols[4].caption("Resolved first call")


def _chart(fig, caption: str, key: str) -> None:
    """Render a figure with its short explanatory caption beneath."""
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    st.caption(caption)


def _table(frame: pd.DataFrame, caption: str, column_config: dict,
           height: int = 380) -> None:
    st.dataframe(
        frame, width="stretch", height=height, hide_index=True,
        column_config=column_config,
    )
    st.caption(caption)


_PROGRESS = st.column_config.ProgressColumn(
    "SLA compliance %", format="%.1f%%", min_value=0, max_value=100
)


def render(df: pd.DataFrame, filtered: pd.DataFrame, selections: dict) -> None:
    """Draw the whole dashboard for the given filtered frame."""
    scope = selections.get("scope", SCOPE_COMPLAINTS)
    _kpi_row(filtered, df)
    if scope == SCOPE_COMPLAINTS:
        st.caption(
            "Scoped to **complaints**, matching their monthly report. Their "
            "headline Monthly Incident Summary additionally excludes "
            "`ENTERPRISE PRODUCT` (48,195 rows); every other pivot uses the full "
            "complaint set shown here. Switch to *All tickets* in the sidebar "
            "for the unscoped view."
        )
    st.divider()

    if filtered.empty:
        st.warning("No tickets match the current filters. Widen them in the sidebar.")
        return

    # === 1. Volume and resolution speed ===================================
    st.subheader("1 · Volume and resolution speed")
    granularity = st.radio(
        "Trend granularity", ["Day", "Week", "Month"],
        index=1, horizontal=True, label_visibility="collapsed",
    )
    _chart(
        charts.ticket_volume_trend(filtered, granularity),
        f"Tickets created per {granularity.lower()}, by Created On. "
        "The final period may be partial.",
        "trend",
    )

    left, right = st.columns(2, gap="medium")
    with left:
        _chart(
            charts.mttr_trend(filtered),
            "Mean time to repair per month — the client's headline metric. "
            "Endpoints are labelled; hover for any month.",
            "mttr",
        )
    with right:
        _chart(
            charts.fcr_rate_over_time(filtered),
            "Monthly share of tickets with FCR = yes. Hover for the ticket "
            "count behind each point.",
            "fcr_trend",
        )

    # === 2. What is breaking ==============================================
    st.subheader("2 · What is breaking")
    left, right = st.columns(2, gap="medium")
    with left:
        _chart(
            charts.issue_category_breakdown(filtered),
            "Ticket count per issue category, largest first.",
            "issue_cat",
        )
    with right:
        _chart(
            charts.top_resolution_codes(filtered),
            "Top 15 resolution codes — the client calls these RFO "
            "(Reason For Outage).",
            "res_codes",
        )

    _chart(
        charts.category_by_month(filtered, "Resolution code",
                                 "RFO breakdown by month"),
        "Their RFO-Wise Breakdown: top 7 reasons stacked per month, everything "
        "else grouped as Other.",
        "rfo_month",
    )
    _chart(
        charts.category_by_month(filtered, "Media Type",
                                 "Media type by month"),
        "Their Media-Wise Analysis: top 7 media types stacked per month.",
        "media_month",
    )

    left, right = st.columns(2, gap="medium")
    with left:
        _chart(
            charts.media_type_breakdown(filtered),
            "Ticket count by media type (top 12). Spelling variants of the same "
            "type are merged on load.",
            "media",
        )
    with right:
        _chart(
            charts.service_request_donut(filtered),
            "Share of tickets by service request type."
            + (" Scoped to complaints, so this is a single slice."
               if scope == SCOPE_COMPLAINTS else ""),
            "svc_req",
        )

    # === 3. SLA and downtime ==============================================
    st.subheader("3 · SLA and downtime")
    left, right = st.columns(2, gap="medium")
    with left:
        _chart(
            charts.sla_status_donut(filtered),
            "Share of tickets meeting the SLA threshold (uptime ≥ 98.5%).",
            "sla_donut",
        )
    with right:
        _chart(
            charts.downtime_bracket_distribution(filtered),
            "Tickets per downtime bucket, ordered by severity rather than "
            "alphabetically.",
            "bracket",
        )

    left, right = st.columns(2, gap="medium")
    with left:
        _chart(
            charts.zone_comparison(filtered),
            "Zones on both measures. Count and percentage sit in separate "
            "panels rather than on one axis, so neither scale distorts the other.",
            "zone",
        )
    with right:
        _chart(
            charts.link_status_donut(filtered),
            "Share of tickets by link status; rows with no recorded status show "
            "as Unknown.",
            "link_status",
        )

    # === 4. When tickets arrive ===========================================
    st.subheader("4 · When tickets arrive")
    _chart(
        charts.weekday_hour_heatmap(filtered),
        "Ticket density by day of week and hour of creation. Darker means more "
        "tickets — this is their 'identify peak hours' view.",
        "heatmap",
    )
    _chart(
        charts.weekday_weekend_trend(filtered),
        "Their weekday and weekend trends, reported separately. The right panel "
        "normalises by number of days, since five weekdays will always out-total "
        "two weekend days.",
        "weekend",
    )

    # === 5. Leaderboards ==================================================
    st.subheader("5 · Vendors, clients and links")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Vendor performance**")
        _table(
            charts.vendor_leaderboard(filtered),
            "Click a column header to sort. Rows with no vendor recorded are "
            "excluded (the `0` placeholder is nulled on load).",
            {
                "Tickets": st.column_config.NumberColumn(format="%d"),
                "Avg downtime (h)": st.column_config.NumberColumn(format="%.2f"),
                "SLA compliance %": _PROGRESS,
            },
        )
    with right:
        st.markdown("**Client leaderboard**")
        _table(
            charts.client_leaderboard(filtered),
            "Top 50 clients by ticket volume, on the deduplicated client name.",
            {
                "Tickets": st.column_config.NumberColumn(format="%d"),
                "Avg uptime %": st.column_config.NumberColumn(format="%.3f"),
                "SLA compliance %": _PROGRESS,
            },
        )

    st.markdown("**Link-wise summary**")
    _table(
        charts.link_leaderboard(filtered),
        "Their Link-Wise Summary: the 50 links with the most incidents, keyed on "
        "Subject with the UCID alongside. The `UCID = 0` placeholder is nulled on "
        "load, so it no longer ranks as the largest link.",
        {
            "UCID": st.column_config.NumberColumn(format="%d"),
            "Tickets": st.column_config.NumberColumn(format="%d"),
            "Avg downtime (h)": st.column_config.NumberColumn(format="%.2f"),
            "Avg uptime %": st.column_config.NumberColumn(format="%.3f"),
            "SLA compliance %": _PROGRESS,
        },
        height=420,
    )

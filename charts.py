"""Chart builders. One function per chart; each returns a Plotly figure.

The dashboard calls the named builders; the chatbot calls :func:`build_chart`
with rows it got back from SQL. Both route through the same styling in
:mod:`theme`, so a chart Claude draws is indistinguishable from a dashboard one.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import theme
from data_loader import WEEKDAY_ORDER

# A hairline in the surface colour keeps a visible gap between adjacent fills.
_BAR_LINE = dict(line=dict(color=theme.SURFACE, width=1))

EMPTY_MESSAGE = "No rows match the current filters."


def _empty_fig(message: str = EMPTY_MESSAGE, height: int = 340) -> go.Figure:
    """A quiet placeholder so an over-filtered view never shows a broken axis."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
        font=dict(family=theme.FONT_FAMILY, size=13, color=theme.INK_MUTED),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return theme.style_fig(fig, height=height, showlegend=False)


def _fmt_counts(values) -> list[str]:
    return [f"{v:,.0f}" for v in values]


def _headroom(series) -> float:
    """Upper x/y bound that leaves room for an outside data label."""
    peak = float(pd.Series(series).max() or 0)
    return peak * 1.18 if peak > 0 else 1.0


# ---------------------------------------------------------------------------
# Dashboard charts
# ---------------------------------------------------------------------------

def ticket_volume_trend(df: pd.DataFrame, granularity: str = "Day") -> go.Figure:
    """Tickets created over time. One series, so no legend - the title names it."""
    if df.empty:
        return _empty_fig()

    freq = {"Day": "D", "Week": "W-MON", "Month": "MS"}.get(granularity, "D")
    counts = df.set_index("Created On").resample(freq).size().reset_index(name="tickets")

    fig = go.Figure(
        go.Scatter(
            x=counts["Created On"], y=counts["tickets"], mode="lines",
            line=dict(color=theme.PRIMARY, width=2),
            fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:,}</b> tickets<extra></extra>",
            name="Tickets",
        )
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Tickets", rangemode="tozero")
    return theme.style_fig(
        fig, title=f"Ticket volume by {granularity.lower()}", showlegend=False
    )


def _hbar(counts: pd.Series, title: str, height: int) -> go.Figure:
    """Shared horizontal-bar construction for the ranked breakdowns."""
    counts = counts.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=counts.values, y=counts.index.astype(str), orientation="h",
            marker=dict(color=theme.PRIMARY, **_BAR_LINE),
            text=_fmt_counts(counts.values), textposition="outside",
            textfont=dict(size=11, color=theme.INK_SECONDARY),
            hovertemplate="%{y}<br><b>%{x:,}</b> tickets<extra></extra>",
        )
    )
    fig.update_xaxes(
        title_text="Tickets", showgrid=True, gridcolor=theme.GRIDLINE,
        range=[0, _headroom(counts.values)],
    )
    fig.update_yaxes(showgrid=False)
    return theme.style_fig(fig, title=title, height=height, showlegend=False)


def issue_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Ticket count per issue category, largest first."""
    if df.empty:
        return _empty_fig(height=420)
    return _hbar(df["Issue Category"].value_counts(), "Issue categories", 420)


def top_resolution_codes(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """The resolution codes that close the most tickets."""
    if df.empty:
        return _empty_fig(height=460)
    counts = df["Resolution code"].value_counts().head(top_n)
    return _hbar(counts, f"Top {top_n} resolution codes", 460)


def media_type_breakdown(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Ticket count by media type."""
    if df.empty:
        return _empty_fig(height=420)
    counts = df["Media Type"].value_counts().head(top_n)
    return _hbar(counts, "Media types", 420)


def _donut(labels, values, colors, title: str, height: int = 340) -> go.Figure:
    """Shared donut construction - share of total across a few categories."""
    values = list(values)
    fig = go.Figure(
        go.Pie(
            labels=[str(x) for x in labels], values=values, hole=0.58, sort=False,
            marker=dict(colors=colors, line=dict(color=theme.SURFACE, width=2)),
            texttemplate="%{percent:.1%}", textposition="inside",
            insidetextfont=dict(size=11, color="#ffffff"),
            hovertemplate="%{label}<br><b>%{value:,}</b> (%{percent:.1%})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{sum(values):,.0f}</b><br><span style='font-size:11px'>total</span>",
        showarrow=False, x=0.5, y=0.5,
        font=dict(family=theme.FONT_FAMILY, size=17, color=theme.INK_PRIMARY),
    )
    return theme.style_fig(fig, title=title, height=height)


def service_request_donut(df: pd.DataFrame) -> go.Figure:
    """Share of tickets by service request type."""
    if df.empty:
        return _empty_fig()
    counts = df["Service Rqst"].value_counts()
    return _donut(
        counts.index, counts.values,
        theme.categorical_colors(len(counts)), "Service request type",
    )


def sla_status_donut(df: pd.DataFrame) -> go.Figure:
    """SLA met vs not met, in the reserved status colours - this is a state."""
    if df.empty:
        return _empty_fig()
    counts = df["SLA Status"].value_counts()
    colors = [
        theme.STATUS_GOOD if str(label).strip().lower() == "sla met"
        else theme.STATUS_CRITICAL
        for label in counts.index
    ]
    return _donut(counts.index, counts.values, colors, "SLA status")


def link_status_donut(df: pd.DataFrame) -> go.Figure:
    """Share of tickets by link status. Missing values are shown, not dropped."""
    if df.empty:
        return _empty_fig()
    counts = df["Link Status"].fillna("Unknown").value_counts()
    return _donut(
        counts.index, counts.values,
        theme.categorical_colors(len(counts)), "Link status",
    )


def downtime_bracket_distribution(df: pd.DataFrame) -> go.Figure:
    """Tickets per downtime bucket, in severity order rather than alphabetical."""
    if df.empty:
        return _empty_fig()

    # `Bracket` is an ordered categorical, so value_counts keeps the logical order.
    counts = df["Bracket"].value_counts(sort=False)
    counts = counts[counts > 0]
    if counts.empty:
        return _empty_fig()

    fig = go.Figure(
        go.Bar(
            x=counts.index.astype(str), y=counts.values,
            marker=dict(color=theme.ordinal_colors(len(counts)), **_BAR_LINE),
            text=_fmt_counts(counts.values), textposition="outside",
            textfont=dict(size=11, color=theme.INK_SECONDARY),
            hovertemplate="%{x}<br><b>%{y:,}</b> tickets<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Tickets", range=[0, _headroom(counts.values)])
    fig.update_layout(bargap=0.3)
    return theme.style_fig(fig, title="Downtime bracket distribution", showlegend=False)


def weekday_hour_heatmap(df: pd.DataFrame) -> go.Figure:
    """When tickets arrive: weekday down the side, hour of day across."""
    if df.empty:
        return _empty_fig(height=380)

    grid = (
        df.pivot_table(index="Weekday", columns="TimeStamp",
                       aggfunc="size", observed=False)
        .reindex(index=WEEKDAY_ORDER)
        .reindex(columns=list(range(24)))
        .fillna(0)
    )
    grid = grid.iloc[::-1]  # Monday at the top of the y-axis.

    fig = go.Figure(
        go.Heatmap(
            z=grid.values, x=[f"{h:02d}" for h in grid.columns],
            y=[str(i) for i in grid.index],
            colorscale=theme.BLUE_SEQUENTIAL, xgap=2, ygap=2,
            colorbar=dict(title="", thickness=10, outlinewidth=0,
                          tickfont=dict(size=10, color=theme.INK_MUTED)),
            hovertemplate="%{y} at %{x}:00<br><b>%{z:,.0f}</b> tickets<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Hour of day", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return theme.style_fig(
        fig, title="Ticket density by weekday and hour", height=380, showlegend=False
    )


def fcr_rate_over_time(df: pd.DataFrame) -> go.Figure:
    """Monthly share of tickets resolved first time."""
    if df.empty:
        return _empty_fig()

    monthly = (
        df.assign(_fcr=df["FCR"].eq("yes"))
        .groupby("created_month", observed=True)["_fcr"]
        .agg(["mean", "size"])
        .reset_index()
    )
    fig = go.Figure(
        go.Scatter(
            x=monthly["created_month"], y=monthly["mean"] * 100,
            mode="lines+markers", line=dict(color=theme.PRIMARY, width=2),
            marker=dict(size=8, color=theme.PRIMARY,
                        line=dict(color=theme.SURFACE, width=2)),
            customdata=monthly["size"],
            hovertemplate="%{x|%b %Y}<br><b>%{y:.1f}%</b> FCR"
                          "<br>%{customdata:,} tickets<extra></extra>",
            name="FCR rate",
        )
    )
    fig.update_yaxes(title_text="FCR rate (%)", ticksuffix="%", rangemode="tozero")
    return theme.style_fig(
        fig, title="First-call resolution rate over time", showlegend=False
    )


def zone_comparison(df: pd.DataFrame) -> go.Figure:
    """Zones on two measures.

    Ticket count and SLA compliance have different units, so each gets its own
    panel rather than being forced onto a shared or secondary axis.
    """
    if df.empty:
        return _empty_fig()

    by_zone = (
        df.dropna(subset=["Zone"])
        .groupby("Zone", observed=True)
        .agg(tickets=("Number", "size"),
             sla=("SLA Status", lambda s: s.eq("SLA Met").mean() * 100))
        .sort_values("tickets", ascending=False)
        .reset_index()
    )
    if by_zone.empty:
        return _empty_fig()

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=("Tickets", "SLA compliance"))
    fig.add_trace(
        go.Bar(x=by_zone["Zone"], y=by_zone["tickets"],
               marker=dict(color=theme.CATEGORICAL[0], **_BAR_LINE),
               text=_fmt_counts(by_zone["tickets"]), textposition="outside",
               textfont=dict(size=11, color=theme.INK_SECONDARY),
               hovertemplate="%{x}<br><b>%{y:,}</b> tickets<extra></extra>",
               showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=by_zone["Zone"], y=by_zone["sla"],
               marker=dict(color=theme.CATEGORICAL[1], **_BAR_LINE),
               text=[f"{v:.1f}%" for v in by_zone["sla"]], textposition="outside",
               textfont=dict(size=11, color=theme.INK_SECONDARY),
               hovertemplate="%{x}<br><b>%{y:.1f}%</b> SLA met<extra></extra>",
               showlegend=False),
        row=1, col=2,
    )
    fig.update_yaxes(range=[0, _headroom(by_zone["tickets"])], row=1, col=1)
    fig.update_yaxes(range=[0, 112], ticksuffix="%", row=1, col=2)
    fig.update_layout(bargap=0.45)
    for annotation in fig.layout.annotations:
        annotation.font = dict(family=theme.FONT_FAMILY, size=11, color=theme.INK_MUTED)
    return theme.style_fig(fig, title="Zone comparison", showlegend=False)


def format_hours(hours) -> str:
    """Render a float number of hours as H:MM:SS, the way their report does."""
    if hours is None or pd.isna(hours):
        return "-"
    # Truncate rather than round: this is how Excel renders the duration in
    # their report, so 3:26:41.632 shows as 3:26:41 in both places.
    total = int(float(hours) * 3600)
    return f"{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def mttr_trend(df: pd.DataFrame) -> go.Figure:
    """Mean time to repair per month - the client's headline monthly metric."""
    if df.empty:
        return _empty_fig()

    monthly = (
        df.groupby("created_month", observed=True)["downtime_hours"]
        .agg(["mean", "size"])
        .reset_index()
        .sort_values("created_month")
    )
    if monthly.empty:
        return _empty_fig()

    fig = go.Figure(
        go.Scatter(
            x=monthly["created_month"], y=monthly["mean"],
            mode="lines+markers", line=dict(color=theme.PRIMARY, width=2),
            marker=dict(size=8, color=theme.PRIMARY,
                        line=dict(color=theme.SURFACE, width=2)),
            customdata=list(zip([format_hours(v) for v in monthly["mean"]],
                                monthly["size"])),
            hovertemplate="%{x|%b %Y}<br><b>%{customdata[0]}</b> MTTR"
                          "<br>%{customdata[1]:,} tickets<extra></extra>",
            name="MTTR",
        )
    )
    # Label the endpoints only - the trend is the story, not every value.
    for position in {0, len(monthly) - 1}:
        row = monthly.iloc[position]
        fig.add_annotation(
            x=row["created_month"], y=row["mean"], text=format_hours(row["mean"]),
            showarrow=False, yshift=16,
            font=dict(family=theme.FONT_FAMILY, size=11, color=theme.INK_SECONDARY),
        )
    fig.update_yaxes(title_text="Hours", rangemode="tozero")
    return theme.style_fig(
        fig, title="Mean time to repair (MTTR) by month", showlegend=False
    )


def weekday_weekend_trend(df: pd.DataFrame) -> go.Figure:
    """Weekday vs weekend volume by month - reported separately by the client.

    Also plots tickets *per day*, because five weekdays will always out-total
    two weekend days; the per-day panel is what says whether weekends are busier.
    """
    if df.empty:
        return _empty_fig()

    scoped = df.dropna(subset=["Weekday"]).copy()
    scoped["span"] = scoped["Weekday"].isin(["Saturday", "Sunday"]).map(
        {True: "Weekend (Sat-Sun)", False: "Weekday (Mon-Fri)"}
    )
    grouped = (
        scoped.groupby(["created_month", "span"], observed=True)
        .agg(tickets=("Weekday", "size"), days=("created_date", "nunique"))
        .reset_index()
        .sort_values("created_month")
    )
    if grouped.empty:
        return _empty_fig()
    grouped["per_day"] = grouped["tickets"] / grouped["days"].replace(0, pd.NA)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        subplot_titles=("Tickets", "Tickets per day"))
    for index, span in enumerate(["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"]):
        part = grouped[grouped["span"] == span]
        color = theme.CATEGORICAL[index]
        fig.add_trace(
            go.Bar(x=part["created_month"], y=part["tickets"], name=span,
                   marker=dict(color=color, **_BAR_LINE),
                   hovertemplate="%{x|%b %Y}<br><b>%{y:,}</b> tickets<extra>"
                                 + span + "</extra>"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=part["created_month"], y=part["per_day"], name=span,
                   marker=dict(color=color, **_BAR_LINE), showlegend=False,
                   hovertemplate="%{x|%b %Y}<br><b>%{y:,.0f}</b> per day<extra>"
                                 + span + "</extra>"),
            row=1, col=2,
        )
    fig.update_layout(barmode="group", bargap=0.28)
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(tickformat="%b")
    for annotation in fig.layout.annotations:
        annotation.font = dict(family=theme.FONT_FAMILY, size=11,
                               color=theme.INK_MUTED)
    return theme.style_fig(fig, title="Weekday vs weekend volume", showlegend=True)


def category_by_month(df: pd.DataFrame, column: str, title: str,
                      top_n: int = 7) -> go.Figure:
    """Stacked monthly breakdown of a category, mirroring their pivot tables.

    Only the top `top_n` categories get their own colour - the palette has eight
    slots and cycling hues past that stops carrying meaning. The rest fold into
    a neutral "Other".
    """
    if df.empty or column not in df.columns:
        return _empty_fig(height=400)

    scoped = df.dropna(subset=[column]).copy()
    if scoped.empty:
        return _empty_fig(height=400)

    top = list(scoped[column].value_counts().head(top_n).index)
    scoped["_cat"] = scoped[column].astype(str).where(
        scoped[column].isin(top), "Other"
    )

    grouped = (
        scoped.groupby(["created_month", "_cat"], observed=True)
        .size().reset_index(name="tickets")
    )
    order = [str(t) for t in top]
    if (scoped["_cat"] == "Other").any():
        order.append("Other")
    palette = theme.categorical_colors(len(top)) + [theme.INK_MUTED]

    fig = go.Figure()
    for index, name in enumerate(order):
        part = grouped[grouped["_cat"] == name].sort_values("created_month")
        label = name if len(name) <= 38 else name[:37] + "\u2026"
        fig.add_trace(
            go.Bar(
                x=part["created_month"], y=part["tickets"], name=label,
                marker=dict(color=palette[index], **_BAR_LINE),
                hovertemplate="%{x|%b %Y}<br><b>%{y:,}</b> tickets<extra>"
                              + label + "</extra>",
            )
        )
    fig.update_layout(barmode="stack", bargap=0.28)
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(title_text="Tickets", rangemode="tozero")
    return theme.style_fig(fig, title=title, height=400, showlegend=True)


# ---------------------------------------------------------------------------
# Leaderboards - returned as frames; Streamlit renders them sortable
# ---------------------------------------------------------------------------

def vendor_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vendor ticket count, average downtime and SLA compliance."""
    columns = ["Vendor", "Tickets", "Avg downtime (h)", "SLA compliance %"]
    scoped = df.dropna(subset=["Vendor Names"])
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    out = (
        scoped.groupby("Vendor Names", observed=True)
        .agg(
            Tickets=("Number", "size"),
            **{"Avg downtime (h)": ("downtime_hours", "mean"),
               "SLA compliance %": ("SLA Status", lambda s: s.eq("SLA Met").mean() * 100)},
        )
        .reset_index()
        .rename(columns={"Vendor Names": "Vendor"})
        .sort_values("Tickets", ascending=False)
    )
    return out.round({"Avg downtime (h)": 2, "SLA compliance %": 1})[columns]


def client_leaderboard(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Per-client ticket count, average uptime and SLA compliance."""
    columns = ["Client", "Tickets", "Avg uptime %", "SLA compliance %"]
    scoped = df.dropna(subset=["Client_Uniq"])
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    out = (
        scoped.groupby("Client_Uniq", observed=True)
        .agg(
            Tickets=("Number", "size"),
            **{"Avg uptime %": ("Uptime %", lambda s: s.mean() * 100),
               "SLA compliance %": ("SLA Status", lambda s: s.eq("SLA Met").mean() * 100)},
        )
        .reset_index()
        .rename(columns={"Client_Uniq": "Client"})
        .sort_values("Tickets", ascending=False)
        .head(top_n)
    )
    return out.round({"Avg uptime %": 3, "SLA compliance %": 1})[columns]


def link_leaderboard(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Worst links by incident count, with their uptime and SLA record.

    Keyed on `Subject`, the human-readable link identifier; the UCID travels
    alongside so a row can be looked up in their own systems.
    """
    columns = ["Link", "UCID", "Tickets", "Avg downtime (h)",
               "Avg uptime %", "SLA compliance %"]
    scoped = df.dropna(subset=["Subject"])
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    out = (
        scoped.groupby("Subject", observed=True)
        .agg(
            UCID=("UCID", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
            Tickets=("Subject", "size"),
            **{"Avg downtime (h)": ("downtime_hours", "mean"),
               "Avg uptime %": ("Uptime %", lambda s: s.mean() * 100),
               "SLA compliance %": ("SLA Status",
                                    lambda s: s.eq("SLA Met").mean() * 100)},
        )
        .reset_index()
        .rename(columns={"Subject": "Link"})
        .sort_values("Tickets", ascending=False)
        .head(top_n)
    )
    return out.round({"Avg downtime (h)": 2, "Avg uptime %": 3,
                      "SLA compliance %": 1})[columns]


# ---------------------------------------------------------------------------
# Generic builders - used by the chatbot's render_chart tool
# ---------------------------------------------------------------------------

def build_chart(chart_type: str, rows, x_key: str, y_key: str,
                series_key: str | None = None, title: str = "") -> go.Figure:
    """Build a figure from tool-supplied rows.

    Raises ValueError on a bad request so the agent gets a correctable message
    back instead of the app crashing.
    """
    data = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if data.empty:
        return _empty_fig("The query returned no rows.")

    chart_type = (chart_type or "").strip().lower()
    if chart_type not in {"line", "bar", "donut", "heatmap"}:
        raise ValueError(
            f"Unknown chart_type '{chart_type}'. "
            "Use one of: line, bar, donut, heatmap, table."
        )

    for key, label in ((x_key, "x_key"), (y_key, "y_key")):
        if key not in data.columns:
            raise ValueError(
                f"{label}='{key}' is not a column in the data. "
                f"Available columns: {list(data.columns)}"
            )
    if series_key and series_key not in data.columns:
        raise ValueError(
            f"series_key='{series_key}' is not a column in the data. "
            f"Available columns: {list(data.columns)}"
        )

    if chart_type == "line":
        return _build_line(data, x_key, y_key, series_key, title)
    if chart_type == "bar":
        return _build_bar(data, x_key, y_key, series_key, title)
    if chart_type == "donut":
        return _donut(data[x_key], data[y_key],
                      theme.categorical_colors(len(data)), title)
    return _build_heatmap(data, x_key, y_key, series_key, title)


def _build_line(data, x_key, y_key, series_key, title) -> go.Figure:
    fig = go.Figure()
    if series_key:
        names = list(pd.unique(data[series_key].astype(str)))
        palette = theme.categorical_colors(len(names))
        parts = [(n, data[data[series_key].astype(str) == n]) for n in names]
    else:
        palette = [theme.PRIMARY]
        parts = [(y_key, data)]

    for i, (name, part) in enumerate(parts):
        part = part.sort_values(x_key)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=part[x_key], y=part[y_key], mode="lines+markers", name=str(name),
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color,
                            line=dict(color=theme.SURFACE, width=2)),
                hovertemplate="%{x}<br><b>%{y:,.2f}</b><extra>" + str(name) + "</extra>",
            )
        )
    fig.update_xaxes(title_text=x_key)
    fig.update_yaxes(title_text=y_key, rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return theme.style_fig(fig, title=title, showlegend=bool(series_key))


def _build_bar(data, x_key, y_key, series_key, title) -> go.Figure:
    fig = go.Figure()

    if series_key:
        names = list(pd.unique(data[series_key].astype(str)))
        palette = theme.categorical_colors(len(names))
        for i, name in enumerate(names):
            part = data[data[series_key].astype(str) == name]
            fig.add_trace(
                go.Bar(x=part[x_key].astype(str), y=part[y_key], name=name,
                       marker=dict(color=palette[i], **_BAR_LINE),
                       hovertemplate="%{x}<br><b>%{y:,.2f}</b><extra>"
                                     + name + "</extra>")
            )
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_xaxes(title_text=x_key)
        fig.update_yaxes(title_text=y_key, rangemode="tozero")
        return theme.style_fig(fig, title=title, showlegend=True)

    # Single series: go horizontal once there are more than a handful of bars,
    # which keeps long category names readable.
    horizontal = len(data) > 7
    ordered = data.sort_values(y_key, ascending=horizontal)
    labels = [f"{v:,.2f}".rstrip("0").rstrip(".") for v in ordered[y_key]]

    if horizontal:
        fig.add_trace(go.Bar(
            x=ordered[y_key], y=ordered[x_key].astype(str), orientation="h",
            marker=dict(color=theme.PRIMARY, **_BAR_LINE),
            text=labels, textposition="outside",
            textfont=dict(size=11, color=theme.INK_SECONDARY),
            hovertemplate="%{y}<br><b>%{x:,.2f}</b><extra></extra>"))
        fig.update_xaxes(title_text=y_key, showgrid=True, gridcolor=theme.GRIDLINE,
                         range=[0, _headroom(ordered[y_key])])
        fig.update_yaxes(title_text="", showgrid=False)
        height = max(340, min(26 * len(data) + 130, 620))
    else:
        fig.add_trace(go.Bar(
            x=ordered[x_key].astype(str), y=ordered[y_key],
            marker=dict(color=theme.PRIMARY, **_BAR_LINE),
            text=labels, textposition="outside",
            textfont=dict(size=11, color=theme.INK_SECONDARY),
            hovertemplate="%{x}<br><b>%{y:,.2f}</b><extra></extra>"))
        fig.update_xaxes(title_text=x_key)
        fig.update_yaxes(title_text=y_key, range=[0, _headroom(ordered[y_key])])
        fig.update_layout(bargap=0.35)
        height = 340

    return theme.style_fig(fig, title=title, height=height, showlegend=False)


def _build_heatmap(data, x_key, y_key, series_key, title) -> go.Figure:
    """`series_key` names the value column; x_key and y_key are the two axes."""
    if not series_key:
        raise ValueError(
            "heatmap needs series_key set to the column holding the cell value "
            "(x_key and y_key are the two axes)."
        )
    grid = data.pivot_table(index=y_key, columns=x_key, values=series_key, aggfunc="sum")
    if set(grid.index.astype(str)) <= set(WEEKDAY_ORDER):
        grid = grid.reindex([d for d in WEEKDAY_ORDER if d in set(grid.index)])
    grid = grid.iloc[::-1]

    fig = go.Figure(
        go.Heatmap(
            z=grid.values, x=[str(c) for c in grid.columns],
            y=[str(i) for i in grid.index],
            colorscale=theme.BLUE_SEQUENTIAL, xgap=2, ygap=2,
            colorbar=dict(title="", thickness=10, outlinewidth=0,
                          tickfont=dict(size=10, color=theme.INK_MUTED)),
            hovertemplate="%{y} / %{x}<br><b>%{z:,.2f}</b><extra></extra>",
        )
    )
    fig.update_xaxes(title_text=x_key, showgrid=False)
    fig.update_yaxes(title_text=y_key, showgrid=False)
    return theme.style_fig(fig, title=title, height=380, showlegend=False)

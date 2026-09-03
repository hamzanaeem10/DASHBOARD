"""Shared visual tokens and Plotly styling.

Every chart in the app - dashboard and chatbot alike - is styled through
:func:`style_fig`, so the two panels look like one system.

The categorical order below is fixed and must not be cycled or reordered: it was
chosen so that adjacent hues stay separable for colourblind readers. Charts also
carry direct value labels, which is what lets the lower-contrast slots (aqua,
yellow, magenta) be used on a light surface at all.
"""

from __future__ import annotations

# Chart surface and ink.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Categorical slots, in fixed assignment order.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

PRIMARY = CATEGORICAL[0]

# Single-hue blue ramp, light -> dark, for continuous magnitude (heatmaps).
BLUE_SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# Ordinal steps for discrete ordered categories (the downtime brackets).
# Starts at step 250 - anything lighter fails contrast on the light surface.
BLUE_ORDINAL = ["#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

# Reserved state colours. Never reused as a series colour.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"


def ordinal_colors(n: int) -> list[str]:
    """`n` steps of the ordinal ramp, spread across its range."""
    if n <= 0:
        return []
    if n >= len(BLUE_ORDINAL):
        return list(BLUE_ORDINAL) + [BLUE_ORDINAL[-1]] * (n - len(BLUE_ORDINAL))
    step = (len(BLUE_ORDINAL) - 1) / max(n - 1, 1)
    return [BLUE_ORDINAL[round(i * step)] for i in range(n)]


def categorical_colors(n: int) -> list[str]:
    """`n` categorical hues in fixed order (repeats only past slot 8)."""
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(max(n, 0))]


def style_fig(fig, *, title: str = "", height: int = 340, showlegend: bool | None = None):
    """Apply the shared chart chrome: recessive grid, muted axes, quiet frame."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=INK_PRIMARY, family=FONT_FAMILY),
            x=0, xanchor="left", y=0.97, yanchor="top",
        ) if title else None,
        height=height,
        margin=dict(l=8, r=16, t=44 if title else 16, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, size=12, color=INK_SECONDARY),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=AXIS,
            font=dict(family=FONT_FAMILY, size=12, color=INK_PRIMARY),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            title_text="", font=dict(size=11, color=INK_SECONDARY),
        ),
        separators=".,",
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)

    # Recessive axes: hairline horizontal grid only, no vertical clutter.
    fig.update_xaxes(
        showgrid=False, zeroline=False,
        linecolor=AXIS, tickcolor=AXIS,
        tickfont=dict(size=11, color=INK_MUTED), title_font=dict(size=11, color=INK_MUTED),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False,
        linecolor="rgba(0,0,0,0)", tickcolor=AXIS,
        tickfont=dict(size=11, color=INK_MUTED), title_font=dict(size=11, color=INK_MUTED),
    )
    return fig

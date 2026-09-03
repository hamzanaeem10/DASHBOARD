"""Tool definitions and execution for the chatbot agent.

Three tools are exposed to Claude:

* ``query_data``   - run read-only DuckDB SQL against the `tickets` table
* ``render_chart`` - turn rows from a previous query into a Plotly figure
* ``get_kpi``      - surface a single number as a metric card

Execution never raises into the agent loop: a failure comes back as an error
tool result so Claude can correct itself and retry.
"""

from __future__ import annotations

import json
import re

import pandas as pd

import charts

# Hard cap on rows handed back to the model, and on what any query may return.
MAX_ROWS = 500

# Read-only enforcement. Word boundaries keep these from matching column names
# such as "Created On" (which contains "create" only as a prefix).
# `replace` is deliberately absent: REPLACE() is a legitimate string function,
# and blocking `create` already stops CREATE OR REPLACE.
FORBIDDEN = re.compile(
    r"\b(drop|delete|update|insert|alter|create|attach|detach|copy|"
    r"pragma|install|load|export|import|set|call|vacuum|truncate|grant)\b",
    re.IGNORECASE,
)

# Anywhere in the query - a trailing OFFSET must not trick us into appending
# a second LIMIT clause.
_HAS_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)
_STARTS_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Schema shown to the model
# ---------------------------------------------------------------------------

SCHEMA = """Table: tickets  (one row per support ticket)

Column                | Type      | Notes
----------------------|-----------|--------------------------------------------
"Number"              | text      | Ticket ID. May be NULL - use COUNT(*) to count tickets.
"Issue Category"      | text      | ~15 values, e.g. WIRELESS ISSUES, VENDOR ISSUES, METRO FIBER
"Service Rqst"        | text      | COMPLAINTS, Information Request, Bandwidth Upgrade,
                      |           | Activity Notification, Site Access Request
"Created On"          | timestamp | Ticket creation time
"Month"               | text      | 3-letter month code: JAN..DEC (not ordered - sort on Created On)
"WeekNum"             | text      | 'Week N'
"UCID"                | bigint    | Client/circuit (link) ID. The literal 0 placeholder
                      |           | is nulled on load, so NULL means "not recorded".
"Vendor"              | text      | Full vendor name. NULL where no vendor.
"Vendor Names"        | text      | Short vendor code (~34 values). NULL where no vendor.
"Media Type"          | text      | e.g. Wireless Radio, Vendor Fiber, METRO-FIBER, VSAT, CMT
"Issue End"           | text      | Network End, Last Mile, Customer End
"Subject"             | text      | Link/site identifier, e.g. AB-0104KorangiRd-S-Fiber.
                      |           | This is the readable link name (~12.6k distinct).
"Client Name"         | text      | Raw client name (~765 values)
"Client_Uniq"         | text      | Deduplicated client name (~400 values) - PREFER THIS for
                      |           | per-client analysis. NULL for some rows.
"Resolution code"     | text      | ~94 codes, e.g. Fiber Break, Link Found UP
"FCR"                 | text      | 'yes' / 'no' - already lowercased
"closed_at"           | timestamp | Close time. NULL for ~2.3k rows. Do NOT use this to
                      |           | compute downtime - "closed_at - Created On" diverges
                      |           | from downtime_hours by a median of ~15.6 hours (they
                      |           | measure different things) and closed_at even
                      |           | precedes Created On on 51 rows. Always use
                      |           | downtime_hours for any duration question.
"Downtime Calc"       | interval  | Downtime duration. Use downtime_hours instead for maths.
"Bracket"             | text      | <= 4hours, 4-6 hours, 6-12 hours, 12-24 hours,
                      |           | > 24 hours, > 48 hours +
"Month Uptime [H]:mm" | interval  | Possible uptime for the month (constant per month)
"Uptime"              | interval  | Actual uptime
"Uptime %"            | double    | 0-1 scale. Multiply by 100 to show a percentage.
"Weekday"             | text      | Monday..Sunday
"TimeStamp"           | bigint    | Hour of day, 0-23
"timeslots"           | text      | Hour bucket label, e.g. '12am - 1am'
"FCR_category"        | text      | COMPLAINT or FCR
"SLA Bracket"         | text      | '>= 98.5%' or '< 98.5%'
"SLA Status"          | text      | 'SLA Met' or 'SLA Not Met'
"Link Status"         | text      | Active, Suspended, Pending on Workflow Tasks,
                      |           | Pending at WF Tasks, Terminated. NULL for ~11.5k rows.
"Zone"                | text      | North, South, Central. NULL for ~2.9k rows.
downtime_hours        | double    | Downtime in hours - USE THIS for averages and sums
created_date          | timestamp | Created On truncated to the day
created_month         | timestamp | Created On truncated to the month - USE THIS for
                      |           | monthly trends so results sort chronologically
"""

SQL_RULES = """SQL rules (DuckDB):
- Column names with spaces or mixed case MUST be double-quoted: "Issue Category".
  The three derived columns (downtime_hours, created_date, created_month) are
  lowercase and need no quotes.
- Count tickets with COUNT(*), never COUNT("Number") - that column has NULLs.
- Rates: AVG(CASE WHEN "SLA Status" = 'SLA Met' THEN 1.0 ELSE 0 END) * 100.
  FCR rate uses "FCR" = 'yes'. Uptime is 0-1, so AVG("Uptime %") * 100.
- Exclude NULL vendors/zones/clients with WHERE ... IS NOT NULL when ranking,
  otherwise the NULL bucket dominates the result.
- When ranking vendors or clients, add a volume floor (e.g. HAVING COUNT(*) >= 30)
  so a vendor with three tickets does not top an "worst average" leaderboard.
  Say in your answer that you applied it.
- Read-only: SELECT / WITH only.

House definitions (these come from the client's own monthly report - use them
so your answers reconcile with the dashboard and with what they already
circulate):
- "Incidents" / "complaints" means "Service Rqst" = 'COMPLAINTS' (50,472 rows),
  NOT all 56,140 tickets. The dashboard defaults to this scope. Unless the user
  clearly means every ticket type, filter to complaints and say that you did.
  Their headline Monthly Incident Summary additionally excludes
  "Issue Category" = 'ENTERPRISE PRODUCT'; every other view does not.
- MTTR (mean time to repair) = AVG(downtime_hours) over complaints. Report it as
  H:MM:SS. It is the same statistic as "average downtime".
- RFO (Reason For Outage) is their name for "Resolution code". If the user says
  RFO, query "Resolution code".
- A "link" is a "Subject" (readable name) or "UCID" (numeric id). Link-level
  questions group by one of those, not by client.
"""


def system_prompt(df: pd.DataFrame) -> str:
    """Build the agent's system prompt, grounded in the loaded dataset."""
    lo = pd.to_datetime(df["Created On"].min())
    hi = pd.to_datetime(df["Created On"].max())
    return f"""You are a data analyst for a network/ISP support-ticket dataset. \
You answer questions by querying the data - never from memory.

The loaded dataset has {len(df):,} tickets created between \
{lo:%d %b %Y} and {hi:%d %b %Y}.

{SCHEMA}
{SQL_RULES}

How to answer:
- ALWAYS call query_data before stating any number. Never estimate, never
  compute from memory, never reuse a number from earlier in the conversation
  without re-querying if the scope changed.
- If a query errors, read the message, fix the SQL and try again.
- When the answer is a trend, comparison, distribution or ranking - anything
  with more than about 5 data points - call render_chart rather than listing
  rows as text. Pass the rows you got back from query_data.
- When the answer is a single number (a rate, a count, an average), call
  get_kpi so it renders as a metric card.
- You may call query_data several times to build one answer, but do not
  re-run a query you have already run - reuse the rows you were given.
- Before you write your final answer, make sure you have called EITHER
  render_chart (for a trend, comparison, distribution or ranking) OR get_kpi
  (for a single number). Answering with plain text alone is only right when
  the user asked something that is genuinely not a number or a series.
- Keep the final text to one or two sentences: the chart or KPI card carries
  the detail, your text states the headline finding and any caveat (a volume
  floor you applied, rows excluded as NULL, a partial final month).
- The user is looking at a dashboard built on this same data, so your numbers
  must be reproducible from the table. Do not editorialise or invent context
  that is not in the data."""


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "query_data",
        "description": (
            "Run a read-only DuckDB SQL query against the `tickets` table and get "
            "the rows back as JSON. This is the only way to read the data - use it "
            "before stating any number. A LIMIT of "
            f"{MAX_ROWS} is applied automatically if you do not supply one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A single DuckDB SELECT (or WITH ... SELECT) statement. "
                        'Double-quote column names containing spaces, e.g. "Issue Category".'
                    ),
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "render_chart",
        "description": (
            "Render a Plotly chart inline in the chat. Prefer this over listing "
            "rows whenever the answer is a trend, comparison, distribution or "
            "ranking. Pass the rows returned by a previous query_data call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "donut", "heatmap", "table"],
                    "description": (
                        "line for time series; bar for comparisons and rankings; "
                        "donut for share-of-total across <= 6 categories; heatmap "
                        "for a two-dimensional grid; table for raw rows."
                    ),
                },
                "data": {
                    "type": "array",
                    "description": "Rows to plot, as a list of flat objects.",
                    "items": {"type": "object"},
                },
                "x_key": {
                    "type": ["string", "null"],
                    "description": (
                        "Column for the x-axis (bar/line), the category labels "
                        "(donut), or the horizontal axis (heatmap). Optional for table."
                    ),
                },
                "y_key": {
                    "type": ["string", "null"],
                    "description": (
                        "Column holding the value (bar/line/donut) or the vertical "
                        "axis (heatmap). Optional for table."
                    ),
                },
                "series_key": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional. Column that splits the data into multiple series. "
                        "For a heatmap this is REQUIRED and names the cell-value column."
                    ),
                },
                "title": {
                    "type": ["string", "null"],
                    "description": "Short chart title.",
                },
            },
            "required": ["chart_type", "data"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_kpi",
        "description": (
            "Show a single headline number as a metric card. Use this when the "
            "answer is one figure - a rate, a count, an average."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "What the number measures."},
                "value": {
                    "type": "string",
                    "description": "The formatted value, e.g. '92.5%' or '4,196'.",
                },
                "delta": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional comparison, e.g. '+2.1 pts vs last month'. "
                        "Only include it if you actually queried the comparison."
                    ),
                },
            },
            "required": ["label", "value"],
            "additionalProperties": False,
        },
    },
]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised for a bad tool request; surfaced to Claude as an error result."""


def guard_sql(sql: str) -> str:
    """Validate and normalise a query. Raises ToolError if it is not allowed."""
    if not sql or not sql.strip():
        raise ToolError("Empty query.")

    cleaned = sql.strip().rstrip(";").strip()

    if not _STARTS_READONLY.match(cleaned):
        raise ToolError("Only SELECT and WITH queries are allowed.")

    # Check for write keywords outside of string literals, so a query filtering
    # on a value like 'Update Monitering System' is not rejected.
    without_literals = re.sub(r"'(?:[^']|'')*'", "''", cleaned)
    match = FORBIDDEN.search(without_literals)
    if match:
        raise ToolError(
            f"'{match.group(0)}' is not permitted - this connection is read-only. "
            "Use a SELECT query instead."
        )

    if ";" in cleaned:
        raise ToolError("Only one statement at a time; remove the ';'.")

    if not _HAS_LIMIT.search(cleaned):
        cleaned = f"{cleaned}\nLIMIT {MAX_ROWS}"
    return cleaned


def run_query(con, sql: str) -> pd.DataFrame:
    """Execute a guarded query and return at most MAX_ROWS rows."""
    guarded = guard_sql(sql)
    try:
        result = con.sql(guarded).df()
    except Exception as exc:  # DuckDB raises many types; all are user-correctable
        raise ToolError(f"SQL error: {exc}") from exc
    return result.head(MAX_ROWS)


def _jsonify(df: pd.DataFrame) -> list[dict]:
    """Make a result frame JSON-safe (timestamps, intervals, NaN, numpy types)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif pd.api.types.is_timedelta64_dtype(out[col]):
            out[col] = out[col].dt.total_seconds() / 3600.0
    out = out.astype(object).where(pd.notna(out), None)
    return json.loads(json.dumps(out.to_dict(orient="records"), default=str))


def execute_tool(name: str, tool_input: dict, con, artifacts: list) -> tuple[str, bool]:
    """Run one tool call.

    Returns ``(content, is_error)``. Charts and KPI cards are appended to
    ``artifacts`` for the chat layer to render; the string returned here is only
    what Claude sees.
    """
    try:
        if name == "query_data":
            frame = run_query(con, tool_input.get("sql", ""))
            rows = _jsonify(frame)
            payload = {
                "row_count": len(rows),
                "columns": list(frame.columns),
                "rows": rows,
            }
            if len(rows) == MAX_ROWS:
                payload["note"] = (
                    f"Truncated at {MAX_ROWS} rows. Aggregate in SQL if you need more."
                )
            if not rows:
                payload["note"] = (
                    "No rows matched. Check spelling and casing of any literal "
                    "values against the schema notes."
                )
            return json.dumps(payload, default=str), False

        if name == "render_chart":
            data = tool_input.get("data") or []
            if not data:
                raise ToolError(
                    "No data supplied. Call query_data first and pass its rows in."
                )
            chart_type = (tool_input.get("chart_type") or "").strip().lower()
            title = tool_input.get("title") or ""

            if chart_type == "table":
                artifacts.append(
                    {"kind": "table", "data": pd.DataFrame(data), "title": title}
                )
                return f"Table rendered with {len(data)} rows.", False

            x_key = tool_input.get("x_key")
            y_key = tool_input.get("y_key")
            if not x_key or not y_key:
                raise ToolError(
                    "x_key and y_key are both required for a "
                    f"{chart_type or 'non-table'} chart."
                )
            fig = charts.build_chart(
                chart_type=chart_type,
                rows=data,
                x_key=x_key,
                y_key=y_key,
                series_key=tool_input.get("series_key"),
                title=title,
            )
            artifacts.append({"kind": "chart", "fig": fig})
            return (
                f"{chart_type} chart rendered with {len(data)} points. "
                "It is now visible to the user - describe the finding, do not "
                "list the underlying rows.",
                False,
            )

        if name == "get_kpi":
            label = tool_input.get("label")
            value = tool_input.get("value")
            if not label or value is None:
                raise ToolError("get_kpi needs both 'label' and 'value'.")
            artifacts.append(
                {
                    "kind": "kpi",
                    "label": str(label),
                    "value": str(value),
                    "delta": tool_input.get("delta"),
                }
            )
            return (
                f"KPI card rendered: {label} = {value}. It is now visible to the "
                "user - state the finding in one short sentence.",
                False,
            )

        raise ToolError(f"Unknown tool '{name}'.")

    except ToolError as exc:
        return str(exc), True
    except Exception as exc:  # never let a tool bug kill the chat
        return f"{type(exc).__name__}: {exc}", True

"""Loading and cleaning for the ticketing workbook.

The whole app runs off the single in-memory DataFrame produced here - the
dashboard filters it, the chatbot queries it through DuckDB - so the two panels
can never disagree on a number.
"""

from __future__ import annotations

import io
import os
import re

import duckdb
import pandas as pd
import streamlit as st

SHEET_NAME = "Tickets"

# The client's existing monthly report scopes every view to complaints, and its
# headline incident summary also drops ENTERPRISE PRODUCT. Keeping these here
# means the dashboard, the chatbot and the README all cite one definition.
COMPLAINT_SERVICE_RQST = "COMPLAINTS"
INCIDENT_EXCLUDED_CATEGORIES = ["ENTERPRISE PRODUCT"]

# Workbook used when nothing has been uploaded yet. Override with TICKETS_XLSX.
DEFAULT_DATA_PATH = os.environ.get(
    "TICKETS_XLSX",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026 Tickets.xlsx"),
)

# Must parse as datetimes; unparseable rows are dropped rather than crashing.
DATETIME_COLS = ["Created On", "closed_at"]

# Stored in Excel as durations.
TIMEDELTA_COLS = ["Downtime Calc", "Month Uptime [H]:mm", "Uptime"]

# These use a literal 0 to mean "no vendor".
VENDOR_COLS = ["Vendor", "Vendor Names"]

# Text columns worth stripping - stray whitespace silently splits categories.
TEXT_COLS = [
    "Number", "Issue Category", "Service Rqst", "Media Type", "Issue End",
    "Client Name", "Resolution code", "Bracket", "Weekday", "WeekNum",
    "FCR_category", "SLA Bracket", "SLA Status", "Link Status", "Zone",
    "Client_Uniq", "Month", "Subject",
]

# Logical (not alphabetical) ordering for the downtime buckets.
BRACKET_ORDER = [
    "<= 4hours",
    "4-6 hours",
    "6-12 hours",
    "12-24 hours",
    "> 24 hours",
    "> 48 hours +",
]

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

MONTH_ORDER = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]


def _clean_vendor(series: pd.Series) -> pd.Series:
    """Replace the literal 0 placeholder (int or string) with a null."""
    cleaned = series.replace([0, 0.0, "0"], pd.NA)
    if cleaned.dtype == object:
        # A blank string means the same thing as a missing vendor.
        cleaned = cleaned.replace(r"^\s*$", pd.NA, regex=True)
    return cleaned


def _merge_case_variants(series: pd.Series) -> pd.Series:
    """Fold spelling variants of the same category into one label.

    `METRO-FIBER`, `Metro-Fiber` and `Metro Fiber` are one media type recorded
    three ways. Rows are grouped on a case- and separator-insensitive key, and
    each group adopts the spelling that appears most often in the data - so the
    surviving label is always one the source actually uses.
    """
    values = series.dropna().astype(str)
    if values.empty:
        return series

    keys = values.str.upper().str.replace(r"[\s\-_/]+", "", regex=True)
    counts = values.groupby([keys, values]).size()

    canonical: dict[str, str] = {}
    for key, group in counts.groupby(level=0):
        # Most frequent spelling wins; ties break alphabetically for stability.
        top_count = group.max()
        canonical[key] = min(
            label for label, n in zip(group.index.get_level_values(1), group.values)
            if n == top_count
        )

    def _map(value):
        if pd.isna(value):
            return value
        key = re.sub(r"[\s\-_/]+", "", str(value).upper())
        return canonical.get(key, value)

    return series.map(_map).astype("string")


def _to_timedelta(series: pd.Series) -> pd.Series:
    """Coerce a column to timedelta regardless of how Excel handed it over.

    Depending on the cell format openpyxl may return real timedeltas,
    ``datetime.time`` objects, a fraction of a day as a float, or an
    ``HH:MM:SS`` string, so handle each rather than assuming one.
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series

    if pd.api.types.is_numeric_dtype(series):
        # Excel stores a duration as a fraction of a 24-hour day.
        return pd.to_timedelta(series.astype("float64"), unit="D", errors="coerce")

    def _coerce(value):
        if value is None or value is pd.NaT:
            return pd.NaT
        if isinstance(value, float) and pd.isna(value):
            return pd.NaT
        if isinstance(value, pd.Timedelta):
            return value
        if hasattr(value, "hour") and not hasattr(value, "year"):  # datetime.time
            return pd.Timedelta(
                hours=value.hour, minutes=value.minute, seconds=value.second,
                microseconds=getattr(value, "microsecond", 0),
            )
        return pd.to_timedelta(str(value), errors="coerce")

    return pd.to_timedelta(series.map(_coerce), errors="coerce")


def clean_tickets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply every cleaning rule. Returns the frame plus a short load report."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report: dict = {"rows_in": len(df)}

    # 1. Vendor placeholders -> null.
    for col in VENDOR_COLS:
        if col in df.columns:
            before = df[col].notna().sum()
            df[col] = _clean_vendor(df[col])
            report[f"{col}_nulled"] = int(before - df[col].notna().sum())

    # 2. FCR is inconsistently cased ('Yes'/'yes'/'No'/'no').
    if "FCR" in df.columns:
        df["FCR"] = df["FCR"].astype("string").str.strip().str.lower()

    # 3. timeslots carries trailing whitespace.
    if "timeslots" in df.columns:
        df["timeslots"] = df["timeslots"].astype("string").str.strip()

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # 4. Media Type records the same type under several spellings; fold them so
    #    one media type is one category everywhere.
    if "Media Type" in df.columns:
        before = df["Media Type"].nunique()
        df["Media Type"] = _merge_case_variants(df["Media Type"])
        report["media_types_merged"] = int(before - df["Media Type"].nunique())

    # 5. Datetimes: coerce bad cells to NaT, then drop rows with no creation
    #    date rather than letting one bad cell crash the load.
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Created On" in df.columns:
        unparseable = df["Created On"].isna()
        report["rows_dropped_unparseable_date"] = int(unparseable.sum())
        df = df.loc[~unparseable].copy()
    else:
        report["rows_dropped_unparseable_date"] = 0

    # Data-quality signals surfaced but not corrected: neither is common enough
    # to justify guessing at a fix, but both are worth showing rather than
    # hiding. Rows are kept as-is.
    if "Number" in df.columns:
        report["duplicate_ticket_numbers"] = int(
            df["Number"].dropna().duplicated().sum()
        )
    if {"Created On", "closed_at"} <= set(df.columns):
        report["closed_before_created"] = int(
            (df["closed_at"].notna() & (df["closed_at"] < df["Created On"])).sum()
        )

    # 6. Durations and the uptime percentage.
    for col in TIMEDELTA_COLS:
        if col in df.columns:
            df[col] = _to_timedelta(df[col])

    if "Uptime %" in df.columns:
        df["Uptime %"] = pd.to_numeric(df["Uptime %"], errors="coerce")
        peak = df["Uptime %"].max(skipna=True)
        # Guard against a workbook that stores 0-100 rather than 0-1.
        if pd.notna(peak) and peak > 1.5:
            df["Uptime %"] = df["Uptime %"] / 100.0

    for col in ("TimeStamp", "UCID"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # UCID uses the same literal-0 placeholder as the vendor columns. Left in,
    # it becomes the single largest "link" in any link-level ranking.
    if "UCID" in df.columns:
        placeholder = df["UCID"].eq(0)
        report["ucid_zero_nulled"] = int(placeholder.sum())
        df.loc[placeholder, "UCID"] = pd.NA

    # 7. Derived columns. SQL cannot easily average an INTERVAL and every trend
    #    chart wants a truncated date, so materialise them once here.
    if "Downtime Calc" in df.columns:
        df["downtime_hours"] = df["Downtime Calc"].dt.total_seconds() / 3600.0
    if "Created On" in df.columns:
        df["created_date"] = df["Created On"].dt.normalize()
        df["created_month"] = df["Created On"].dt.to_period("M").dt.to_timestamp()

    # 8. Ordered categorical so the Bracket chart sorts by severity, not A-Z.
    if "Bracket" in df.columns:
        present = [str(b) for b in pd.Series(df["Bracket"].dropna().unique())]
        order = [b for b in BRACKET_ORDER if b in present]
        order += [b for b in present if b not in order]
        df["Bracket"] = pd.Categorical(df["Bracket"], categories=order, ordered=True)

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def _read_workbook(source) -> pd.DataFrame:
    """Read only the Tickets sheet, from an uploaded file or a path."""
    try:
        return pd.read_excel(source, sheet_name=SHEET_NAME, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(
            f"The workbook has no sheet named '{SHEET_NAME}'. This app reads "
            "that sheet only."
        ) from exc


@st.cache_data(show_spinner="Loading and cleaning the workbook…", max_entries=2)
def load_data(file_bytes: bytes | None, path: str | None = None,
              signature: str = "") -> tuple[pd.DataFrame, dict]:
    """Load and clean the Tickets sheet.

    Cached on the upload's raw bytes, or on ``signature`` (path + mtime + size)
    for the bundled file - so re-runs are instant but a replaced file is
    re-read. Note the name must not start with an underscore: Streamlit
    excludes underscore-prefixed arguments from the cache key.
    """
    source = io.BytesIO(file_bytes) if file_bytes is not None else path
    return clean_tickets(_read_workbook(source))


def get_connection(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-process DuckDB connection with the frame registered as `tickets`."""
    con = duckdb.connect(database=":memory:")
    con.register("tickets", df)
    return con


def file_signature(path: str) -> str:
    """Path plus mtime, so the cache invalidates if the file is replaced."""
    try:
        return f"{path}:{os.path.getmtime(path)}:{os.path.getsize(path)}"
    except OSError:
        return path

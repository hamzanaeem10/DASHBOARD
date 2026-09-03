# Ticket Analytics

A single-page Streamlit app over a network/ISP support-ticket workbook. Two
panels, both reading the same in-memory DataFrame so the numbers can never
disagree:

- **Dashboard** — filters, five KPI cards, and thirteen charts/tables.
- **Ask the data** — a chatbot that writes its own SQL against the data and
  draws its own charts, via tool-calling.

Everything runs in memory. No database, no auth, no persistence — the app
resets on restart.

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### API key

The chatbot needs an LLM key. Copy the template and fill it in:

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

The app ships configured for **Groq**:

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
```

To use Claude instead, change two lines — no code changes:

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

`.env` is gitignored. Keys are only ever read from the environment, never
hardcoded.

### Data

By default the app loads `2026 Tickets.xlsx` from the project folder. Point it
somewhere else with `TICKETS_XLSX` in `.env`:

```ini
TICKETS_XLSX=E:\dashboardtickets\2026 Tickets.xlsx
```

Either way you can upload a different `.xlsx` in the sidebar at any time; the
upload replaces the in-memory data and re-registers it with DuckDB. Only the
sheet named **Tickets** is read.

## Run

```bash
streamlit run app.py
```

Opens on <http://localhost:8501>.

---

## Files

| File | Role |
|---|---|
| `app.py` | Entrypoint: page config, file upload, DuckDB registration, the two tabs |
| `data_loader.py` | `load_data()`, cleaning rules, cached on the file's bytes/mtime |
| `theme.py` | Colour tokens and the shared Plotly styling |
| `charts.py` | One function per chart; also the generic builders the chatbot uses |
| `dashboard.py` | The Dashboard tab: filters, KPI row, chart grid |
| `agent_tools.py` | Tool schemas, SQL safety, tool execution, the system prompt |
| `llm.py` | Provider layer — Groq or Anthropic behind one interface |
| `chatbot.py` | The chat tab: message loop and the tool-calling agent loop |
| `smoke_test.py` | Headless end-to-end check (not part of the app) |

---

## Cleaning rules applied on load

Applied in `data_loader.clean_tickets()`; the sidebar shows a load report of
what each pass did.

| Rule | Effect on the supplied workbook |
|---|---|
| Literal `0` in `Vendor` / `Vendor Names` → null | 39,711 rows per column |
| `FCR` lowercased | `Yes`/`No` folded into `yes`/`no` (715 rows) |
| `timeslots` trailing whitespace stripped | 55,456 rows |
| `Created On` / `closed_at` parsed as datetime | errors coerced to `NaT`; rows with no creation date are dropped and counted (0 in this file) |
| Duration columns parsed as timedelta | `Downtime Calc`, `Uptime`, `Month Uptime [H]:mm` |
| `Uptime %` numeric on a 0–1 scale | auto-divides by 100 if a workbook stores 0–100 |
| `Bracket` ordered by severity | so charts sort logically, not alphabetically |
| Text columns stripped | stray whitespace otherwise splits categories silently |
| **`Media Type` spelling variants merged** | 4 merged (20 → 16). See the note below. |

### Derived columns

Added on load and available to both panels — and to the chatbot's SQL:

- `downtime_hours` — `Downtime Calc` as a float. SQL cannot easily average an
  `INTERVAL`, so averages and sums use this.
- `created_date`, `created_month` — `Created On` truncated, so monthly trends
  sort chronologically rather than by the `Month` string (`APR`, `AUG`, …).

### Note: Media Type merging

This goes beyond the original spec, and was agreed during the build. The source
records the same media type under several spellings — `METRO-FIBER` (9,807),
`Metro-Fiber` (84) and `Metro Fiber` (7); likewise `ETHERNET`/`Ethernet` and
`MICROWAVE`/`Microwave`. Left alone these render as separate bars.

Rows are grouped on a case- and separator-insensitive key, and each group
adopts **the spelling that appears most often in the data**, so every surviving
label is one the source actually uses. About 350 rows out of 56,140 change
label; no row is dropped. To turn this off, delete the `_merge_case_variants`
call in `clean_tickets()`.

---

## The chatbot

A standard tool-calling loop — no LangChain. Conversation history lives in
`st.session_state`, so follow-up questions keep their context.

Three tools:

| Tool | What it does |
|---|---|
| `query_data` | Runs DuckDB SQL against the frame registered as `tickets` |
| `render_chart` | Builds a Plotly figure (line/bar/donut/heatmap/table) from rows a query returned |
| `get_kpi` | Renders a single number as a metric card |

`query_data` is **read-only and enforced as such**:

- Must start with `SELECT` or `WITH`.
- `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `ATTACH`, `COPY`,
  `PRAGMA` and friends are rejected. The check ignores string literals, so a
  legitimate query filtering on the category `'Update Monitering System'` still
  runs.
- Multiple statements are rejected.
- `LIMIT 500` is appended if the query has no limit, and results are capped at
  500 rows regardless.

Errors are returned to the model as tool results rather than raised, so it can
read the message, correct its SQL and retry instead of crashing the app.

**Scope:** the chatbot always queries the full dataset. The sidebar filters
apply to the Dashboard tab only — ask for a slice in the question instead
("…in the Central zone", "…since June").

---

## Design notes

- **Zone comparison uses two panels, not two axes.** Ticket count and SLA
  percentage have different units; putting them on a shared or secondary axis
  makes the visual comparison meaningless, so each gets its own panel.
- **Bars carry direct value labels.** Three of the categorical palette slots sit
  below 3:1 contrast on the light surface, so labels are load-bearing for
  legibility, not decoration.
- **`COUNT(*)`, never `COUNT("Number")`.** `Number` is null on 1,739 rows;
  counting it silently undercounts. The schema shown to the model says so too.
- The categorical palette is fixed-order and validated for colourblind
  separation; `SLA Status` uses reserved status colours rather than series hues.

## Verifying it works

```bash
python smoke_test.py
```

Checks the cleaning rules against the real file, builds every chart, exercises
the SQL guard (including that writes are blocked), asserts the dashboard's
pandas KPIs equal the chatbot's SQL for the same metric, and — if a key is
present — runs all four example questions through the live agent loop.

---

## Not built (out of scope for this MVP)

No database or persistence, no auth or multi-user support, no schema-migration
pipeline beyond the cleaning rules above, and no deployment config beyond
running locally or on Streamlit Community Cloud.

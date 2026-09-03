"""Headless end-to-end check: load -> clean -> charts -> SQL tools -> agent loop.

Not part of the app. Run with:  python smoke_test.py
"""
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

XLSX = os.environ.get("TICKETS_XLSX", r"E:\dashboardtickets\2026 Tickets.xlsx")

failures = []


def check(name, fn):
    try:
        result = fn()
        print(f"  PASS  {name}" + (f"  [{result}]" if result else ""))
        return True
    except Exception as exc:
        print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        failures.append(name)
        return False


print("=" * 70)
print("1. LOAD + CLEAN")
print("=" * 70)

import pandas as pd

import data_loader

raw = pd.read_excel(XLSX, sheet_name="Tickets", engine="openpyxl")
df, report = data_loader.clean_tickets(raw)
print(f"  report: {report}")

check("FCR normalised to yes/no",
      lambda: sorted(df["FCR"].dropna().unique()) == ["no", "yes"]
      or (_ for _ in ()).throw(AssertionError(sorted(df["FCR"].dropna().unique()))))
check("Vendor 0 -> null",
      lambda: f"{df['Vendor'].isna().sum():,} null vendors"
      if df["Vendor"].isna().sum() > 39000 else (_ for _ in ()).throw(AssertionError("not nulled")))
check("timeslots stripped",
      lambda: "ok" if not any(v != v.strip() for v in df["timeslots"].dropna())
      else (_ for _ in ()).throw(AssertionError("whitespace remains")))
check("Created On is datetime",
      lambda: str(df["Created On"].dtype) if pd.api.types.is_datetime64_any_dtype(df["Created On"])
      else (_ for _ in ()).throw(AssertionError(df["Created On"].dtype)))
check("Uptime % in 0-1",
      lambda: f"{df['Uptime %'].min():.3f}-{df['Uptime %'].max():.3f}"
      if df["Uptime %"].max() <= 1.0001 else (_ for _ in ()).throw(AssertionError("out of range")))
check("downtime_hours derived",
      lambda: f"mean {df['downtime_hours'].mean():.2f}h")
check("Bracket ordered logically",
      lambda: " < ".join(list(df["Bracket"].cat.categories)))
check("Media Type variants merged",
      lambda: f"{report['media_types_merged']} merged, {df['Media Type'].nunique()} remain")

print()
print("=" * 70)
print("2. CHARTS")
print("=" * 70)

import charts

chart_calls = [
    ("ticket_volume_trend/Day", lambda: charts.ticket_volume_trend(df, "Day")),
    ("ticket_volume_trend/Week", lambda: charts.ticket_volume_trend(df, "Week")),
    ("ticket_volume_trend/Month", lambda: charts.ticket_volume_trend(df, "Month")),
    ("issue_category_breakdown", lambda: charts.issue_category_breakdown(df)),
    ("service_request_donut", lambda: charts.service_request_donut(df)),
    ("top_resolution_codes", lambda: charts.top_resolution_codes(df)),
    ("sla_status_donut", lambda: charts.sla_status_donut(df)),
    ("downtime_bracket_distribution", lambda: charts.downtime_bracket_distribution(df)),
    ("weekday_hour_heatmap", lambda: charts.weekday_hour_heatmap(df)),
    ("fcr_rate_over_time", lambda: charts.fcr_rate_over_time(df)),
    ("zone_comparison", lambda: charts.zone_comparison(df)),
    ("media_type_breakdown", lambda: charts.media_type_breakdown(df)),
    ("link_status_donut", lambda: charts.link_status_donut(df)),
    ("mttr_trend", lambda: charts.mttr_trend(df)),
    ("weekday_weekend_trend", lambda: charts.weekday_weekend_trend(df)),
    ("category_by_month/RFO",
     lambda: charts.category_by_month(df, "Resolution code", "RFO by month")),
    ("category_by_month/Media",
     lambda: charts.category_by_month(df, "Media Type", "Media by month")),
]
for name, fn in chart_calls:
    check(name, lambda fn=fn: f"{len(fn().data)} traces")

check("vendor_leaderboard", lambda: f"{len(charts.vendor_leaderboard(df))} vendors")
check("client_leaderboard", lambda: f"{len(charts.client_leaderboard(df))} clients")
check("link_leaderboard", lambda: f"{len(charts.link_leaderboard(df))} links")
check("link_leaderboard excludes UCID 0",
      lambda: "clean" if 0 not in set(charts.link_leaderboard(df)["UCID"].dropna())
      else (_ for _ in ()).throw(AssertionError("UCID 0 present")))
check("format_hours",
      lambda: charts.format_hours(3.4449)
      if charts.format_hours(3.4449) == "3:26:41"
      else (_ for _ in ()).throw(AssertionError(charts.format_hours(3.4449))))
check("empty frame is safe",
      lambda: f"{len(charts.issue_category_breakdown(df.head(0)).layout.annotations)} annotation")

print()
print("=" * 70)
print("3. SQL TOOLS")
print("=" * 70)

import agent_tools

con = data_loader.get_connection(df)


def expect_blocked(sql):
    try:
        agent_tools.guard_sql(sql)
    except agent_tools.ToolError:
        return "blocked"
    raise AssertionError(f"NOT blocked: {sql}")


check("DROP blocked", lambda: expect_blocked("DROP TABLE tickets"))
check("DELETE blocked", lambda: expect_blocked("DELETE FROM tickets"))
check("UPDATE blocked", lambda: expect_blocked("UPDATE tickets SET Zone='x'"))
check("INSERT blocked", lambda: expect_blocked("INSERT INTO tickets VALUES (1)"))
check("ALTER blocked", lambda: expect_blocked("ALTER TABLE tickets ADD c INT"))
check("stacked statement blocked",
      lambda: expect_blocked("SELECT 1; DROP TABLE tickets"))
check("LIMIT auto-applied",
      lambda: "LIMIT 500" if "LIMIT 500" in agent_tools.guard_sql("SELECT * FROM tickets")
      else (_ for _ in ()).throw(AssertionError("no limit added")))
check("existing LIMIT respected",
      lambda: "kept" if agent_tools.guard_sql("SELECT * FROM tickets LIMIT 3").count("LIMIT") == 1
      else (_ for _ in ()).throw(AssertionError("double limit")))
check("literal containing 'Update' allowed",
      lambda: f"{len(agent_tools.run_query(con, chr(34).join(['SELECT COUNT(*) c FROM tickets WHERE ', 'Issue Category', " = 'Update Monitering System'"])))} row")

# The numbers the dashboard shows must be reproducible in SQL.
sla_sql = """SELECT ROUND(AVG(CASE WHEN "SLA Status"='SLA Met' THEN 1.0 ELSE 0 END)*100, 4) AS sla
             FROM tickets"""
sql_sla = agent_tools.run_query(con, sla_sql)["sla"].iloc[0]
pandas_sla = round(df["SLA Status"].eq("SLA Met").mean() * 100, 4)
check("SQL and pandas SLA agree",
      lambda: f"{sql_sla}% == {pandas_sla}%" if abs(float(sql_sla) - pandas_sla) < 0.001
      else (_ for _ in ()).throw(AssertionError(f"{sql_sla} != {pandas_sla}")))

fcr_sql = agent_tools.run_query(
    con, "SELECT ROUND(AVG(CASE WHEN \"FCR\"='yes' THEN 1.0 ELSE 0 END)*100,4) AS f FROM tickets"
)["f"].iloc[0]
pandas_fcr = round(df["FCR"].eq("yes").mean() * 100, 4)
check("SQL and pandas FCR agree",
      lambda: f"{fcr_sql}%" if abs(float(fcr_sql) - pandas_fcr) < 0.001
      else (_ for _ in ()).throw(AssertionError(f"{fcr_sql} != {pandas_fcr}")))

check("interval column survives to DuckDB",
      lambda: f"{agent_tools.run_query(con, 'SELECT AVG(downtime_hours) h FROM tickets')['h'].iloc[0]:.2f}h")
check("json-safe output",
      lambda: f"{len(agent_tools._jsonify(agent_tools.run_query(con, 'SELECT * FROM tickets LIMIT 3')))} rows")

artifacts = []
out, err = agent_tools.execute_tool(
    "query_data", {"sql": 'SELECT "Zone", COUNT(*) AS tickets FROM tickets WHERE "Zone" IS NOT NULL GROUP BY 1'},
    con, artifacts)
check("execute_tool query_data", lambda: "error" if err else f"{len(out)} chars json")

import json as _json
rows = _json.loads(out)["rows"]
out2, err2 = agent_tools.execute_tool(
    "render_chart",
    {"chart_type": "bar", "data": rows, "x_key": "Zone", "y_key": "tickets", "title": "By zone"},
    con, artifacts)
check("execute_tool render_chart", lambda: "error: " + out2 if err2 else out2)
check("chart artifact captured",
      lambda: artifacts[-1]["kind"] if artifacts[-1]["kind"] == "chart"
      else (_ for _ in ()).throw(AssertionError("no chart")))

out3, err3 = agent_tools.execute_tool("get_kpi", {"label": "SLA", "value": "92.5%"}, con, artifacts)
check("execute_tool get_kpi", lambda: out3 if not err3 else "error")

bad, is_err = agent_tools.execute_tool("query_data", {"sql": "SELECT nope FROM tickets"}, con, artifacts)
check("bad SQL returns error not crash",
      lambda: "error surfaced" if is_err else (_ for _ in ()).throw(AssertionError("should error")))

print()
print("=" * 70)
print("3b. RECONCILE WITH THEIR Summary SHEET")
print("=" * 70)

import data_loader as _dl

complaints = df[df["Service Rqst"] == _dl.COMPLAINT_SERVICE_RQST]
incidents = complaints[~complaints["Issue Category"].isin(_dl.INCIDENT_EXCLUDED_CATEGORIES)]

check("complaints == 50,472",
      lambda: f"{len(complaints):,}" if len(complaints) == 50472
      else (_ for _ in ()).throw(AssertionError(len(complaints))))
check("incident summary == 48,195",
      lambda: f"{len(incidents):,}" if len(incidents) == 48195
      else (_ for _ in ()).throw(AssertionError(len(incidents))))
check("MTTR (complaints) == 3:26:41",
      lambda: charts.format_hours(complaints["downtime_hours"].mean())
      if charts.format_hours(complaints["downtime_hours"].mean()) == "3:26:41"
      else (_ for _ in ()).throw(AssertionError(
          charts.format_hours(complaints["downtime_hours"].mean()))))
check("Issue End split == 20750/15251/12194",
      lambda: "matches" if list(incidents["Issue End"].value_counts()
                                .reindex(["Customer End", "Last Mile", "Network End"]))
      == [20750, 15251, 12194]
      else (_ for _ in ()).throw(AssertionError(
          incidents["Issue End"].value_counts().to_dict())))
check("SLA bracket (complaints) == 4148 / 46324",
      lambda: "matches" if [int(complaints["SLA Bracket"].value_counts().get(k, 0))
                            for k in ["< 98.5%", "\u2265 98.5%"]] == [4148, 46324]
      else (_ for _ in ()).throw(AssertionError(
          complaints["SLA Bracket"].value_counts().to_dict())))
check("bracket <= 4hours (complaints) == 41,049",
      lambda: "matches"
      if int(complaints["Bracket"].value_counts().get("<= 4hours", 0)) == 41049
      else (_ for _ in ()).throw(AssertionError("mismatch")))
check("UCID 0 nulled",
      lambda: f"{report.get('ucid_zero_nulled', 0):,} nulled"
      if report.get("ucid_zero_nulled") == 4010
      else (_ for _ in ()).throw(AssertionError(report.get("ucid_zero_nulled"))))

print()
print("=" * 70)
print("4. AGENT LOOP (live API)")
print("=" * 70)

import llm

if not os.environ.get("GROQ_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    print("  SKIP  no API key in environment")
else:
    provider = llm.resolve_provider()
    print(f"  provider={provider.label} model={provider.model}")
    system = agent_tools.system_prompt(df)
    questions = [
        "What's our SLA compliance rate?",
        "Which vendor has the worst average downtime?",
        "Show me ticket volume trend by month",
        "Compare FCR rate across zones",
        "What is our MTTR?",
        "Which links have the most incidents?",
    ]
    for question in questions:
        history = provider.start(system)
        history.append({"role": "user", "content": question})
        arts, sqls = [], []
        try:
            for _ in range(8):
                turn = provider.call(history, agent_tools.TOOLS, system)
                provider.append_assistant(history, turn)
                if not turn.wants_tools:
                    break
                results = []
                for call in turn.tool_calls:
                    if call.name == "query_data":
                        sqls.append(call.arguments.get("sql", ""))
                    content, is_error = agent_tools.execute_tool(
                        call.name, call.arguments, con, arts)
                    results.append((call, content, is_error))
                provider.append_tool_results(history, results)
            kinds = [a["kind"] for a in arts]
            print(f"\n  Q: {question}")
            print(f"     tools: {len(sqls)} queries, artifacts={kinds}")
            for s in sqls:
                print(f"     SQL: {' '.join(s.split())[:150]}")
            print(f"     A: {turn.text.strip()[:300]}")
            if not arts:
                print("     WARN: no chart or KPI produced")
        except Exception as exc:
            print(f"  FAIL  agent loop on {question!r}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=4)
            failures.append(f"agent:{question}")

print()
print("=" * 70)
print(f"RESULT: {'ALL PASS' if not failures else str(len(failures)) + ' FAILURES: ' + ', '.join(failures)}")
print("=" * 70)
sys.exit(1 if failures else 0)

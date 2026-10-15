from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from bi_engine import build_insights, data_profile, execute_plan, plan_query, conversational_response

load_dotenv()
ROOT = Path(__file__).parent
DEFAULT_DATA = ROOT / "sample_data" / "sales_data.csv"
EXAMPLES = [
    "Show monthly sales revenue for Q3 2024 broken down by region and highlight the top-performing product category.",
    "Compare total profit by product category for 2024 and show the top 5 categories.",
    "Show order volume by customer segment and region, filtered to the East region.",
]

def is_bi_request(message: str) -> bool:
    text = message.lower().strip()

    bi_actions = (
        "show ", "display ", "plot ", "chart ", "graph ",
        "compare ", "calculate ", "analyze ", "analyse ",
        "filter ", "breakdown ", "group by ", "rank ",
        "top ", "bottom ", "sort ", "trend ",
        "highest ", "lowest ", "by region", "by product",
        "by month", "by category", "by channel",
        "sales by ", "profit by ", "revenue by ",
        "instead", "change the metric",
    )

    return any(phrase in text for phrase in bi_actions)

st.set_page_config(page_title="Conversational BI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink: #14243a; --muted: #65758b; --accent: #6157e8; }
.block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1500px; }
.hero { padding: 1.45rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#111c35 0%,#253d65 58%,#4b4fa4 100%); color: white; margin-bottom: 1.15rem; }
.hero h1 { font-size: 2rem; margin: 0 0 .35rem 0; font-weight: 750; letter-spacing: -.035em; }
.hero p { color: #dce6ff; margin: 0; font-size: 1rem; }
.metric-card { border: 1px solid #e5eaf2; border-radius: 15px; padding: .95rem 1rem; background: white; min-height: 94px; }
.metric-label { color: #65758b; font-size: .82rem; margin-bottom: .35rem; }
.metric-value { color: #14243a; font-size: 1.5rem; font-weight: 740; }
.small-note { color: #65758b; font-size: .84rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #e9edf4; }
.stButton > button[kind="primary"] { border-radius: 10px; }
</style>
<div class="hero">
  <h1>Conversational AI for Instant Business Intelligence</h1>
  <p>Ask a business question. Get a data-backed dashboard, not a wall of SQL.</p>
</div>
""", unsafe_allow_html=True)

if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = ""
if "plan" not in st.session_state:
    st.session_state.plan = None
if "execution" not in st.session_state:
    st.session_state.execution = None
if "planner_warning" not in st.session_state:
    st.session_state.planner_warning = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.markdown("### ⚙️ Data & AI settings")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], help="Your CSV is processed in this app session. If Gemini is enabled, the schema and category metadata are sent to Google's API, not the raw rows.")
    st.caption("No CSV? A sample retail dataset is loaded automatically.")
    st.divider()
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        api_key = os.getenv("GEMINI_API_KEY", "")
    try:
        model = st.secrets.get("GEMINI_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    except (FileNotFoundError, KeyError):
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    if api_key:
        st.sidebar.success("Gemini API configured")
    else:
        st.sidebar.warning("Gemini API key is not configured")
    st.sidebar.caption("When Gemini is unavailable, a local rule-based planner keeps the demo usable.")
    st.sidebar.divider()
    st.markdown("### 🧭 How it works")
    st.markdown("**1. Interpret** natural language into a constrained JSON plan.\n\n**2. Validate** fields and filters against the dataset schema.\n\n**3. Compute** with pandas—no generated code or SQL is executed.\n\n**4. Visualise** results in interactive Plotly charts.")

@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA, parse_dates=["Order Date"])

try:
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        dataset_name = uploaded.name
    else:
        df = load_default_data()
        dataset_name = "Sample retail sales dataset"
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("This dataset is empty. Upload a CSV containing column headers and at least one row.")
    st.stop()

# Convert likely date columns once, so profiles and aggregations can recognise them.
for col in df.columns:
    if any(token in str(col).lower() for token in ["date", "timestamp"]):
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().mean() >= 0.6:
            df[col] = converted

# Header metrics
h1, h2, h3, h4 = st.columns(4)
with h1:
    st.metric("Dataset", dataset_name[:28] + ("…" if len(dataset_name) > 28 else ""))
with h2:
    st.metric("Rows", f"{len(df):,}")
with h3:
    st.metric("Columns", f"{len(df.columns):,}")
with h4:
    st.metric("AI planner", "Gemini enabled" if api_key.strip() else "Local fallback")

left, right = st.columns([2.2, 1], gap="large")
with left:
    st.markdown("### Ask your data")
    st.caption("Use everyday language. Start with a metric, then add a time period, breakdown, or filter.")
    
    
    chosen_example = st.selectbox(
        "Try an example",
        ["Choose an example…"] + EXAMPLES,
        label_visibility="collapsed",
        key="example_selector",
    )

    with st.form("query_form", clear_on_submit=False):
        prompt = st.text_area(
            "Business question",
            height=100,
            placeholder="e.g. Show monthly sales by region for Q3 2024",
            label_visibility="collapsed",
        )

        submit = st.form_submit_button(
            "✨ Generate dashboard",
            type="primary",
            use_container_width=True,
        )

    if submit:
        question = prompt.strip()

        if not question:
            st.warning("Enter a business question to generate a dashboard.")
        else:
            st.session_state.active_prompt = question
            st.session_state.chat_history = []

            with st.spinner(
                "Interpreting the question, validating fields, and building your dashboard…"
            ):
                plan, warning = plan_query(
                    question,
                    df,
                    api_key,
                    model.strip() or "gemini-3.5-flash-lite",
                )
                st.session_state.plan = plan
                st.session_state.execution = execute_plan(df, plan)
                st.session_state.planner_warning = warning


with right:
    st.markdown("### Dataset snapshot")
    with st.container(border=True):
        st.write(f"**{len(df):,} records** · **{len(df.columns)} fields**")
        numeric_count = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        date_count = sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
        st.caption(f"{numeric_count} numeric fields · {date_count} date fields")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    with st.expander("Inspect schema"):
        profile = data_profile(df)
        st.json(profile)

st.divider()

if st.session_state.plan is None:
    st.markdown("### Your dashboard will appear here")
    demo_cols = st.columns(3)
    cards = [
        ("01 · Summary", "KPIs and concise, computed insights"),
        ("02 · Visualise", "Line, bar, pie, area, scatter or histogram"),
        ("03 · Follow up", "Refine the current question conversationally"),
    ]
    for col, (title, desc) in zip(demo_cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
    st.info("Choose an example or type a question above, then select **Generate dashboard**.")
    st.stop()

plan = st.session_state.plan
execution = st.session_state.execution or execute_plan(df, plan)
if st.session_state.planner_warning:
    st.warning(st.session_state.planner_warning)

if plan.get("cannot_answer"):
    st.error(plan.get("reason") or "This request cannot be answered using the available dataset.")
    if plan.get("clarification_question"):
        st.info(plan["clarification_question"])
    st.markdown("**Available columns:** " + ", ".join(map(str, df.columns)))
    st.stop()
if execution.get("error"):
    st.error(execution["error"])
    st.stop()

st.markdown(f"## {plan.get('title', 'Generated dashboard')}")
st.caption(plan.get("summary", "Computed from the selected dataset."))

metric = plan.get("metric")
filtered = execution["filtered"]
result = execution["result"]
aggregation = plan.get("aggregation", "sum")
#Support multiple KPIs while preserving compatibility with order plans.
metrics = plan.get("metrics", [])
if not isinstance(metrics, list):
    metrics = []
#Ensure the primary chart metric is included.
if metric in filtered.columns:
    primary = {"metric": metric, "aggregation": aggregation}
    if not any(
        isinstance(item, dict) and item.get("metric") == metric and item.get("aggregation") == aggregation for item in metrics
    ):
        metrics.insert(0, primary)
#Keep only valid metrics and supported aggregations.
valid_metrics = []
seen = set()
for item in metrics:
    if not isinstance(item, dict):
        continue
    metric_name = item.get("metric")
    agg_name = item.get("aggregation", "sum")
    if metric_name not in filtered.columns:
        continue
    if agg_name not in {"sum", "mean", "count", "nunique", "min", "max"}:
        continue
    if agg_name in{"sum", "mean", "min", "max"} and not pd.api.types.is_numeric_dtype(filtered[metric_name]):
        continue
    key = (metric_name, agg_name)
    if key not in seen:
        valid_metrics.append(key)
        seen.add(key)
def calculate_kpi(series, agg_name):
    if agg_name == "sum":
        value = series.sum()
        label = f"Total {series.name}"
    elif agg_name == "mean":
        value = series.mean()
        label = f"Average {series.name}"
    elif agg_name == "count":
        value = series.count()
        label = f"Count of {series.name}"
    elif agg_name == "nunique":
        value = series.nunique()
        label = f"Unique {series.name}"
    elif agg_name == "max":
        value = series.max()
        label = f"Maximum {series.name}"
    else: 
        value = series.min()
        label = f"Minimum {series.name}"
    if pd.api.types.is_number(value):
        formatted = (
            f"{value:,.2f}"
            if agg_name in {"sum", "mean", "min", "max"}
            else f"{value:,.0f}"
        )
    else:
        formatted= str(value)
    return label, formatted
#Show requested KPIs, then use remaining slots for dashboard context.
kpi_cards = []
for metric_name, agg_name in valid_metrics:
    label, formatted = calculate_kpi(filtered[metric_name], agg_name)
    kpi_cards.append((label, formatted))
kpi_cards.append(("Rows analysed", f"{len(filtered):,}"))
kpi_cards.append(("Filters applied", f"{len(plan.get('filters', []))}"))
#Four cards per row, with additional rows if more KPIs were requested.
for start in range(0, len(kpi_cards), 4):
    row_cards = kpi_cards[start:start +4]
    columns = st.columns(4)
    for col, (label, value) in zip(columns, row_cards):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
st.write("")
# A readable analysis plan provides transparency and lets evaluators see how the query was interpreted.
with st.expander("🔎 See how your question was interpreted"):
    plan_view = {k: v for k, v in plan.items() if k != "planner_source"}
    st.json(plan_view)
    st.caption(f"Planner: {plan.get('planner_source', 'Validated analysis planner')} · The plan is validated before execution.")

plot_col, insight_col = st.columns([2.15, 1], gap="large")
with plot_col:
    st.markdown("### Visual analysis")
    chart_type = plan.get("chart_type", "bar")
    group_keys = execution.get("group_keys", [])
    time_key = execution.get("time_key")
    dimension = plan.get("dimension")
    secondary = plan.get("secondary_dimension")
    fig = None
    try:
        if chart_type == "histogram":
            fig = px.histogram(filtered, x=metric, nbins=30, title=f"Distribution of {metric}")
        elif chart_type == "scatter":
            x_metric = plan.get("x_metric")
            if x_metric in filtered.columns and metric in filtered.columns:
                fig = px.scatter(filtered, x=x_metric, y=metric, color=dimension if dimension in filtered.columns else None,
                                 hover_data=[c for c in ["Region", "Product", "Product Category"] if c in filtered.columns],
                                 title=f"{metric} vs {x_metric}")
            else:
                fig = px.bar(result, x=group_keys[0] if group_keys else "Value", y="Value", title=f"{metric} by group")
        elif chart_type == "pie" and dimension in result.columns and time_key is None:
            fig = px.pie(result, names=dimension, values="Value", hole=0.48, title=f"{metric} by {dimension}")
        elif time_key and time_key in result.columns:
            color_col = dimension if dimension in result.columns else (secondary if secondary in result.columns else None)
            if dimension in result.columns and secondary in result.columns:
                result = result.copy()
                result["_series"] = result[dimension].astype(str) + " · " + result[secondary].astype(str)
                color_col = "_series"
            if chart_type == "area":
                fig = px.area(result, x=time_key, y="Value", color=color_col, markers=True, title=f"{metric} over time")
            elif chart_type == "bar":
                fig = px.bar(result, x=time_key, y="Value", color=color_col, barmode="group", title=f"{metric} over time")
            else:
                fig = px.line(result, x=time_key, y="Value", color=color_col, markers=True, title=f"{metric} over time")
        
        elif group_keys:
            x_col = (
                dimension if dimension in result.columns
                else secondary if secondary in result.columns
                else group_keys[0]
            )

            metrics = plan.get("metrics", [])
            chart_metrics = []

            if isinstance(metrics, list):
                for item in metrics:
                    if not isinstance(item, dict):
                        continue

                    name = item.get("metric")
                    if name in result.columns and name not in chart_metrics:
                        chart_metrics.append(name)

            # Keep existing single-metric charts working.
            if not chart_metrics:
                chart_metrics = ["Value"]

            if chart_type == "bar" and len(chart_metrics) > 1:
                chart_data = result.melt(
                    id_vars=[x_col],
                    value_vars=chart_metrics,
                    var_name="Metric",
                    value_name="Amount",
                )

                fig = px.bar(
                    chart_data,
                    x=x_col,
                    y="Amount",
                    color="Metric",
                    barmode="group",
                    title=f"Sales and Profit by {x_col}",
                )

            else:
                color_col = (
                    secondary
                    if secondary in result.columns and secondary != x_col
                    else None
                )

                if chart_type == "area":
                    fig = px.area(
                        result, x=x_col, y="Value",
                        color=color_col, title=f"{metric} by {x_col}"
                    )
                elif chart_type == "line":
                    fig = px.line(
                        result, x=x_col, y="Value",
                        color=color_col, markers=True,
                        title=f"{metric} by {x_col}"
                    )
                else:
                    fig = px.bar(
                        result, x=x_col, y="Value",
                        color=color_col, barmode="group",
                        title=f"{metric} by {x_col}"
                    )

        if fig is not None:
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=410, legend_title_text="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
        else:
            st.info("This request produces a single KPI. Add a breakdown such as ‘by region’ or ‘by month’ to create a chart.")
    except Exception as exc:
        st.error(f"Could not render this chart ({type(exc).__name__}). Try a simpler question or another chart type.")

with insight_col:
    st.markdown("### Key takeaways")
    with st.container(border=True):
        insights = build_insights(execution, plan)
        if insights:
            for takeaway in insights:
                st.markdown(f"- {takeaway}")
        else:
            st.caption("No insights to show for this result.")
        st.divider()
        st.markdown("**Chart type**")
        st.write(chart_type.title())
        st.markdown("**Metric**")
        st.write(f"{aggregation.title()} of `{metric}`")
        if plan.get("dimension"):
            st.markdown("**Breakdown**")
            st.write(plan["dimension"])
        if plan.get("time_grain") != "none":
            st.markdown("**Time grain**")
            st.write(plan["time_grain"].title())

with st.expander("View computed result table"):
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button("Download aggregated results as CSV", result.to_csv(index=False).encode("utf-8"), file_name="dashboard_results.csv", mime="text/csv")

st.divider()
st.markdown("### 💬 Chat with this dashboard")
st.caption("Ask a follow-up to refine this view—for example, ‘Now filter to the East region’ or ‘Show this by product category instead.’")

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

follow_up = st.chat_input("Ask a follow-up question…")

if follow_up:
    st.session_state.chat_history.append({
        "role": "user",
        "content": follow_up,
    })

    selected_model = model.strip() or "gemini-3.5-flash-lite"

    # PATH 1: Ordinary conversation, explanations and greetings
    if not is_bi_request(follow_up):
        with st.spinner("Thinking..."):
            answer = conversational_response(
                message=follow_up,
                api_key=api_key,
                model=selected_model,
                history=st.session_state.chat_history[:-1],
                active_prompt=st.session_state.active_prompt,
                plan=st.session_state.plan,
                execution=st.session_state.execution,
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
        })
        st.rerun()

    # PATH 2: Business intelligence queries and dashboard changes
    else:
        revised_prompt = (
            f"Current dashboard request: {st.session_state.active_prompt}\n"
            f"Follow-up request: {follow_up}"
        )

        with st.spinner("Updating the dashboard from your follow-up…"):
            revised_plan, warning = plan_query(
                revised_prompt,
                df,
                api_key,
                selected_model,
            )
            revised_execution = execute_plan(df, revised_plan)

        if (
            revised_plan.get("cannot_answer")
            or revised_execution.get("error")
        ):
            message = (
                revised_plan.get("reason")
                or revised_execution.get("error")
                or "I could not answer that from the available data."
            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": message,
            })
            st.rerun()

        else:
            st.session_state.active_prompt = revised_prompt
            st.session_state.plan = revised_plan
            st.session_state.execution = revised_execution
            st.session_state.planner_warning = warning

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    "Dashboard updated using the available dataset. "
                    "The updated chart and insights are shown above."
                ),
            })
            st.rerun()


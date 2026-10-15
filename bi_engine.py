"""Core natural-language planning and safe dataframe execution for Conversational BI."""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "metric": {"type": ["string", "null"]},
        "aggregation": { "type": "string", "enum": ["sum", "mean", "count", "nunique", "min", "max"]},
        "metrics": { "type": "array", "items":{ "type": "object", "properties": {"metric": {"type": "string"}, "aggregation": { "type": "string", "enum": ["sum", "mean", "count", "nunique", "min", "max"]}},
                                               "required": ["metric", "aggregation"], "additionalProperties": False }}, 
                                               "dimension": {"type" : ["string", "null"]},
        "secondary_dimension": {"type": ["string", "null"]},
        "date_column": {"type": ["string", "null"]},
        "time_grain": {"type": "string", "enum": ["none", "day", "week", "month", "quarter", "year"]},
        "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "area", "scatter", "histogram"]},
        "x_metric": {"type": ["string", "null"]},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "operator": {"type": "string", "enum": ["eq", "contains", "gt", "gte", "lt", "lte", "in", "between"]},
                    "value": {"type": ["string", "number", "null"]},
                    "values": {"type": "array", "items": {"type": ["string", "number"]}},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                },
                "required": ["column", "operator", "value", "values", "start_date", "end_date"],
            },
        },
        "sort": {"type": "string", "enum": ["asc", "desc", "none"]},
        "limit": {"type": "integer"},
        "cannot_answer": {"type": "boolean"},
        "reason": {"type": "string"},
        "clarification_question": {"type": "string"},
    },
    "required":[
        "title", "summary", "metric", "aggregation", "metrics", "dimension", "secondary_dimension", "date_column",
        "time_grain", "chart_type", "x_metric", "filters", "sort", "limit", "cannot_answer", "reason", "clarification_question", 
    ],
}

AGGREGATIONS = {"sum", "mean", "count", "nunique", "min", "max"}
CHARTS = {"bar", "line", "pie", "area", "scatter", "histogram"}
GRAINS = {"none", "day", "week", "month", "quarter", "year"}
OPERATORS = {"eq", "contains", "gt", "gte", "lt", "lte", "in", "between"}


def _column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}


def _column_matches(df: pd.DataFrame, candidate: Any) -> str | None:
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    lookup = _column_lookup(df)
    return lookup.get(candidate.strip().lower())


def data_profile(df: pd.DataFrame) -> dict[str, Any]:
    """A compact schema summary. Raw row data is not included in the LLM prompt."""
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        item: dict[str, Any] = {"name": str(col), "dtype": str(s.dtype), "nulls": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            item["min"] = _json_safe(s.min())
            item["max"] = _json_safe(s.max())
        elif pd.api.types.is_datetime64_any_dtype(s):
            item["min"] = _json_safe(s.min())
            item["max"] = _json_safe(s.max())
        else:
            vals = s.dropna().astype(str).unique().tolist()
            item["distinct_count"] = len(vals)
            if len(vals) <= 25:
                item["values"] = vals
            else:
                item["sample_values"] = vals[:12]
        columns.append(item)
    return {"rows": int(len(df)), "columns": columns}


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _gemini_plan(prompt: str, df: pd.DataFrame, api_key: str, model: str) -> dict[str, Any]:
    """Ask Gemini for a constrained analysis plan, never for executable code or SQL."""
    from google import genai

    profile = data_profile(df)
    instruction = f"""You are the query planner for a business intelligence dashboard.
Convert the user request into one JSON analysis plan matching the supplied JSON schema.
Rules:
- Use only exact column names and real categorical values from this data profile.
- Never invent a column, value, metric, or factual result. The profile contains schema and metadata, not all rows.
- Do not return Python, SQL, expressions, or executable code. The application performs the computation.
- Numeric metric candidates should be used for sum/mean/min/max. For order counts, prefer a unique identifier column with aggregation=nunique.
- Use line/area for time series, bar for comparisons, pie for a small part-to-whole comparison, scatter for relationships between two numeric measures, histogram for distributions.
- Set date_column and time_grain when time grouping is requested. Use month for monthly, quarter for quarterly, year for yearly.
- Add filters only when the user explicitly requests or implies them. If a requested field/measure is absent, set cannot_answer=true and explain briefly. If the question is materially ambiguous, ask a clarification question rather than fabricate.
- If the prompt includes follow-up context, retain the original analysis and apply the requested change.
- Keep limit between 1 and 100. Use sort=desc for top/best/highest and sort=asc for bottom/lowest.
- If the user requests multiple metrics or KPIs, include every requested metric in the "metrics" array.
- For a request for total sales and total profit, include the exact dataset columns Sales and Profit, each with aggregation "sum", if both columns exist.
- Each item in the "metrics" array must contain a metric field and an aggregation field.
- Always include the primary chart metric in the "metrics" array.
- Keep the existing "metric" and "aggregation" fields for backward compatibility.
- Never omit a requested valid KPI just because another metric is selected for the chart.

DATA PROFILE:
{json.dumps(profile, ensure_ascii=False, default=str)}

USER REQUEST:
{prompt}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=instruction,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PLAN_SCHEMA,
            "temperature": 0.1,
        },
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response.")
    parsed = json.loads(text)
    parsed["planner_source"] = "Gemini API"
    return normalize_plan(parsed, df)


def _find_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        name = str(col).lower()
        if any(token in name for token in ("date", "time", "month", "timestamp")):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.6:
                return col
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def fallback_plan(prompt: str, df: pd.DataFrame) -> dict[str, Any]:
    """Deterministic fallback for demos and quota/API outages; supports common BI intents."""
    q = prompt.lower()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    date_col = _find_date_column(df)
    cols = list(df.columns)

    def find_col(needles: list[str], candidates: list[Any] | None = None) -> Any | None:
        haystack = candidates if candidates is not None else cols
        for needle in needles:
            for c in haystack:
                if needle in str(c).lower():
                    return c
        return None

    metric: Any | None = None
    order_count_intent = any(w in q for w in ["number of orders", "order count", "how many orders", "count orders", "count of orders", "order volume", "orders by", "order volume by"])
    if order_count_intent:
        metric = find_col(["order id", "order_id", "order number", "order"], cols)
    elif any(w in q for w in ["profit", "margin", "profitable"]):
        metric = find_col(["profit", "margin"], numeric_cols)
    elif any(w in q for w in ["quantity", "units", "items sold"]):
        metric = find_col(["quantity", "units", "qty"], numeric_cols)
    elif any(w in q for w in ["discount"]):
        metric = find_col(["discount"], numeric_cols)
    elif any(w in q for w in ["revenue", "sales", "turnover", "amount"]):
        metric = find_col(["sales", "revenue", "amount", "total"], numeric_cols)
    if metric is None:
        # For custom CSVs, recognize an explicitly mentioned numeric column name.
        metric = next((c for c in numeric_cols if str(c).strip().lower() in q), None)
    if metric is None:
        available = ", ".join(str(c) for c in numeric_cols) or "none"
        return normalize_plan({
            "title": "More information needed",
            "summary": "The request does not identify a measure that can be calculated from this dataset.",
            "metric": None, "aggregation": "sum", "dimension": None, "secondary_dimension": None,
            "date_column": date_col, "time_grain": "none", "chart_type": "bar", "x_metric": None,
            "filters": [], "sort": "none", "limit": 10, "cannot_answer": True,
            "reason": f"I could not map the requested measure to a numeric field. Available numeric fields: {available}.",
            "clarification_question": "Which available numeric measure should I analyse?",
            "planner_source": "Rule-based fallback",
        }, df)

    if any(w in q for w in ["average", "avg", "mean"]):
        aggregation = "mean"
    elif any(w in q for w in ["maximum", "highest value", "max "]):
        aggregation = "max"
    elif any(w in q for w in ["minimum", "lowest value", "min "]):
        aggregation = "min"
    elif any(w in q for w in ["number of orders", "order count", "how many orders", "count orders", "count of orders", "order volume", "orders by", "order volume by"]):
        aggregation = "nunique" if metric is not None else "count"
    else:
        aggregation = "sum"

    dimensions = [
        ("region", ["region", "territory", "market"]),
        ("product category", ["category", "product category", "category name"]),
        ("product", ["product name", "product", "item"]),
        ("segment", ["segment", "customer segment", "customer type"]),
        ("channel", ["channel", "sales channel"]),
        ("country", ["country"]),
        ("city", ["city"]),
        ("state", ["state", "province"]),
    ]
    detected_dimensions: list[Any] = []
    for label, aliases in dimensions:
        mentioned = (label in q) or any(a in q for a in aliases[1:]) or (label == "product category" and "category" in q)
        # “product category” is one dimension; do not also infer the separate Product field from the substring.
        if label == "product" and "product category" in q and "product name" not in q:
            mentioned = False
        candidate = find_col(aliases) if mentioned else None
        if candidate is not None and candidate not in detected_dimensions:
            detected_dimensions.append(candidate)
    dimension = detected_dimensions[0] if detected_dimensions else None
    secondary_dimension = detected_dimensions[1] if len(detected_dimensions) > 1 else None
    if dimension is None and any(w in q for w in [" by ", "breakdown", "split by", "across regions", "per region"]):
        # Use the first useful low-cardinality categorical field, excluding IDs.
        categorical = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) and c != date_col and df[c].nunique(dropna=True) <= 30 and not any(x in str(c).lower() for x in ["id", "name", "date"])]
        dimension = categorical[0] if categorical else None

    time_grain = "none"
    if date_col and any(w in q for w in ["monthly", "month by month", "by month", "each month", "over time", "trend", "time series"]):
        time_grain = "month"
    elif date_col and any(w in q for w in ["quarterly", "by quarter", "each quarter"]):
        time_grain = "quarter"
    elif date_col and any(w in q for w in ["yearly", "by year", "each year", "annual"]):
        time_grain = "year"
    elif date_col and "daily" in q:
        time_grain = "day"

    chart_type = "bar"
    if time_grain != "none":
        chart_type = "line"
    elif any(w in q for w in ["share", "proportion", "composition", "percentage of total", "distribution"]):
        chart_type = "pie" if dimension else "histogram"
    elif any(w in q for w in ["relationship", "correlation", "versus", " vs ", "against"]):
        chart_type = "scatter"
    elif any(w in q for w in ["distribution of", "histogram"]):
        chart_type = "histogram"
    elif any(w in q for w in ["trend", "over time"]):
        chart_type = "line" if date_col else "bar"

    if aggregation in {"sum", "mean", "min", "max"} and metric is None:
        return {
            "title": "More information needed", "summary": "No numeric measure is available for this calculation.",
            "metric": None, "aggregation": aggregation, "dimension": dimension,
            "secondary_dimension": secondary_dimension, "date_column": date_col, "time_grain": time_grain,
            "chart_type": chart_type, "x_metric": None, "filters": [], "sort": "none", "limit": 10,
            "cannot_answer": True, "reason": "The dataset has no suitable numeric column for this request.",
            "clarification_question": "Which available numeric column should be analysed?", "planner_source": "Rule-based fallback",
        }

    filters: list[dict[str, Any]] = []
    # Infer categorical equality filters only when the user signals filtering/context and a real value is named.
    filter_language = any(w in q for w in ["only ", "filter", "where ", "for ", "within ", "in the ", "in ", "from "])
    if filter_language:
        categories = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c]) and c != date_col and df[c].nunique(dropna=True) <= 50]
        candidates: list[tuple[int, Any, Any]] = []
        for c in categories:
            for val in df[c].dropna().astype(str).unique():
                if len(val.strip()) >= 2 and val.lower() in q:
                    candidates.append((len(val), c, val))
        # Keep explicit values across multiple columns, longest values first; dedupe by column.
        seen_cols: set[Any] = set()
        for _, c, val in sorted(candidates, reverse=True, key=lambda x: x[0]):
            if c not in seen_cols:
                filters.append({"column": c, "operator": "eq", "value": val, "values": [], "start_date": None, "end_date": None})
                seen_cols.add(c)

    # Quarter filtering for prompts such as "Q3 2024".
    quarter_match = re.search(r"\bq\s*([1-4])\s*(?:of\s*)?(20\d{2})\b", q)
    if quarter_match and date_col:
        quarter, year = int(quarter_match.group(1)), int(quarter_match.group(2))
        start_month = 1 + (quarter - 1) * 3
        start_date = pd.Timestamp(year=year, month=start_month, day=1)
        end_date = start_date + pd.offsets.MonthBegin(3) - pd.Timedelta(days=1)
        filters = [f for f in filters if f["column"] != date_col]
        filters.append({"column": date_col, "operator": "between", "value": None, "values": [], "start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")})
    elif date_col:
        year_match = re.search(r"\b(20\d{2})\b", q)
        if year_match:
            year = int(year_match.group(1))
            start_date = pd.Timestamp(year=year, month=1, day=1)
            end_date = pd.Timestamp(year=year, month=12, day=31)
            filters = [f for f in filters if f["column"] != date_col]
            filters.append({"column": date_col, "operator": "between", "value": None, "values": [], "start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")})

    # In follow-ups like “filter to East and show categories”, the filtered field should not dominate the chart.
    filtered_columns = {f["column"] for f in filters if f.get("operator") in {"eq", "contains", "in"}}
    if dimension in filtered_columns and secondary_dimension and secondary_dimension not in filtered_columns:
        dimension, secondary_dimension = secondary_dimension, None
    elif secondary_dimension in filtered_columns:
        secondary_dimension = None

    sort = "desc" if any(w in q for w in ["top", "best", "highest", "largest", "most", "best-performing", "top-performing"]) else ("asc" if any(w in q for w in ["bottom", "worst", "lowest", "smallest", "least"]) else "none")
    limit_match = re.search(r"\btop\s+(\d{1,2})\b", q)
    limit = max(1, min(100, int(limit_match.group(1)))) if limit_match else 10
    x_metric = None
    if chart_type == "scatter":
        x_metric = next((c for c in numeric_cols if c != metric), None)
        if x_metric is None:
            chart_type = "bar"

    return normalize_plan({
        "title": _title_from_prompt(prompt),
        "summary": "A data-backed view built from the columns available in this dataset.",
        "metric": metric, "aggregation": aggregation, "dimension": dimension,
        "secondary_dimension": secondary_dimension, "date_column": date_col, "time_grain": time_grain,
        "chart_type": chart_type, "x_metric": x_metric, "filters": filters,
        "sort": sort, "limit": limit, "cannot_answer": False, "reason": "",
        "clarification_question": "", "planner_source": "Rule-based fallback",
    }, df)


def _title_from_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip(" .?!")
    if not cleaned:
        return "Business intelligence dashboard"
    return cleaned[0].upper() + cleaned[1:]


def normalize_plan(plan: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Validate model output against the uploaded dataset before execution."""
    normalized = dict(plan)
    for key in ["metric", "dimension", "secondary_dimension", "date_column", "x_metric"]:
        normalized[key] = _column_matches(df, normalized.get(key))

    agg = str(normalized.get("aggregation", "sum")).lower()
    normalized["aggregation"] = agg if agg in AGGREGATIONS else "sum"
    #Normalize and validate multiple requested metrics
    raw_metrics = normalized.get("metrics", [])
    valid_metrics = []
    seen_metrics = set()
    if isinstance(raw_metrics, list):
        for item in raw_metrics:
            if not isinstance(item, dict):
                continue
            metric_name = _column_matches(df, item.get("metric"))
            aggregation_name = str(item.get("aggregation", "sum")).lower()
            if metric_name is None:
                continue
            if aggregation_name not in AGGREGATIONS:
                aggregation_name = "sum"
            if(
                aggregation_name in {"sum", "mean", "min", "max"} and not pd.api.types.is_numeric_dtype(df[metric_name])
            ):
                continue
            metric_key = (metric_name, aggregation_name)
            if metric_key not in seen_metrics:
                valid_metrics.append({
                    "metric": metric_name,
                    "aggregation": aggregation_name,
                })
                seen_metrics.add(metric_key)
    #Preserve the primary metric for backward compatibility
    primary_metric = normalized.get("metric")
    primary_aggregation = normalized["aggregation"]
    if primary_metric is not None:
        primary_key = (primary_metric, primary_aggregation)
        if primary_key not in seen_metrics:
            valid_metrics.insert(0,{
                "metric": primary_metric, 
                "aggregation": primary_aggregation,
            })
    normalized["metrics"] = valid_metrics
    chart = str(normalized.get("chart_type", "bar")).lower()
    normalized["chart_type"] = chart if chart in CHARTS else "bar"
    grain = str(normalized.get("time_grain", "none")).lower()
    normalized["time_grain"] = grain if grain in GRAINS else "none"
    if normalized["time_grain"] != "none" and normalized["date_column"] is None:
        normalized["date_column"] = _find_date_column(df)
    if normalized["date_column"] is None:
        normalized["time_grain"] = "none"
    sort = str(normalized.get("sort", "none")).lower()
    normalized["sort"] = sort if sort in {"asc", "desc", "none"} else "none"
    try:
        normalized["limit"] = max(1, min(100, int(normalized.get("limit", 10))))
    except (TypeError, ValueError):
        normalized["limit"] = 10
    normalized["filters"] = _normalize_filters(normalized.get("filters", []), df)
    normalized["title"] = str(normalized.get("title") or "Business intelligence dashboard")[:140]
    normalized["summary"] = str(normalized.get("summary") or "Analysis generated from the selected dataset.")[:300]
    normalized["cannot_answer"] = bool(normalized.get("cannot_answer", False))
    normalized["reason"] = str(normalized.get("reason") or "")[:500]
    normalized["clarification_question"] = str(normalized.get("clarification_question") or "")[:300]

    if normalized["metric"] is not None:
        if normalized["aggregation"] in {"sum", "mean", "min", "max"} and not pd.api.types.is_numeric_dtype(df[normalized["metric"]]):
            normalized["cannot_answer"] = True
            normalized["reason"] = f"'{normalized['metric']}' is not numeric, so {normalized['aggregation']} is not supported."
        if normalized["aggregation"] in {"count", "nunique"}:
            pass
    elif normalized["aggregation"] in {"sum", "mean", "min", "max"}:
        normalized["cannot_answer"] = True
        normalized["reason"] = normalized["reason"] or "The query planner did not identify a valid metric in the dataset."
        normalized["clarification_question"] = normalized["clarification_question"] or "Which available numeric measure should I analyse?"
    return normalized


def _normalize_filters(filters: Any, df: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(filters, list):
        return []
    result = []
    for f in filters[:20]:
        if not isinstance(f, dict):
            continue
        col = _column_matches(df, f.get("column"))
        op = str(f.get("operator", "eq")).lower()
        if not col or op not in OPERATORS:
            continue
        result.append({
            "column": col, "operator": op, "value": f.get("value"),
            "values": f.get("values") if isinstance(f.get("values"), list) else [],
            "start_date": f.get("start_date"), "end_date": f.get("end_date"),
        })
    return result


def plan_query(prompt: str, df: pd.DataFrame, api_key: str | None = None, model: str = "gemini-3.8-flash") -> tuple[dict[str, Any], str | None]:
    """Prefer Gemini; safely fall back on deterministic planning if not configured or unavailable."""
    if api_key and api_key.strip():
        try:
            return _gemini_plan(prompt, df, api_key.strip(), model), None
        except Exception as exc:  # graceful fallback for quota, network and parsing issues
            plan = fallback_plan(prompt, df)
            return plan, f"Gemini planner unavailable ({type(exc).__name__}): {exc}. Used the local rule-based planner for this request."
    return fallback_plan(prompt, df), "Gemini API key not configured. Used the local rule-based planner; add a key to enable LLM-based query interpretation."


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    result = df.copy()
    for f in filters:
        col = f["column"]
        if col not in result.columns:
            continue
        series = result[col]
        op = f["operator"]
        try:
            if op == "eq":
                value = f.get("value")
                if pd.api.types.is_numeric_dtype(series):
                    mask = pd.to_numeric(series, errors="coerce") == float(value)
                elif pd.api.types.is_datetime64_any_dtype(series) or "date" in str(col).lower():
                    sdates = pd.to_datetime(series, errors="coerce")
                    date_value = pd.to_datetime(value, errors="coerce")
                    mask = sdates.dt.normalize() == date_value.normalize()
                else:
                    mask = series.astype(str).str.casefold() == str(value).casefold()
                    if not mask.any():
                        mask = series.astype(str).str.casefold().str.contains(re.escape(str(value).casefold()), na=False)
            elif op == "contains":
                mask = series.astype(str).str.contains(re.escape(str(f.get("value", ""))), case=False, na=False)
            elif op in {"gt", "gte", "lt", "lte"}:
                if pd.api.types.is_numeric_dtype(series):
                    lhs = pd.to_numeric(series, errors="coerce")
                    rhs = float(f.get("value"))
                else:
                    lhs = pd.to_datetime(series, errors="coerce")
                    rhs = pd.to_datetime(f.get("value"), errors="coerce")
                mask = {"gt": lhs > rhs, "gte": lhs >= rhs, "lt": lhs < rhs, "lte": lhs <= rhs}[op]
            elif op == "in":
                vals = [str(v).casefold() for v in f.get("values", [])]
                mask = series.astype(str).str.casefold().isin(vals)
            elif op == "between":
                if pd.api.types.is_numeric_dtype(series):
                    lhs = pd.to_numeric(series, errors="coerce")
                    lo, hi = f.get("value"), f.get("values")
                    lo = float(lo) if lo is not None else float("-inf")
                    hi = float(hi[0]) if isinstance(hi, list) and hi else float("inf")
                else:
                    lhs = pd.to_datetime(series, errors="coerce")
                    lo = pd.to_datetime(f.get("start_date"), errors="coerce") if f.get("start_date") else lhs.min()
                    if f.get("end_date"):
                        hi = pd.to_datetime(f.get("end_date"), errors="coerce")
                        if pd.notna(hi):
                            hi = hi.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
                    else:
                        hi = lhs.max()
                mask = lhs.between(lo, hi, inclusive="both")
            else:
                continue
            result = result.loc[mask.fillna(False)]
        except (TypeError, ValueError, OverflowError):
            # Invalid model-generated filter never executes arbitrary code; ignore it gracefully.
            continue
    return result



def execute_plan(df: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    """Execute validated analytics with pandas, supporting multiple metrics."""
    filtered = apply_filters(df, plan.get("filters", []))
    metric = plan.get("metric")

    def failure(message, group_keys=None, time_key=None):
        return {
            "filtered": filtered,
            "result": pd.DataFrame(),
            "group_keys": group_keys or [],
            "time_key": time_key,
            "metric": metric,
            "error": message,
        }

    if plan.get("cannot_answer"):
        return failure(
            plan.get("reason")
            or "This question cannot be answered from the available columns."
        )

    if metric not in filtered.columns:
        return failure("The requested metric is not present in this dataset.")

    if filtered.empty:
        return failure(
            "No rows match the requested filters. Try a broader date range or different filter."
        )

    # Collect all requested metrics and retain the primary metric for compatibility.
    requested_metrics = plan.get("metrics", [])
    if not isinstance(requested_metrics, list):
        requested_metrics = []

    requested_metrics = [
        item.copy() for item in requested_metrics
        if isinstance(item, dict)
    ]

    if not any(item.get("metric") == metric for item in requested_metrics):
        requested_metrics.insert(0, {
            "metric": metric,
            "aggregation": plan.get("aggregation", "sum"),
        })

    valid_metrics = []
    seen = set()
    allowed_aggs = {"sum", "mean", "count", "nunique", "min", "max"}

    for item in requested_metrics:
        name = item.get("metric")
        agg = str(item.get("aggregation", "sum")).lower()

        if name not in filtered.columns or agg not in allowed_aggs:
            continue

        if agg in {"sum", "mean", "min", "max"}:
            if not pd.api.types.is_numeric_dtype(filtered[name]):
                continue

        key = (name, agg)
        if key not in seen:
            valid_metrics.append(key)
            seen.add(key)

    if not any(name == metric for name, _ in valid_metrics):
        return failure(
            f"The primary metric '{metric}' is invalid or cannot be aggregated."
        )

    working = filtered.copy()
    group_keys = []
    time_key = None
    date_col = plan.get("date_column")
    grain = plan.get("time_grain", "none")

    if grain != "none" and date_col in working.columns:
        dt = pd.to_datetime(working[date_col], errors="coerce")
        time_key = "_time_period"

        if grain == "day":
            working[time_key] = dt.dt.floor("D")
        elif grain == "week":
            working[time_key] = dt.dt.to_period("W").dt.start_time
        elif grain == "month":
            working[time_key] = dt.dt.to_period("M").dt.start_time
        elif grain == "quarter":
            working[time_key] = dt.dt.to_period("Q").dt.start_time
        elif grain == "year":
            working[time_key] = dt.dt.to_period("Y").dt.start_time

        working = working.loc[working[time_key].notna()]
        group_keys.append(time_key)

    dimension = plan.get("dimension")
    secondary = plan.get("secondary_dimension")

    if dimension in working.columns and dimension not in group_keys:
        group_keys.append(dimension)

    if secondary in working.columns and secondary not in group_keys:
        group_keys.append(secondary)

    if working.empty:
        return failure("No valid rows remain after processing the date grouping.",
                       group_keys, time_key)

    try:
        if group_keys:
            result = working[group_keys].drop_duplicates().copy()
            for name, agg in valid_metrics:
                values = working.groupby(
                    group_keys, dropna=False
                )[name].agg(agg).rename(name).reset_index()
                result = result.merge(values, on=group_keys, how="left")
        else:
            result = pd.DataFrame(index=[0])
            for name, agg in valid_metrics:
                result[name] = [working[name].agg(agg)]

        # Keep the original Value column for existing charts and insights.
        primary_agg = plan.get("aggregation", "sum")
        primary_key = (metric, primary_agg)

        if primary_key not in valid_metrics:
            primary_key = next(
                (key for key in valid_metrics if key[0] == metric),
                None
            )

        result["Value"] = result[primary_key[0]]

    except Exception as exc:
        return failure(
            f"Could not calculate the requested summary ({type(exc).__name__}).",
            group_keys, time_key
        )

    if time_key and time_key in result.columns:
        result = result.sort_values(time_key)
    elif plan.get("sort") in {"asc", "desc"}:
        result = result.sort_values(
            "Value", ascending=plan["sort"] == "asc"
        )

    limit = plan.get("limit", 100)
    if len(result) > limit and not time_key:
        result = result.head(limit)

    return {
        "filtered": filtered,
        "result": result.reset_index(drop=True),
        "group_keys": group_keys,
        "time_key": time_key,
        "metric": metric,
        "error": None,
    }



def build_insights(execution: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    result: pd.DataFrame = execution["result"]
    filtered: pd.DataFrame = execution["filtered"]
    if result.empty or "Value" not in result:
        return []
    insights: list[str] = []
    total = result["Value"].sum() if plan.get("aggregation") in {"sum", "count", "nunique"} else None
    metric_label = str(execution.get("metric") or "metric")
    if total is not None and pd.notna(total):
        if plan.get("aggregation") == "sum":
            insights.append(f"Total {metric_label}: {total:,.2f} across the displayed groups.")
        else:
            insights.append(f"Combined displayed value: {total:,.0f}.")
    # Roll up multi-dimensional/time-series results to surface the most important category.
    if plan.get("aggregation") in {"sum", "count", "nunique"}:
        for dimension_key in [plan.get("secondary_dimension"), plan.get("dimension")]:
            if dimension_key in result.columns and dimension_key:
                rollup = result.groupby(dimension_key, dropna=False)["Value"].sum().sort_values(ascending=False)
                if not rollup.empty:
                    insights.append(f"Leading {dimension_key}: {rollup.index[0]} ({rollup.iloc[0]:,.2f}).")
                    break
    keys = [k for k in execution.get("group_keys", []) if k in result.columns]
    if keys and not result.empty:
        valid = result.dropna(subset=["Value"])
        if not valid.empty:
            hi = valid.loc[valid["Value"].idxmax()]
            label = " · ".join(str(hi[k]) for k in keys)
            insights.append(f"Highest displayed group: {label} ({hi['Value']:,.2f}).")
            if len(valid) > 1:
                lo = valid.loc[valid["Value"].idxmin()]
                label_low = " · ".join(str(lo[k]) for k in keys)
                insights.append(f"Lowest displayed group: {label_low} ({lo['Value']:,.2f}).")
    insights.append(f"Analysis used {len(filtered):,} rows after applying filters.")
    return insights[:4]

def conversational_response(
    message: str,
    api_key: str,
    model: str,
    history: list[dict] | None = None,
    active_prompt: str = "",
    plan: dict | None = None,
    execution: dict | None = None,
) -> str:
    """Answer conversational questions using recent chat and dashboard context."""
    import json
    from google import genai

    if not api_key or not api_key.strip():
        return (
            "Conversational AI needs a configured Gemini API key. "
            "You can still generate dashboards using the local BI planner."
        )

    history = history or []

    # Include only a small amount of dashboard context, not the raw dataset.
    dashboard_context = {
        "current_question": active_prompt,
        "chart_title": (plan or {}).get("title", ""),
        "chart_summary": (plan or {}).get("summary", ""),
        "metric": (plan or {}).get("metric"),
        "dimension": (plan or {}).get("dimension"),
    }

    if execution:
        result = execution.get("result")
        if result is not None and not result.empty:
            dashboard_context["displayed_results"] = (
                result.head(15).to_dict(orient="records")
            )
        dashboard_context["execution_error"] = execution.get("error")

    recent_history = [
        {
            "role": item.get("role", "user"),
            "content": str(item.get("content", ""))[:1500],
        }
        for item in history[-8:]
    ]

    instruction = f"""
You are a friendly conversational business intelligence assistant.

Respond naturally to greetings and general conversation.
Explain the current dashboard clearly when asked.
Use the supplied dashboard results for factual claims.
Never invent figures or claim that you calculated something you did not.
If the available context does not contain the answer, say so.
If the user requests a new chart, a changed metric, a filter,
or another analysis, explain briefly that this request should be
handled by the dashboard analysis workflow.
Keep answers clear and concise.

CURRENT DASHBOARD CONTEXT:
{json.dumps(dashboard_context, default=str)}

RECENT CONVERSATION:
{json.dumps(recent_history, ensure_ascii=False)}

LATEST USER MESSAGE:
{message}
"""

    try:
        client = genai.Client(api_key=api_key.strip())
        response = client.models.generate_content(
            model=model,
            contents=instruction,
        )
        answer = getattr(response, "text", None)
        if answer and answer.strip():
            return answer.strip()
        return "I couldn't generate a response just now. Please try again."
    except Exception:
        return (
            "I'm unable to respond conversationally right now. "
            "Please try again in a moment; your existing dashboard "
            "analysis functionality remains available."
        )


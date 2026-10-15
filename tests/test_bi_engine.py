import pandas as pd

from bi_engine import apply_filters, execute_plan, fallback_plan, normalize_plan


def sample_df():
    return pd.DataFrame({
        "Order Date": pd.to_datetime(["2024-07-01", "2024-08-01", "2024-10-01", "2024-08-15"]),
        "Region": ["East", "West", "East", "East"],
        "Product Category": ["Furniture", "Technology", "Office Supplies", "Furniture"],
        "Sales": [100.0, 200.0, 300.0, 50.0],
        "Profit": [10.0, 25.0, 30.0, -2.0],
        "Order ID": ["A", "B", "C", "D"],
    })


def test_fallback_selects_monthly_sales_by_region():
    df = sample_df()
    plan = fallback_plan("Show monthly sales by region", df)
    assert plan["metric"] == "Sales"
    assert plan["dimension"] == "Region"
    assert plan["time_grain"] == "month"
    assert plan["chart_type"] == "line"


def test_quarter_filter_works():
    df = sample_df()
    plan = fallback_plan("Show sales for Q3 2024 by region", df)
    execution = execute_plan(df, plan)
    assert execution["error"] is None
    assert len(execution["filtered"]) == 3
    assert execution["filtered"]["Sales"].sum() == 350.0


def test_filter_case_insensitive():
    df = sample_df()
    filtered = apply_filters(df, [{"column": "Region", "operator": "eq", "value": "east", "values": [], "start_date": None, "end_date": None}])
    assert len(filtered) == 3


def test_invalid_metric_is_not_accepted_as_column():
    df = sample_df()
    plan = normalize_plan({"metric": "Customer Happiness", "aggregation": "sum", "cannot_answer": False}, df)
    assert plan["metric"] != "Customer Happiness"


def test_multidimensional_quarter_query():
    df = sample_df()
    plan = fallback_plan("Show monthly sales for Q3 2024 broken down by region and product category", df)
    execution = execute_plan(df, plan)
    assert plan["secondary_dimension"] == "Product Category"
    assert plan["time_grain"] == "month"
    assert execution["error"] is None
    assert execution["filtered"]["Sales"].sum() == 350.0


def test_year_filter_and_order_volume():
    df = sample_df()
    plan = fallback_plan("Show order volume by region for 2024", df)
    execution = execute_plan(df, plan)
    assert plan["metric"] == "Order ID"
    assert plan["aggregation"] == "nunique"
    assert len(execution["filtered"]) == 4


def test_product_category_does_not_also_group_by_product():
    df = sample_df()
    plan = fallback_plan("Compare sales by product category", df)
    assert plan["dimension"] == "Product Category"
    assert plan["secondary_dimension"] is None


def test_followup_filter_does_not_use_filtered_region_as_primary_group():
    df = sample_df()
    prompt = "Original request: Compare total profit by product category for 2024. Follow-up request: Now filter this to the East region and show the top 5 categories."
    plan = fallback_plan(prompt, df)
    assert plan["metric"] == "Profit"
    assert plan["dimension"] == "Product Category"
    assert plan["filters"][0]["column"] == "Region"


def test_unknown_measure_fails_closed_instead_of_using_first_numeric_field():
    df = sample_df()
    plan = fallback_plan("Show customer happiness by region", df)
    assert plan["cannot_answer"] is True
    assert plan["metric"] is None
    assert "Available numeric fields" in plan["reason"]


def test_order_volume_uses_distinct_order_id_not_quantity():
    df = sample_df()
    plan = fallback_plan("Show order volume by region", df)
    execution = execute_plan(df, plan)
    assert plan["metric"] == "Order ID"
    assert plan["aggregation"] == "nunique"
    assert execution["error"] is None


def test_between_date_filter_includes_entire_end_day():
    df = pd.DataFrame({
        "Order Date": pd.to_datetime(["2024-09-30 08:00:00", "2024-09-30 22:00:00", "2024-10-01 00:00:00"]),
        "Sales": [10, 20, 30],
    })
    from bi_engine import apply_filters
    filtered = apply_filters(df, [{
        "column": "Order Date", "operator": "between", "value": None, "values": [],
        "start_date": "2024-09-01", "end_date": "2024-09-30"
    }])
    assert filtered["Sales"].tolist() == [10, 20]

# 10-minute presentation script

## 0:00–1:00 — Problem and persona
“Executives often know what they need to ask, but not how to write SQL or configure a BI dashboard. Basic reporting can become a bottleneck. This project lets them ask in plain English and turns the request into a validated data analysis and interactive dashboard.”

## 1:00–2:00 — Architecture
Show this pipeline: question → Gemini structured JSON plan → schema validation → pandas computation → Plotly chart/KPIs/insights. Emphasize that the LLM does not fabricate metric values and the app does not execute generated code.

## 2:00–3:30 — Query 1: simple comparison
Ask: “Compare total sales by product category and show the top 3 categories.”
Point out the KPI, categories sorted by the requested measure, interactive tooltips, computed key takeaways and downloadable result table.

## 3:30–5:30 — Query 2: time filter and breakdown
Ask: “Show monthly sales for Q3 2024 broken down by region and product category.”
Open “See how your question was interpreted.” Show the selected metric, date column, quarter filter, monthly grain, breakdown fields and line chart. Explain that the plan must use real columns in the uploaded data.

## 5:30–7:00 — Follow-up question
Ask: “Compare total profit by product category for 2024.” Then follow up: “Now filter this to the East region and show the top 5 categories.”
Show that the dashboard updates without the user manually configuring a chart.

## 7:00–8:00 — Upload a CSV
Upload a small CSV with clear headers and at least one numeric field. Point out that the schema snapshot updates and that no hardcoded column mapping is needed for the Gemini planner.

## 8:00–9:00 — Error handling and trust
Ask a question about an absent field, such as customer happiness, and explain that the correct behavior is to say that the dataset cannot answer it—not to invent a result. Show the warning/fallback behavior if the API key is missing or unavailable.

## 9:00–10:00 — Impact and next steps
Summarize: “The prototype reduces the gap between business questions and usable visuals, supports uploaded CSVs, and keeps data computation in a validated execution layer. The next production step is a read-only SQL connector, access control, query evaluation benchmarks and audit logs.”

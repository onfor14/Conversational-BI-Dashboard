# Conversational AI for Instant Business Intelligence Dashboards

A portfolio-ready prototype that turns plain-English business questions into interactive, data-backed dashboards. Built with Streamlit, pandas, Plotly, and Google's Gemini API.

## What it does

- **Natural-language analytics:** ask questions such as “Show monthly sales by region for Q3 2024.”
- **Schema-aware planning:** Gemini returns a constrained JSON analysis plan (metric, aggregation, dimensions, time grain, chart type, filters). The app validates every column and filter against the active CSV before computing anything.
- **Real data calculations:** pandas calculates totals, averages, counts, groupings and filters. The app does not execute model-generated Python or SQL.
- **Interactive charts:** Plotly line, bar, pie, area, scatter and histogram charts, including tooltips and zoom.
- **Upload your own CSV:** change datasets without editing source code.
- **Follow-up chat:** refine the current dashboard in natural language.
- **Graceful failure:** when the API key is absent, the API is rate-limited, or the API response is invalid, a rule-based local planner keeps common demo questions working. Unsupported questions are surfaced rather than answered with invented business metrics.
- **Transparency:** inspect the plan used to create the dashboard, view the computed result table and download the aggregate results.

## Architecture

```text
User's natural-language question
              │
              ▼
  Gemini structured-output planner
  (local rule-based fallback if needed)
              │
              ▼
  Plan validation against CSV schema
              │
              ▼
  pandas filters + group-by + aggregation
              │
              ▼
  Plotly chart + KPI cards + computed insights
              │
              └── Follow-up request loops back to planner
```

The application uses a constrained analysis plan rather than asking the LLM to produce executable SQL or Python. This keeps execution deterministic and prevents prompt-generated code from being run. For a production system, the execution layer could be backed by read-only SQL with parameterized queries and explicit authorization controls.

## Project structure

```text
conversational_bi_dashboard/
├── app.py                     # Streamlit UI and dashboard flow
├── bi_engine.py               # Gemini planner, validation, filters, calculations, insights
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── sample_data/
│   └── sales_data.csv         # Synthetic demo data (1,600 rows)
└── tests/
    └── test_bi_engine.py      # Basic planner/filter tests
```

## Run locally (Windows)

### 1. Install Python

Use Python 3.11 or 3.12 for a straightforward dependency setup. During installation on Windows, enable **Add Python to PATH**.

### 2. Open the project directory

Extract the ZIP, open the `conversational_bi_dashboard` folder in VS Code, then select **Terminal → New Terminal**.

### 3. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your machine has Python 3.12 instead, replace `py -3.11` with `py -3.12`.

### 4. Start the app

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser. The sample sales dataset works without an API key.

### 5. Enable Gemini

1. Create or open a Google AI Studio account at <https://aistudio.google.com/app/apikey>.
2. Generate an API key.
3. Paste it into **Gemini API key** in the app sidebar. Alternatively, copy `.env.example` to `.env` and add the key to `GEMINI_API_KEY`.
4. Keep `.env` private; never upload the key to GitHub.

When Gemini is enabled, the app sends the user's question plus a schema/profile (column names, types, numeric ranges and some categorical values) to Gemini. It does not intentionally send raw CSV rows. Do not use confidential or regulated data without reviewing your organization's policies and the API provider's terms.

## Demo questions for the presentation

Use these three in order; the questions progress from a simple comparison to time-series and follow-up analysis.

1. **Basic — category comparison**  
   `Compare total sales by product category and show the top 3 categories.`

2. **Intermediate — time + multiple dimensions + date filter**  
   `Show monthly sales for Q3 2024 broken down by region and product category.`

3. **Advanced — conversational change**  
   First ask: `Compare total profit by product category for 2024.`  
   Then in the chat box ask: `Now filter this to the East region and show the top 5 categories.`

Also show a CSV upload and the **See how your question was interpreted** expander. If the Gemini API quota is unavailable during the presentation, switch to the local fallback and use supported prompts.

## Run tests

```powershell
python -m pytest -q
```

## Publish to a public GitHub repository

1. Create a new **public** repository on <https://github.com/new>, for example `conversational-business-intelligence`.
2. From the project directory, run:

```powershell
git init
git add .
git commit -m "Build conversational BI dashboard prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/conversational-business-intelligence.git
git push -u origin main
```

3. Open the repository page and confirm that `app.py`, `bi_engine.py`, `README.md`, tests and sample data are present. Confirm that `.env` and any API key are **not** in the repository.

If Git asks you to authenticate, use GitHub's browser sign-in or the credential manager; do not put a password or token in the remote URL.

## Deploy as a public demo (optional)

1. Push the repository to GitHub.
2. Open <https://share.streamlit.io/> and connect the repository.
3. Choose `app.py` as the entrypoint.
4. In the deployment's app settings, add `GEMINI_API_KEY` and `GEMINI_MODEL` as secrets. Do not commit secrets to the repository.
5. Open the deployed app in a private/incognito window and test the sample questions, the uploader and fallback mode.

## Talking points for an interview

**Problem:** Business users know their questions but often need analysts to write SQL and assemble visualizations. That slows down decision-making and creates repetitive reporting work.

**Solution:** This prototype separates language understanding from deterministic data execution. An LLM converts the request into a constrained analytics plan; validation checks every field; pandas runs the calculation; Plotly presents the result.

**Why not let the LLM write and execute SQL/Python directly?** Constrained planning is easier to validate and safer to operate. The model selects allowed operations, while the application owns the actual computation.

**How do you handle hallucinations?** The model receives a schema profile rather than arbitrary instructions to invent metrics. Its plan is validated against real columns and supported aggregations. Empty results and missing metrics are reported to the user; values displayed in KPIs and insights are computed from the dataframe.

**What would you do next?** Add a read-only SQL connector with parameterized queries, role-based access control, better schema/business glossary metadata, query-plan evaluation tests, cached queries, audit logs, and chart-level filters.

## Known prototype limitations

- The local fallback intentionally covers common sales, profit, quantity, time, category and region questions; broad, ambiguous questions work best with Gemini enabled.
- Multi-turn follow-ups are interpreted in the context of the original question; the demo is not a persistent conversation memory system.
- The prototype works on a dataframe loaded into app memory. For larger datasets or multiple users, introduce a database/query service and stronger access controls.
- Generated insights are descriptive summaries, not causal explanations or forecasts.

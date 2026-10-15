# Conversational BI Dashboard

An AI-powered Business Intelligence dashboard that allows users to explore sales data through natural-language queries and interactive visualizations.

## Features

- Natural-language interaction for business data analysis
- Sales and profit analysis
- Interactive charts and visualizations
- Data processing and analysis using Pandas
- AI-powered query interpretation using Google Gemini
- Interactive dashboard built with Streamlit and Plotly

## Tech Stack

- **Python** — core programming language
- **Streamlit** — dashboard and user interface
- **Pandas** — data processing and analysis
- **Plotly** — interactive data visualizations
- **Google Gemini API** — natural-language query interpretation

## Project Structure

```text
conversational-bi-dashboard/
├── app.py
├── bi_engine.py
├── requirements.txt
├── README.md
├── DEMO_SCRIPT.md
├── .env.example
├── sample_data/
└── tests/
```

## Getting Started

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Configure your Gemini API key using an environment variable or the appropriate secrets configuration.
5. Run the application using `streamlit run app.py`.

## Security

Never commit your real API key or secret configuration files. Use environment variables locally and deployment secrets in hosted environments.

## Purpose

This project demonstrates how conversational AI, data analytics, and interactive visualization can be combined to make business intelligence more accessible.
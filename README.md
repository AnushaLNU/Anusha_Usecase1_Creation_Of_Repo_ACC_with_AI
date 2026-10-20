# Agentic Medallion Pipeline

An interactive Streamlit application that profiles uploaded CSV files and guides them through Bronze, Silver, and Gold data layers. The app generates transformation rules for each layer, lets you review and approve them, and produces a final report.

## Requirements

- Python 3.10 or newer
- An Anthropic API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` to your key. Never commit `.env`.
4. Start the application:

   ```powershell
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit, upload one or more CSV files, and enter the business question to answer.

The pipeline stores profiles, transformation specifications, intermediate data, audit logs, and reports under `data/`. These generated and uploaded files are excluded from Git.

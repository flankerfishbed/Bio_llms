# Biological Paper Critic – Multi-Model LLM Peer Reviewer

This project is a Streamlit web app that helps you critically review biology/biotech research papers
using multiple LLM roles (summariser, methods reviewer, statistics checker, next-experiments designer)
plus a final judge model, all via OpenRouter.

> This tool is meant as an assistive aid and **does not replace real expert peer review**.

## Features

- Upload a PDF paper or paste text directly.
- Run multiple specialised LLM roles on the paper:
  - Summariser
  - Methods Reviewer
  - Statistics Checker
  - Next Experiments Designer
- Combine expert outputs with a Judge model for a final, structured review and verdict:
  - Accept / Minor revision / Major revision / Reject
- View all role outputs in a clean Streamlit UI.
- Download the full review as a Markdown report.

## Requirements

- Python 3.9+
- An OpenRouter API key (`OPENROUTER_API_KEY`)

## Installation

1. Clone or copy this project into a new directory.
2. (Optional but recommended) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your OpenRouter API key in the environment, for example on Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "your_api_key_here"
```

On macOS/Linux bash:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

## Running the App

From the project directory, run:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

## Usage Overview

1. In the sidebar, confirm that `OPENROUTER_API_KEY` is detected.
2. (Optional) Toggle:
   - Skip statistics review
   - Short run (more aggressive truncation of long papers)
3. Upload a PDF **or** paste the paper text/abstract into the text area.
   - If both are provided, the PDF is preferred.
4. Click **Run Review**.
5. After processing, inspect:
   - Verdict and brief reason
   - Paper summary
   - Methods review
   - Statistics review (or note if skipped)
   - Suggested next experiments
   - Judge's full review
6. Use the **Download review as Markdown** button to save the full report.

## Notes

- The app truncates long papers to stay within typical LLM context limits. A note is added when truncation occurs.
- Model names for each role are defined centrally in `llm_client.py` and can be adjusted easily.
- The code is structured to keep the Streamlit UI (`app.py`) thin, delegating logic to:
  - `llm_client.py` – OpenRouter client and model registry
  - `council_roles.py` – role definitions and prompts
  - `paper_processing.py` – PDF/text extraction and cleaning
  - `orchestrator.py` – multi-model orchestration and report assembly



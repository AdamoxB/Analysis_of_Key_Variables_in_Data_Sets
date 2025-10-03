# PyCaret Auto‑Model Trainer

A Streamlit application that lets users upload a dataset, sample it, choose target and ignore columns, automatically detects whether the problem is classification or regression, runs PyCaret experiments, compares models, and visualises key metrics.

## Features

- Upload CSV / JSON / Excel files.
- Sample data by percentage for quick preview.
- Automatic problem type detection (classification if any column has `object` dtype).
- Setup & run PyCaret classification experiments.
- Compare top 5 or all models.
- Visualise feature importance, confusion matrix and ROC curve.
- Save plots as PNG files in the `plots/` directory.

## Installation

```bash
# Clone repo
git clone <repo-url>
cd <project-root>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Environment Variables
No environment variables are required for this project.

The file .env.example is included as a placeholder.

Run the App
streamlit run app.py
Open the displayed URL (usually http://localhost:8501) in your browser.

Testing
Unit‑test scaffolding is available under tests/.

Add tests and run:

python -m unittest discover tests

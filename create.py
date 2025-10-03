#!/usr/bin/env python3

import os

# -----------------------------------------------
# Define the files and their contents.
# -----------------------------------------------
file_contents = {
  "app.py": '''\
# app.py
"""
Streamlit application for automated model training with PyCaret.
"""

import os
import shutil
import time
from pathlib import Path
import streamlit as st
import pandas as pd

# Local imports
from utils.data_loader import load_data
from utils.plot_utils import toss
from models.classification_experiment import ClassificationRunner

# --------------------------------------------------------------------------- #
# Configuration (taken from the JSON)
# --------------------------------------------------------------------------- #

st.set_page_config(layout="wide")

PYCARET_SETUP_PARAMS = {
    "session_id": 123,
    "ignore_features": None,          # will be overridden by user selection
    "fix_imbalance": True,
    "normalize": True,
    "transformation": True,
    "verbose": False,
}

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def delete_old_plots():
    """Remove all PNG files in the plots directory."""
    for file_path in PLOTS_DIR.glob("*.png"):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"File not found (already deleted): {file_path}")
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")

def save_plot(fig, filename: str):
    """Save a matplotlib figure to the plots directory."""
    path = PLOTS_DIR / f"{filename}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path

# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.title("PyCaret Auto‑Model Trainer")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV/JSON/Excel file",
    type=["csv", "json", "xlsx", "xls"],
)

if uploaded_file:
    # 1. Load data
    sep = st.sidebar.text_input("CSV separator (default ',')", value=",")
    df = load_data(uploaded_file, sep=sep)
    if df is None:
        st.error("Unsupported file format.")
    else:
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

        # 2. Sample data
        sample_pct = st.sidebar.slider(
            "Sample % of data for quick preview",
            min_value=10,
            max_value=100,
            value=20,
            step=5,
        )
        min_df = df.sample(frac=sample_pct / 100, random_state=42)
        st.subheader("Random sample")
        st.write(min_df.head())

        # 3. Target & ignore columns
        target_col = st.sidebar.selectbox(
            "Target column", options=df.columns.tolist()
        )
        cols_to_ignore = st.sidebar.multiselect(
            "Columns to ignore",
            options=[c for c in df.columns if c != target_col],
            default=[],
        )

        # 4. Detect problem type
        if any(df[col].dtype == object for col in df.columns):
            problem_type = "classification"
        else:
            problem_type = "regression"

        st.sidebar.markdown(f"**Detected problem type:** {problem_type}")

        # ------------------------------------------------------------------- #
        # Tabs
        # ------------------------------------------------------------------- #

        tab1, tab2, tab3 = st.tabs(
            ["Random Rows", "Missing Data Analysis", "Setup & Run"]
        )

        with tab1:
            st.write("Sample of the dataset:")
            st.dataframe(min_df.head(10))

        with tab2:
            st.subheader("Missing data")
            missing_info = df.isnull().mean() * 100
            st.table(missing_info.round(2).to_frame(name="Missing %"))

        # ------------------------------------------------------------------- #
        # Setup & Run Tab
        # ------------------------------------------------------------------- #

        with tab3:
            st.subheader(f"PyCaret {problem_type.capitalize()} Setup")

            if problem_type == "classification":
                # Store ignore features in the params dict
                PYCARET_SETUP_PARAMS["ignore_features"] = cols_to_ignore

                # Setup experiment
                st.info("Setting up PyCaret experiment…")
                runner = ClassificationRunner(
                    df=df,
                    target=target_col,
                    ignore_features=cols_to_ignore,
                    config=PYCARET_SETUP_PARAMS,
                )
                runner.setup()
                st.success("Setup complete!")

                # Model comparison
                compare_range = st.radio(
                    "Select model comparison range",
                    options=["Top 5", "All"],
                    index=0,
                )

                if st.button("Run Time series models:"):
                    st.info("Running model comparison…")
                    best_model = runner.compare(range_top=compare_range)
                    st.session_state["best_model"] = best_model
                    st.success(f"Best model: {best_model}")

                    # Pull metrics table
                    from pycaret.classification import pull

                    metrics_df = pull()
                    st.subheader("Model Metrics")
                    st.dataframe(metrics_df)

                    # Delete old plots
                    delete_old_plots()

                    # Generate and save plots
                    plot_types = ["feature_selection", "confusion_matrix", "auc"]
                    for ptype in plot_types:
                        fig = runner.plot(p_type=ptype)
                        path = save_plot(fig, f"{best_model}_{ptype}")
                        st.image(path)

            else:  # regression (not fully implemented – placeholder)
                st.warning(
                    "Regression workflow is not yet implemented. "
                    "Please upload a classification dataset."
                )
else:
    st.info("Upload a file to begin.")
''',
  "utils/__init__.py": '''\
# utils/__init__.py
''' ,
  "utils/data_loader.py": '''\
# utils/data_loader.py
"""
Utility functions for loading data files.
"""

import pandas as pd
from typing import Optional


def load_data(uploaded_file, sep: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV, JSON, or Excel file into a DataFrame.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object.
    sep : str
        Separator for CSV files.

    Returns
    -------
    pd.DataFrame | None
        The loaded DataFrame, or None if the format is unsupported.
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=sep)
        elif name.endswith((".json", ".jsn")):
            df = pd.read_json(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None
    except Exception as e:  # pragma: no cover
        print(f"Error loading file {uploaded_file.name}: {e}")
        return None

    return df
''',
  "utils/plot_utils.py": '''\
# utils/plot_utils.py
"""
Utility for generating and caching PyCaret plots.
"""

import io
from typing import Any

import streamlit as st


def toss(model: Any, plot_type: str, session_key: str):
    """
    Generate a PyCaret plot, store it in Streamlit's session state,
    and display the image.

    Parameters
    ----------
    model : Any
        The trained model or experiment object.
    plot_type : str
        Type of plot to generate (e.g., 'feature_selection', 'confusion_matrix').
    session_key : str
        Key under which the plot buffer will be stored in st.session_state.
    """
    from pycaret.classification import plot_model

    fig = plot_model(model, plot=plot_type, verbose=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    # Store buffer for later use (e.g., downloading)
    st.session_state[session_key] = buf
''',
  "models/__init__.py": '''\
# models/__init__.py
''' ,
  "models/classification_experiment.py": '''\
# models/classification_experiment.py
"""
Thin wrapper around PyCaret's ClassificationExperiment.
"""

from pycaret.classification import (
    setup as cl_setup,
    compare_models as cl_compare_models,
    plot_model as cl_plot_model,
)

class ClassificationRunner:
    """
    Encapsulates a PyCaret classification experiment.
    """

    def __init__(self, df, target, ignore_features, config):
        self.df = df
        self.target = target
        self.ignore_features = ignore_features
        self.config = config
        self.experiment = None

    def setup(self):
        """
        Initialise the PyCaret experiment with the provided configuration.
        """
        self.experiment = cl_setup(
            data=self.df,
            target=self.target,
            ignore_features=self.ignore_features,
            session_id=self.config.get("session_id", 123),
            fix_imbalance=self.config.get("fix_imbalance", True),
            normalize=self.config.get("normalize", True),
            transformation=self.config.get("transformation", True),
            verbose=self.config.get("verbose", False),
        )

    def compare(self, range_top="Top 5"):
        """
        Run model comparison and return the best model.
        """
        if not self.experiment:
            raise RuntimeError("Experiment has not been set up.")
        # The default behaviour of cl_compare_models uses the global
        # experiment created by setup().
        best = cl_compare_models(sort="Accuracy", n_select=1)
        return best[0]  # first model in list

    def plot(self, p_type):
        """
        Generate a plot for the current experiment.
        """
        if not self.experiment:
            raise RuntimeError("Experiment has not been set up.")
        fig = cl_plot_model(p_type=p_type, verbose=False)
        return fig
''',
  "requirements.txt": '''\
streamlit>=1.10
pandas>=2.0
numpy>=1.24
pycaret>=3.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
''',
  "README.md": '''\
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
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

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
''',
".env.example": '''\

No environment variables needed for this demo.
''',
"tests/init.py": '''\

tests/init.py
'''
}


# -----------------------------------------------
# Create directories (if needed) and write files.
# -----------------------------------------------
for path, content in file_contents.items():
    dir_path = os.path.dirname(path)
    if dir_path:                     # skip makedirs for top‑level files
        os.makedirs(dir_path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("Files created:", ", ".join(file_contents.keys()))

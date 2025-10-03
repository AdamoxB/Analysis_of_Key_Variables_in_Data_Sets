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

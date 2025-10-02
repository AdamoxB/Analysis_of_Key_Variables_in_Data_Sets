# --------------------------------------------------------------
#  Data Feature Analysis – Streamlit + PyCaret
# --------------------------------------------------------------

import os
import io
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.datasets import get_data

# ------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Data Feature Analysis",
    layout="wide",
    page_icon="📊",
)

# ------------------- COLORS & STYLES ---------------------------------
BLUE   = "#007BFF"
GREEN  = "#28A745"
RED   = "#DC3545"

st.markdown(f"""
<style>
h1, h2, h3 {{ color: {BLUE}; }}
p {{ font-size: 1.1rem; }}
.metric-value {{ color: {GREEN}; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
#  CACHING UTILITIES
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_file(file, sep):
    """Load csv/json/xlsx and return a DataFrame."""
    if file is None:
        return None

    ext = Path(file.name).suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file, sep=sep)
        elif ext in {".json"}:
            df = pd.read_json(file)
        else:  # .xls/.xlsx
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"❌ Błąd przy wczytywaniu pliku: {e}")
        return None


@st.experimental_singleton(show_spinner=False)
def get_experiment(df, target, ignore_features, problem_type):
    """Return a ready‑to‑use PyCaret experiment."""
    if problem_type == "Klasyfikacja":
        exp = ClassificationExperiment()
    else:
        exp = RegressionExperiment()

    exp.setup(
        data=df,
        target=target,
        ignore_features=ignore_features,
        session_id=123,
        normalize=True,
        transformation=True,
        fix_imbalance=True if problem_type == "Klasyfikacja" else False,
        verbose=False,
    )
    return exp


# ------------------------------------------------------------------
#  SIDEBAR – INPUTS
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Opcje")

uploaded_file = st.sidebar.file_uploader(
    "1️⃣ Wybierz plik", type=["csv", "json", "xls", "xlsx"]
)

sep_options = [",", ";", "\t", " ", "custom"]
tab_sep = st.sidebar.selectbox("Separator:", sep_options, index=0)
if tab_sep == "custom":
    separator = st.sidebar.text_input("Podaj własny separator", value=",")
else:
    separator = tab_sep

percent_sample = st.sidebar.number_input(
    "📊 Procent danych do analizy",
    min_value=1,
    max_value=100,
    value=50,
)

# ------------------------------------------------------------------
#  MAIN LOGIC
# ------------------------------------------------------------------
if uploaded_file is None:
    st.info("📂 Wybierz plik, aby rozpocząć.")
else:
    # ---- Load data -------------------------------------------------
    df = load_file(uploaded_file, separator)
    if df is None or df.empty:
        st.warning("Brak danych w pliku.")
    else:
        # ---- Sample data --------------------------------------------
        sample_frac = percent_sample / 100
        sample_df = df.sample(frac=sample_frac, random_state=123)

        # ---- Decide problem type ------------------------------------
        def detect_problem_type(df):
            return "Klasyfikacja" if df.select_dtypes(include="object").shape[1] > 0 else "Regresja"

        problem_type = detect_problem_type(sample_df)
        st.success(f"📌 **Typ problemu:** {problem_type}")

        # ---- UI – tabs -----------------------------------------------
        tab_overview, tab_missing, tab_experiment, tab_results = st.tabs(
            ["Przegląd danych", "Brakujące dane", f"Setup ({problem_type})", "Wyniki"]
        )

        with tab_overview:
            st.header("Random sample")
            st.dataframe(sample_df.sample(10))

        with tab_missing:
            st.subheader("Procent braków (w %)")
            missing_pct = sample_df.isna().mean() * 100
            st.bar_chart(missing_pct)

            st.subheader("Typy danych i unikatowe wartości")
            buffer = io.StringIO()
            sample_df.info(buf=buffer)
            st.text(buffer.getvalue())
            st.write(sample_df.describe(include="all").T.round(2))

        with tab_experiment:
            # ---- Column to model ---------------------------------------
            target_col = st.selectbox("Wybierz kolumnę docelową", sample_df.columns)

            ignore_cols = st.multiselect(
                "Ignorowane kolumny",
                options=[c for c in sample_df.columns if c != target_col],
                default=[],
            )

            # ---- Run experiment ----------------------------------------
            exp = get_experiment(sample_df, target_col, ignore_cols, problem_type)
            st.session_state["experiment"] = exp  # keep reference

            # Choose model set
            model_set = st.radio(
                "Zakres porównania modeli",
                options=["Szybki", "Kompleksowy"],
                index=0,
            )

            if model_set == "Szybki":
                compare_kwargs = {"include": ["rf", "lr", "gbc", "knn"], "fold": 5, "verbose": False}
            else:
                compare_kwargs = {}

            if st.button("💻 Uruchom modele"):
                with st.spinner("Trening modeli… (może zająć kilka minut)"):
                    best_model = exp.compare_models(**compare_kwargs)
                st.session_state["best_model"] = best_model
                metrics_df = exp.pull()
                st.session_state["metrics"] = metrics_df

        with tab_results:
            # ---- Show metrics --------------------------------------------------
            if "metrics" in st.session_state:
                st.subheader("Metryki")
                st.dataframe(st.session_state.metrics)

            # ---- Plotting ----------------------------------------------------
            if "best_model" in st.session_state:
                best_model = st.session_state["best_model"]

                st.subheader("Feature Importance")
                exp.plot_model(best_model, plot="feature", display_format="streamlit")

                st.subheader("Confusion Matrix (classification)")
                if problem_type == "Klasyfikacja":
                    exp.plot_model(best_model, plot="confusion_matrix", display_format="streamlit")

                st.subheader("ROC Curve")
                if problem_type == "Klasyfikacja":
                    exp.plot_model(best_model, plot="auc", display_format="streamlit")

            # ---- Save / Load model ------------------------------------------
            if st.button("💾 Zapisz najlepszy model"):
                exp.save_model(st.session_state["best_model"], f"{target_col}_best")
                st.success(f"Model zapisany jako `{target_col}_best.pkl`")

            uploaded_model = st.file_uploader(
                "📤 Załaduj zapisany model", type="pkl"
            )
            if uploaded_model:
                loaded_model = exp.load_model(uploaded_model)
                st.session_state["loaded_model"] = loaded_model
                st.success("Model załadowany!")

            # ---- Download metrics -------------------------------------------
            if "metrics" in st.session_state:
                csv_bytes = io.BytesIO()
                st.session_state.metrics.to_csv(csv_bytes, index=False)
                csv_bytes.seek(0)

                st.download_button(
                    label="📥 Pobierz metryki jako CSV",
                    data=csv_bytes,
                    file_name=f"{target_col}_metrics.csv",
                    mime="text/csv",
                )

# ------------------------------------------------------------------
#  CLEAN‑UP (optional) – delete temp files if any
# ------------------------------------------------------------------
temp_files = ["Feature Importance.png", "Confusion Matrix.png", "ROC.png"]
for f in temp_files:
    try:
        os.remove(f)
    except FileNotFoundError:
        pass


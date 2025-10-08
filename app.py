import streamlit as st
import pandas as pd
import numpy as np
from pycaret.datasets import get_data
from pycaret.classification import setup as cl_setup, compare_models as cl_compare_models, create_model as cl_create_model, plot_model as cl_plot_model, finalize_model as cl_finalize_model, save_model as cl_save_model, load_model as cl_load_model, predict_model as cl_predict_model, pull, ClassificationExperiment
from pycaret.regression import setup as re_setup, compare_models as re_compare_models, create_model as re_create_model, plot_model as re_plot_model, finalize_model as re_finalize_model, save_model as re_save_model, load_model as re_load_model, predict_model as re_predict_model, pull, RegressionExperiment
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import io
import shutil
import random  # Importowanie modułu 'random'

st.set_page_config(layout='wide')

# Dane testowe regresji
data = []
for _ in range(1000):
    x0 = random.randint(100, 200)
    x1 = random.randint(0, 50)
    noise = random.gauss(0, 10)
    y = 2 * x0 + 0.9 * x1 + noise
    data.append({
        'x0': x0,
        'x1': x1,
        'x2': random.randint(0, 10),
        'y': y,
    })

df_test = pd.DataFrame(data)

#================================
def plot_scatter(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='x0', y='y')
    plt.title('Scatter Plot of x0 vs y')
    st.pyplot(plt)

# def display_setup_info(setup):
#     st.write("Setup Information:")
#     for key, value in setup.items():
#         if isinstance(value, dict):
#             st.write(f"{key}:")
#             for sub_key, sub_value in value.items():
#                 st.write(f"  {sub_key}: {sub_value}")
#         else:
#             st.write(f"{key}: {value}")



#==========================






def load_data(uploaded_file, sep):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, sep=sep)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return pd.DataFrame()
def read_uploaded_file(file, sep) -> pd.DataFrame:
    """Return a pandas DataFrame from an uploaded CSV / Excel file."""
    return load_data(file, sep)

# Function to create and display plots
def toss(model, plot_type, session_key):
    fig, ax = plt.subplots()
    cl_plot_model(model, plot=plot_type, display_format="streamlit")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.session_state[session_key] = buf.getvalue()
    # Wyświetlanie wykresu z session_state

    if session_key in st.session_state:
        st.image(st.session_state[session_key])

# Main Page
st.title("Analiza Kluczowych Cech w Zbiorach Danych")

# Sidebar
# st.sidebar.title("Opcje")

# uploaded_file = st.sidebar.file_uploader("1. Wybierz plik", type=["csv", "json", "xls", "xlsx"])
with st.sidebar:
    st.header("📂 Select Dataset")
    # 1. Built‑in PyCaret datasets (you can add more if you like)

    builtin_ds = ["iris", "juice", "titanic", "diabetes","insurance","diamond"]

    # Dodawanie DataFrame testowego do listy zbiorników danych
    builtin_ds.append("df_test")
    selected_builtin = st.selectbox("Choose a built‑in dataset", ["-- none --"] + builtin_ds)



    # 2. File uploader
    st.subheader("Upload your own data")
    uploaded_file = st.file_uploader("CSV / Excel",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )
    col1, col2 = st.columns(2)
    with col1:
        tab = st.radio("przecinek/tabulacja:", [",", "	"])
    with col2:
        sep = st.text_input("Podaj własny separator", tab)
    nrproc = st.number_input("Ile procent losowego dataframe wziąć do analizy", 0, 100, 100)

# with st.sidebar:
    # col1, col2 = st.columns(2)
    # with col1:
    #     tab = st.radio("przecinek/tabulacja:", [",", "	"])
    # with col2:
    #     sep = st.text_input("Podaj własny separator", tab)
    # nrproc = st.number_input("Ile procent losowego dataframe wziąć do analizy", 0, 100, 100)


df: pd.DataFrame | None = None
if selected_builtin == "df_test":
    df = df_test

if selected_builtin != "-- none --":
    # load dataset from pycaret
    try:
        df = get_data(selected_builtin, profile=False)
        st.success(f"✅ Loaded **{selected_builtin}** dataset.")
    except Exception as e:
        st.error(f"Error loading built‑in dataset: {e}")

elif selected_builtin == "df_test":
    df = df_test

if uploaded_file is not None:
    # overwrite if user uploads a file
    df = read_uploaded_file(uploaded_file, sep)
    if not df.empty:
        st.success("✅ Uploaded data loaded.")


# if uploaded_file is not None:
if df is not None and not df.empty:

    if not df.empty:
        # st.success("✅ Uploaded data .")

        if df is not None:
            nr = nrproc  # Ile % data frame do trenowania
            min_df = df.sample(round((nr / 100) * len(df)))

            columns_to_ignore = st.multiselect('Ignorowane Kolumny', df.columns.tolist())
            
            if len(df.columns) < 2:
                st.warning('Ustaw separator!')
            else:
                st.info('Ustawiłeś separator poprawnie :)')
                kolumna1 = st.sidebar.selectbox('Wybierz badaną kolumnę odniesienia ', df.columns)
                
                def problem_type(df):
                    if df.select_dtypes(include='object').shape[1] > 0:
                        return "Klasyfikacja"
                    else:
                        return "Regresja"

                info = problem_type(df)
                st.write(f"Typ problemu: {info}")

                tab1, tab2, tab3 = st.tabs(["losowe wiersze", "Brakujące dane", f"Setup : {info}"])

                with tab1:
                    st.header("Losowe Wiersze")
                    st.dataframe(min_df.sample(10))

                with tab2:
                    taba, tabb = st.tabs(["Brakujące dane w %", "typy danych w kolumnach"])
                    with taba:
                        st.write(min_df.isna().sum() / len(min_df) * 100)
                    with tabb:
                        buffer = pd.io.common.StringIO()
                        min_df.info(buf=buffer)
                        s = buffer.getvalue()
                        st.text(s)
                        st.text(f"----------------------------------------------------------")
                        st.write(f"unikatowe wartości :")
                        st.write(min_df.nunique())
                        st.write(min_df.describe().round(2).T)

                with tab3:
                    if info == "Klasyfikacja":
                        # === CLASSIFICATION SETUP ===                    # tab_cl, tab_re = st.tabs([""🚀 Setup: Klasyfikacja", "typy danych w kolumnach"])
                        cl_setup(
                            data=min_df,
                            target=kolumna1,
                            session_id=123,
                            ignore_features=columns_to_ignore,
                            fix_imbalance=True,
                            normalize=True,
                            transformation=True,
                            verbose=False,
                        )
                        
                        exp = ClassificationExperiment()
                        exp.setup(
                            data=min_df,
                            target=kolumna1,
                            session_id=123,
                            ignore_features=columns_to_ignore,
                            fix_imbalance=True,
                            normalize=True,
                            transformation=True,
                            verbose=False,
                        )

                        wybor = {
                            "kompleksowy": {},
                            "szybki okrojony": {'include': ['rf', 'lr', 'gbc', 'knn'], 'fold': 5, 'verbose': False}
                        }

                        conf_compare = st.radio("zakres porównanania modeli ", list(wybor.keys()))
                        
                        if st.button("Run Classification Models"):
                            cl_best_model = exp.compare_models(**wybor[conf_compare])
                            st.session_state.cl_best_model = cl_best_model
                            
                            metrics = exp.pull()
                            st.session_state.metrics = metrics
                            st.write(st.session_state.metrics)
                            
                            for file_name in ['Feature Importance.png', 'Confusion Matrix.png', 'AUC.png']:
                                try:
                                    os.remove(file_name)
                                    print(f"Usunięto: {file_name}")
                                except FileNotFoundError:
                                    print(f"Plik nie znaleziony: {file_name}")
                                except Exception as e:
                                    print(f"Błąd przy usuwaniu pliku {file_name}: {e}")

                            if hasattr(cl_best_model, 'coef_') or hasattr(cl_best_model, 'feature_importances_'):
                                cl_plot_model(cl_best_model, plot='feature', display_format="streamlit", save=True)
                                st.image('Feature Importance.png', use_container_width=True)
                            else:
                                st.error(
                                    'Generowanie wykresu istotności cech NIE jest możliwe dla tej kolumny. Zmień kolumnę docelową.'
                                )
                            

                            cl_plot_model(cl_best_model, plot='confusion_matrix', display_format="streamlit", save=True)
                            st.image('Confusion Matrix.png', use_container_width=True)


                            cl_plot_model(cl_best_model, plot='auc', display_format="streamlit", save=True)
                            st.image('AUC.png', use_container_width=True)

                        if st.button("Zapisz modele"):
                            for file_name in ['Feature Importance.png', 'Confusion Matrix.png', 'AUC.png']:
                                if os.path.exists(file_name):
                                    new_file_name = file_name.replace('.png', '_saved.png')
                                    shutil.copy(file_name, new_file_name)
                                else:
                                    st.warning(f'Brak pliku: {file_name}')

                        if st.button("wczytaj modele"):
                            st.write(st.session_state.metrics)
                            for file_name in ['Feature Importance_saved.png', 'Confusion Matrix_saved.png', 'AUC_saved.png']:
                                if os.path.exists(file_name):
                                    st.image(file_name, use_container_width=True)
                                else:
                                    st.warning(f'Brak pliku do wyświetlenia : {file_name}')

                    # === REGRESSION SETUP ===
                    elif info == "Regresja":

                        st.subheader("🚀 Setup: Regresja")
                        
                        # Setup pycaret regression
                        re_setup(
                            data=min_df,
                            target=kolumna1,
                            session_id=123,
                            ignore_features=columns_to_ignore,
                            normalize=True,
                            transformation=True,
                            verbose=False,
                        )

                        exp_re = RegressionExperiment()  # Reuse same class, but use regression setup


                        exp_re.setup(
                            data=min_df,
                            target=kolumna1,
                            session_id=123,
                            ignore_features=columns_to_ignore,
                            normalize=True,
                            transformation=True,
                            verbose=False,
                        )

                        # Model comparison options
                        wybor_re = {
                            "kompleksowy": {},
                            "szybki okrojony": {'include': ['rf', 'lr', 'knn'], 'fold': 5, 'verbose': False}#pip install --upgrade xgboost
                        }

                        conf_compare_re = st.radio("zakres porównanania modeli ", list(wybor_re.keys()))




                        if st.button("Run Regression Models"):
                            re_best_model = exp_re.compare_models(**wybor_re[conf_compare_re])
                            st.session_state.re_best_model = re_best_model

                            metrics = exp_re.pull()
                            st.session_state.metrics = metrics
                            st.write(st.session_state.metrics)

                            # Clean up old plots
                            for file_name in [
                                'Residuals.png', 'Prediction Error.png', 'R2.png',
                                'Prediction vs Actual.png', 'Feature Importance.png'
                            ]:
                                try:
                                    os.remove(file_name)
                                except FileNotFoundError:
                                    pass
                                except Exception as e:
                                    st.warning(f"Błąd podczas usuwania {file_name}: {e}")

                            # Residuals Plot
                            fig = re_plot_model(re_best_model, plot='residuals')
                            fig.savefig('Residuals.png')
                            st.pyplot(fig)
                            # re_plot_model(re_best_model, plot='residuals', display_format="streamlit", save=True)
                            # st.image('Residuals.png', use_container_width=True)

                            # Prediction Error
                            re_plot_model(re_best_model, plot='prediction_error', display_format="streamlit", save=True)
                            st.image('Prediction Error.png', use_container_width=True)

                            # R²
                            re_plot_model(re_best_model, plot='r2', display_format="streamlit", save=True)
                            st.image('R2.png', use_container_width=True)

                            # Prediction vs Actual
                            re_plot_model(re_best_model, plot='prediction_error', display_format="streamlit", save=True)
                            st.image('Prediction vs Actual.png', use_container_width=True)

                            # Feature Importance
                            if hasattr(re_best_model, 'coef_') or hasattr(re_best_model, 'feature_importances_'):
                                re_plot_model(re_best_model, plot='feature', display_format="streamlit", save=True)
                                st.image('Feature Importance.png', use_container_width=True)
                            else:
                                st.warning('Brak danych o istotności cech dla tego modelu.')

                        if st.button("Zapisz modele (Regresja)"):
                            for file_name in [
                                'Residuals.png', 'Prediction Error.png', 'R2.png',
                                'Prediction vs Actual.png', 'Feature Importance.png'
                            ]:
                                if os.path.exists(file_name):
                                    new_file_name = file_name.replace('.png', '_saved.png')
                                    shutil.copy(file_name, new_file_name)
                                else:
                                    st.warning(f'Brak pliku: {file_name}')

                        if st.button("Wczytaj modele (Regresja)"):
                            st.write(st.session_state.metrics)
                            for file_name in [
                                'Residuals_saved.png', 'Prediction Error_saved.png', 'R2_saved.png',
                                'Prediction vs Actual_saved.png', 'Feature Importance_saved.png'
                            ]:
                                if os.path.exists(file_name):
                                    st.image(file_name, use_container_width=True)
                                else:
                                    st.warning(f'Brak pliku do wyświetlenia: {file_name}')

                          
                        plot_scatter(min_df)#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                        # display_setup_info(exp_re.setup_config)          
else:
    st.info("Wgraj plik, aby rozpocząć analizę.")
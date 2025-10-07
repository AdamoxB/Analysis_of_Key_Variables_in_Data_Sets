import streamlit as st
import pandas as pd
import numpy as np
from pycaret.datasets import get_data
from pycaret.classification import setup as cl_setup, compare_models as cl_compare_models, create_model as cl_create_model, plot_model as cl_plot_model, finalize_model as cl_finalize_model, save_model as cl_save_model, load_model as cl_load_model, predict_model as cl_predict_model, pull, ClassificationExperiment
from pycaret.regression import setup as re_setup, compare_models as re_compare_models, create_model as re_create_model, plot_model as re_plot_model, finalize_model as re_finalize_model, save_model as re_save_model, load_model as re_load_model, predict_model as re_predict_model, pull
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import io
import shutil
import time

st.set_page_config(layout='wide')

# Function to load data
def load_data(uploaded_file, sep):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, sep=sep)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
    return None

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
st.sidebar.title("Opcje")
uploaded_file = st.sidebar.file_uploader("1. Wybierz plik", type=["csv", "json", "xls", "xlsx"])

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        tab = st.radio("przecinek/tabulacja:", [",", "	"])
    with col2:
        sep = st.text_input("Podaj własny separator", tab)
    nrproc = st.number_input("Ile procent losowego dataframe wziąć do analizy", 0, 100, 100)
  

# Logica aplikacji
if uploaded_file:   
    df = load_data(uploaded_file, sep)

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


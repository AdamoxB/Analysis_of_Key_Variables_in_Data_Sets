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

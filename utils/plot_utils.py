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

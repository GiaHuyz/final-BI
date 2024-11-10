import streamlit as st
import pandas as pd
import os
from pathlib import Path

path = Path(__file__).parent / "../../data/raw/oil.csv"

def display_oil_impact():
    oil_data = pd.read_csv(path, parse_dates=["date"])
    oil_data.set_index("date", inplace=True)

    st.line_chart(oil_data["dcoilwtico"])

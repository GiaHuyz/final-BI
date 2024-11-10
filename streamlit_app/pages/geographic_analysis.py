import streamlit as st
import pandas as pd

def display_geographic_analysis(train_data):
    store_sales = train_data.groupby("store_nbr")["sales"].sum()
    st.bar_chart(store_sales)

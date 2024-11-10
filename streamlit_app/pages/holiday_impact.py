import streamlit as st
import pandas as pd

def display_holiday_impact(train_data, holidays_data):
    holiday_sales = train_data[train_data["date"].isin(holidays_data.index)].groupby("date")["sales"].sum()
    st.line_chart(holiday_sales)

    holiday_types = holidays_data["type"].value_counts()
    st.bar_chart(holiday_types)

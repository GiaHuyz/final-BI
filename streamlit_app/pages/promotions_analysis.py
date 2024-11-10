import streamlit as st
import pandas as pd

def display_promotion_analysis(train_data):
    promoted_sales = train_data.groupby("onpromotion")["sales"].sum()
    st.bar_chart(promoted_sales)

    daily_promoted_sales = train_data[train_data["onpromotion"] > 0].groupby("date")["sales"].sum()
    st.line_chart(daily_promoted_sales)

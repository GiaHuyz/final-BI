import streamlit as st
import pandas as pd

def display_product_analysis(train_data):
    st.subheader("Product Sales Analysis")
    product_sales = train_data.groupby("family")["sales"].sum().sort_values()
    st.bar_chart(product_sales)

    st.subheader("Average Sales by Product")
    product_avg_sales = train_data.groupby("family")["sales"].mean().sort_values()
    st.bar_chart(product_avg_sales)

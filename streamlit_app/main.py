import streamlit as st
import pandas as pd
import os

dirname = os.path.dirname(__file__)

st.set_page_config(page_title="Store Sales BI Dashboard", layout="wide")
st.title("Store Sales BI Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
menu_options = ["Overview", "Sales by Promotion", "Geographic Analysis", "Product Analysis", "Oil Price Impact", "Holiday Impact"]
choice = st.sidebar.selectbox("Select Analysis Page", menu_options)

# Load processed data
@st.cache_data
def load_data():
    train = pd.read_csv(os.path.join(dirname, "../data/processed/train_processed.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(dirname, "../data/processed/holidays_events_processed.csv"), parse_dates=["date"])
    return train, holidays

train_data, holidays_data = load_data()

# Conditional page loading
if choice == "Overview":
    st.subheader("Overview of Sales Data")
    from pages import overview
    overview.display_overview(train_data)

elif choice == "Sales by Promotion":
    st.subheader("Sales Analysis by Promotions")
    from pages import promotions_analysis
    promotions_analysis.display_promotion_analysis(train_data)

elif choice == "Geographic Analysis":
    st.subheader("Geographic Sales Analysis")
    from pages import geographic_analysis
    geographic_analysis.display_geographic_analysis(train_data)

elif choice == "Product Analysis":
    st.subheader("Product Sales Analysis")
    from pages import product_analysis
    product_analysis.display_product_analysis(train_data)

elif choice == "Oil Price Impact":
    st.subheader("Impact of Oil Prices on Sales")
    from pages import oil_price_impact
    oil_price_impact.display_oil_impact()

elif choice == "Holiday Impact":
    st.subheader("Impact of Holidays on Sales")
    from pages import holiday_impact
    holiday_impact.display_holiday_impact(train_data, holidays_data)

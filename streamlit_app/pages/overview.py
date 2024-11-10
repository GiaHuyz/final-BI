import streamlit as st
import pandas as pd

def display_overview(train_data):
    st.sidebar.header("Filters")

    min_date = train_data["date"].min()
    max_date = train_data["date"].max()

    # Bộ lọc theo khoảng thời gian
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )

    if st.sidebar.button("Reset Date Range"):
        date_range = [min_date, max_date]
    
    # Bộ lọc loại sản phẩm
    product_family = st.sidebar.multiselect(
        "Select Product Family", 
        options=train_data["family"].unique(), 
        default=train_data["family"].unique()
    )

    # Bộ lọc cửa hàng
    stores = st.sidebar.multiselect(
        "Select Store", 
        options=train_data["store_nbr"].unique(), 
        default=train_data["store_nbr"].unique()
    )


    # Lọc dữ liệu dựa trên lựa chọn của người dùng
    filtered_data = train_data[
        (train_data["family"].isin(product_family)) &
        (train_data["store_nbr"].isin(stores)) &
        ((train_data["date"] >= pd.to_datetime(date_range[0])) & (train_data["date"] <= pd.to_datetime(date_range[1])))
    ]

    # Hiển thị tổng doanh số
    total_sales = filtered_data["sales"].sum()
    st.metric("Total Sales", f"${total_sales:,.2f}")

    # Biểu đồ doanh số hàng ngày
    daily_sales = filtered_data.groupby("date")["sales"].sum()
    st.subheader("Daily Sales")
    st.line_chart(daily_sales)

    # Biểu đồ doanh số theo loại sản phẩm
    product_sales = filtered_data.groupby("family")["sales"].sum().sort_values()
    st.subheader("Sales by Product Family")
    st.bar_chart(product_sales)

    # Biểu đồ doanh số theo cửa hàng
    store_sales = filtered_data.groupby("store_nbr")["sales"].sum().sort_values()
    st.subheader("Sales by Store")
    st.bar_chart(store_sales)

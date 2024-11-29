import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from forecasting_model import ForecastingModel

dirname = os.path.dirname(__file__)

st.set_page_config(
    page_title="Store Sales BI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1c2539;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Store Sales BI Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
menu_options = [
    "Overview",
    "Sales Forecast",
    "Sales by Promotion",
    "Geographic Analysis",
    "Product Analysis",
    "Oil Price Impact",
    "Holiday Impact"
]
choice = st.sidebar.selectbox("Select Analysis Page", menu_options)

# Load processed data
@st.cache_data
def load_data():
    """Load all processed data files"""
    data = {}
    try:
        data['train'] = pd.read_csv(
            os.path.join(dirname, "../data/processed/train_processed.csv"),
            parse_dates=["date"]
        )
        data['stores'] = pd.read_csv(
            os.path.join(dirname, "../data/processed/stores_processed.csv")
        )
        data['transactions'] = pd.read_csv(
            os.path.join(dirname, "../data/processed/transactions_processed.csv"),
            parse_dates=["date"]
        )
        data['holidays'] = pd.read_csv(
            os.path.join(dirname, "../data/processed/holidays_processed.csv"),
            parse_dates=["date"]
        )
        data['oil'] = pd.read_csv(
            os.path.join(dirname, "../data/processed/oil_processed.csv"),
            parse_dates=["date"]
        )
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data
data = load_data()

if data is None:
    st.error("Failed to load data. Please check if the processed data files exist.")
    st.stop()


# Display key metrics in Overview
if choice == "Overview":
    st.subheader("Overview of Sales Data")
    
    # Add filters in sidebar
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = data['train']["date"].min()
    max_date = data['train']["date"].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )
    
    if st.sidebar.button("Reset Date Range"):
        date_range = [min_date, max_date]
    
    # Product family filter
    product_family = st.sidebar.multiselect(
        "Select Product Family", 
        options=data['train']["family"].unique(), 
        default=data['train']["family"].unique()
    )
    
    # Store filter
    stores = st.sidebar.multiselect(
        "Select Store", 
        options=data['train']["store_nbr"].unique(), 
        default=data['train']["store_nbr"].unique()
    )
    
    # Filter data based on user selection
    filtered_data = data['train'][
        (data['train']["family"].isin(product_family)) &
        (data['train']["store_nbr"].isin(stores)) &
        ((data['train']["date"] >= pd.to_datetime(date_range[0])) & 
         (data['train']["date"] <= pd.to_datetime(date_range[1])))
    ]
    
    # Key metrics with filtered data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_data['sales'].sum()
        st.metric("Total Sales", f"${total_sales:,.2f}")
    
    with col2:
        avg_daily_sales = filtered_data.groupby('date')['sales'].sum().mean()
        st.metric("Average Daily Sales", f"${avg_daily_sales:,.2f}")
    
    with col3:
        total_stores = filtered_data['store_nbr'].nunique()
        st.metric("Total Stores", total_stores)
    
    with col4:
        total_products = filtered_data['family'].nunique()
        st.metric("Product Families", total_products)
    
    # Recent trends with filtered data
    st.subheader("Recent Sales Trends")
    recent_sales = filtered_data.groupby('date')['sales'].sum().reset_index()
    fig = px.line(
        recent_sales,
        x='date',
        y='sales',
        title='Daily Total Sales'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performing stores and products with filtered data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Performing Stores")
        top_stores = filtered_data.groupby('store_nbr')['sales'].mean().reset_index()
        top_stores = top_stores.sort_values('sales', ascending=False).head(10)
        fig = px.bar(
            top_stores,
            x='store_nbr',
            y='sales',
            title='Top 10 Stores by Average Sales'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Product Families")
        top_products = filtered_data.groupby('family')['sales'].mean().reset_index()
        top_products = top_products.sort_values('sales', ascending=False).head(10)
        fig = px.bar(
            top_products,
            x='family',
            y='sales',
            title='Top 10 Product Families by Average Sales'
        )
        st.plotly_chart(fig, use_container_width=True)

elif choice == "Sales Forecast":
    st.subheader("Sales Forecasting")
    
    # Add filters for forecasting
    st.sidebar.header("Forecast Settings")
    
    # Store selection
    store_numbers = sorted(data['train']['store_nbr'].unique())
    selected_store = st.sidebar.selectbox(
        "Select Store",
        store_numbers,
        index=0
    )
    
    # Product family selection
    families = sorted(data['train']['family'].unique())
    selected_family = st.sidebar.selectbox(
        "Select Product Family",
        families,
        index=0
    )
    
    # Forecast horizon selection
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=30,
        value=14,
        step=7
    )
    
    # Initialize and load forecasting model
    model_path = os.path.join(dirname, "../models/lgbm_model.pkl")
    forecaster = ForecastingModel(model_path=model_path)
    
    # Filter data for selected store and family
    store_family_data = data['train'][
        (data['train']['store_nbr'] == selected_store) & 
        (data['train']['family'] == selected_family)
    ].copy()
    
    if len(store_family_data) == 0:
        st.warning("No historical data available for this store and product family combination.")
        st.stop()
        
    # Calculate and display metrics
    historical_data = store_family_data.sort_values('date')
    last_n_days = historical_data.tail(forecast_days)
    avg_daily_sales = last_n_days['sales'].mean()
    avg_daily_sales_fmt = '{:,.2f}'.format(avg_daily_sales)
    
    # Prepare data for forecasting
    latest_date = store_family_data['date'].max()
    forecast_start = latest_date + timedelta(days=1)
    forecast_end = forecast_start + timedelta(days=forecast_days-1)
    
    # Create future dates DataFrame
    future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq='D')
    future_df = pd.DataFrame({'date': future_dates})
    future_df['store_nbr'] = selected_store
    future_df['family'] = selected_family
    
    # Add necessary features from other datasets
    future_df = pd.merge(future_df, data['stores'][['store_nbr', 'type_code', 'cluster_code']], 
                        on='store_nbr', how='left')
    
    # Handle oil prices
    latest_oil_price = data['oil'].loc[data['oil']['date'] <= latest_date, 'dcoilwtico'].iloc[-1]
    future_df['dcoilwtico'] = latest_oil_price
    
    # Handle holidays
    future_df = pd.merge(future_df, data['holidays'], on='date', how='left')
    holiday_cols = [col for col in future_df.columns if col.startswith('holiday_') and col.endswith('_y')]
    future_df[holiday_cols] = future_df[holiday_cols].fillna(0)
    
    # Set default values for other features
    future_df['onpromotion'] = 0  # Assume no promotions for future dates
    
    # Combine historical and future data for feature creation
    historical_data = store_family_data[['date', 'store_nbr', 'family', 'sales', 'onpromotion']].copy()
    historical_data = pd.merge(
        historical_data,
        data['stores'][['store_nbr', 'type_code', 'cluster_code']],
        on='store_nbr',
        how='left'
    )
    historical_data = pd.merge(
        historical_data,
        data['oil'][['date', 'dcoilwtico']],
        on='date',
        how='left'
    )
    historical_data = pd.merge(
        historical_data,
        data['holidays'],
        on='date',
        how='left'
    )
    
    combined_df = pd.concat([historical_data, future_df], axis=0).sort_values('date')
    combined_df[holiday_cols] = combined_df[holiday_cols].fillna(0)
    combined_df['dcoilwtico'] = combined_df['dcoilwtico'].fillna(method='ffill')
    
    # Make predictions
    try:
        predictions = forecaster.predict(combined_df)
        if predictions is not None:
            print(f"Generated predictions shape: {predictions.shape}")
            print(f"Future df shape: {future_df.shape}")
            print(f"Forecast days: {forecast_days}")
            print(f"Predictions range: {predictions.min():.2f} to {predictions.max():.2f}")
            
            # Ensure we're getting the right number of predictions
            future_predictions = predictions[-forecast_days:].copy()  # Make a copy of the predictions
            print(f"Future predictions shape: {future_predictions.shape}")
            print(f"Future predictions range: {future_predictions.min():.2f} to {future_predictions.max():.2f}")
            
            # Create a fresh copy of the DataFrame
            future_df = future_df.copy()
            
            # Create a new predicted_sales column with numpy array
            future_df['predicted_sales'] = np.array(future_predictions)
            
            print("\nVerification after assignment:")
            print("Predictions array:", future_predictions)
            print("\nFirst few rows of future_df:")
            print(future_df[['date', 'store_nbr', 'family', 'predicted_sales']].head())
            
            # Verify the assignment worked
            if future_df['predicted_sales'].isna().any():
                print("\nWarning: NaN values found in predictions")
                print("Rows with NaN:")
                print(future_df[future_df['predicted_sales'].isna()][['date', 'store_nbr', 'family', 'predicted_sales']])
            
            print(f"\nFuture df predictions range: {future_df['predicted_sales'].min():.2f} to {future_df['predicted_sales'].max():.2f}")
            print(f"Number of NaN values in predictions: {future_df['predicted_sales'].isna().sum()}")
            
            avg_forecast = future_df['predicted_sales'].mean()
            avg_forecast_fmt = '{:,.2f}'.format(avg_forecast)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"Average Daily Sales (Last {forecast_days} Days)",
                    avg_daily_sales_fmt,
                    delta=None
                )
            
            with col2:
                st.metric(
                    f"Average Forecasted Sales (Next {forecast_days} Days)",
                    avg_forecast_fmt,
                    delta='{:,.2f}'.format(avg_forecast - avg_daily_sales),
                    delta_color="normal"
                )
            
            with col3:
                trend = ((avg_forecast - avg_daily_sales) / avg_daily_sales) * 100
                st.metric("Sales Trend", f"{trend:,.1f}%")
            
            # Create visualization
            fig = go.Figure()
            
            # Historical sales (last 30 days)
            fig.add_trace(go.Scatter(
                x=store_family_data['date'].tail(30),
                y=store_family_data['sales'].tail(30),
                name='Historical Sales',
                line=dict(color='blue')
            ))
            
            # Forecasted sales
            fig.add_trace(go.Scatter(
                x=future_df['date'],
                y=future_df['predicted_sales'],
                name='Forecasted Sales',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Sales Forecast for Store {selected_store} - {selected_family}',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast table
            st.subheader("Forecast Table")
            
            # Debug information
            print("\nDebug Information:")
            print(f"Future DataFrame Info:")
            print(future_df.info())
            print("\nFuture DataFrame Head:")
            print(future_df.head())
            print("\nFuture DataFrame Tail:")
            print(future_df.tail())
            
            # Format the date column
            future_df['date'] = pd.to_datetime(future_df['date']).dt.strftime('%Y-%m-%d')
            
            # Format sales with thousand separator and 2 decimal places
            future_df['predicted_sales'] = future_df['predicted_sales'].apply(lambda x: '{:,.2f}'.format(x))
            
            # Display the forecast table
            st.dataframe(
                future_df[['date', 'predicted_sales']],
                column_config={
                    "date": "Date",
                    "predicted_sales": "Predicted Sales"
                },
                hide_index=True
            )
            
        else:
            st.error("Failed to generate predictions. Please check the model and try again.")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

elif choice == "Sales by Promotion":
    def analyze_promotions(data):
        st.header("Sales Analysis by Promotions")
        
        try:
            # Calculate promotion metrics
            promo_stats = data['train'].groupby('onpromotion').agg({
                'sales': ['mean', 'median', 'count', 'std']
            }).round(2)
            promo_stats.columns = ['mean_sales', 'median_sales', 'count', 'std_sales']
            promo_stats = promo_stats.reset_index()
            
            # Calculate percentage difference
            regular_sales = promo_stats.loc[promo_stats['onpromotion'] == 0, 'mean_sales'].values[0]
            promo_sales = promo_stats.loc[promo_stats['onpromotion'] == 1, 'mean_sales'].values[0]
            sales_increase = ((promo_sales - regular_sales) / regular_sales) * 100
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Average Sales - Regular",
                    f"${regular_sales:,.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Average Sales - On Promotion",
                    f"${promo_sales:,.2f}",
                    delta=f"{sales_increase:+.1f}%",
                    delta_color="normal"
                )
            
            with col3:
                promo_count = promo_stats.loc[promo_stats['onpromotion'] == 1, 'count'].values[0]
                total_count = promo_stats['count'].sum()
                promo_percentage = (promo_count / total_count) * 100
                st.metric(
                    "Products on Promotion",
                    f"{promo_count:,}",
                    f"{promo_percentage:.1f}% of total"
                )
            
            # Create bar chart comparing sales distribution
            fig = go.Figure()
            
            # Add bar for regular sales
            fig.add_trace(go.Bar(
                name='Regular Sales',
                x=['Regular'],
                y=[regular_sales],
                error_y=dict(
                    type='data',
                    array=[promo_stats.loc[promo_stats['onpromotion'] == 0, 'std_sales'].values[0]],
                    visible=True
                ),
                marker_color='rgb(55, 83, 109)'
            ))
            
            # Add bar for promotional sales
            fig.add_trace(go.Bar(
                name='Promotional Sales',
                x=['On Promotion'],
                y=[promo_sales],
                error_y=dict(
                    type='data',
                    array=[promo_stats.loc[promo_stats['onpromotion'] == 1, 'std_sales'].values[0]],
                    visible=True
                ),
                marker_color='rgb(26, 118, 255)'
            ))
            
            # Update layout
            fig.update_layout(
                title='Average Sales Comparison: Regular vs Promotional',
                xaxis_title='Sales Type',
                yaxis_title='Average Sales ($)',
                barmode='group',
                showlegend=True,
                height=500,
                title_x=0.5
            )
            
            # Add hover template
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Sales Type: %{x}",
                    "Average Sales: $%{y:,.2f}",
                    "Standard Deviation: $%{error_y.array:,.2f}"
                ])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add detailed statistics table
            st.subheader("Detailed Statistics")
            
            stats_table = pd.DataFrame({
                'Metric': ['Regular Sales', 'Promotional Sales'],
                'Average Sales': [f"${regular_sales:,.2f}", f"${promo_sales:,.2f}"],
                'Median Sales': [
                    f"${promo_stats.loc[promo_stats['onpromotion'] == 0, 'median_sales'].values[0]:,.2f}",
                    f"${promo_stats.loc[promo_stats['onpromotion'] == 1, 'median_sales'].values[0]:,.2f}"
                ],
                'Number of Products': [
                    f"{promo_stats.loc[promo_stats['onpromotion'] == 0, 'count'].values[0]:,}",
                    f"{promo_stats.loc[promo_stats['onpromotion'] == 1, 'count'].values[0]:,}"
                ],
                'Standard Deviation': [
                    f"${promo_stats.loc[promo_stats['onpromotion'] == 0, 'std_sales'].values[0]:,.2f}",
                    f"${promo_stats.loc[promo_stats['onpromotion'] == 1, 'std_sales'].values[0]:,.2f}"
                ]
            })
            
            st.dataframe(
                stats_table,
                column_config={
                    "Metric": st.column_config.TextColumn("Sales Type", width="medium"),
                    "Average Sales": st.column_config.TextColumn("Average Sales", width="medium"),
                    "Median Sales": st.column_config.TextColumn("Median Sales", width="medium"),
                    "Number of Products": st.column_config.TextColumn("Number of Products", width="medium"),
                    "Standard Deviation": st.column_config.TextColumn("Standard Deviation", width="medium")
                },
                hide_index=True,
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error analyzing promotions: {str(e)}")
            print(f"Detailed error: {e}")
    
    analyze_promotions(data)

elif choice == "Geographic Analysis":
    st.header("Geographic Sales Analysis")
    
    try:
        # Prepare data for geographic analysis
        print("Preparing data for geographic analysis...")
        print("Stores data columns:", data['stores'].columns.tolist())
        print("Train data columns:", data['train'].columns.tolist())
        
        # Calculate total sales per store
        store_sales = data['train'].groupby('store_nbr')['sales'].sum().reset_index()
        print("\nStore sales shape:", store_sales.shape)
        
        # Merge with store information
        geo_sales = pd.merge(
            store_sales,
            data['stores'],
            on='store_nbr',
            how='left'
        )
        print("\nGeo sales columns after merge:", geo_sales.columns.tolist())
        
        # Group by city and calculate total sales
        city_sales = geo_sales.groupby('city')['sales'].sum().sort_values(ascending=False)
        print("\nTop 5 cities by sales:")
        print(city_sales.head())
        
        # Create a bar chart for city-wise sales
        fig = px.bar(
            x=city_sales.index,
            y=city_sales.values,
            title='Sales by City',
            labels={'x': 'City', 'y': 'Total Sales ($)'},
            color=city_sales.values,
            color_continuous_scale='Viridis'
        )
        
        # Customize layout
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            height=500,
            yaxis_title="Total Sales ($)",
            xaxis_title="City",
            title_x=0.5,
            margin=dict(t=50, b=100)  # Increase bottom margin for rotated labels
        )
        
        # Format axis labels
        fig.update_traces(
            hovertemplate="<br>".join([
                "City: %{x}",
                "Sales: $%{y:,.2f}",
            ])
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top cities table
        st.subheader("Top Cities by Sales")
        top_cities = pd.DataFrame({
            'City': city_sales.index,
            'Total Sales': city_sales.values
        }).head(10)
        
        # Format sales values
        top_cities['Total Sales'] = top_cities['Total Sales'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            top_cities,
            column_config={
                "City": st.column_config.TextColumn("City", width="medium"),
                "Total Sales": st.column_config.TextColumn("Total Sales", width="medium")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Add some insights
        st.subheader("Geographic Insights")
        total_sales = city_sales.sum()
        top_5_sales = city_sales.head().sum()
        top_5_percentage = (top_5_sales / total_sales) * 100
        
        st.write(f"""
        - Top 5 cities account for {top_5_percentage:.1f}% of total sales
        - Average sales per city: ${city_sales.mean():,.2f}
        - Number of cities: {len(city_sales)}
        """)
        
    except Exception as e:
        st.error(f"Error analyzing geographic sales: {str(e)}")
        print(f"Detailed error: {e}")
        print("\nDataFrame info:")
        if 'geo_sales' in locals():
            print("\nGeo sales info:")
            print(geo_sales.info())
        else:
            print("\nStores info:")
            print(data['stores'].info())
            print("\nTrain info:")
            print(data['train'].info())

elif choice == "Product Analysis":
    st.subheader("Product Sales Analysis")
    
    # Add filters
    st.sidebar.header("Product Analysis Filters")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [data['train']['date'].min(), data['train']['date'].max()]
    )
    
    # Filter data by date
    mask = (data['train']['date'] >= pd.to_datetime(date_range[0])) & (data['train']['date'] <= pd.to_datetime(date_range[1]))
    filtered_data = data['train'][mask]
    
    # Product family performance
    family_sales = filtered_data.groupby('family')['sales'].agg(['sum', 'mean']).round(2)
    family_sales.columns = ['Total Sales', 'Average Daily Sales']
    family_sales = family_sales.sort_values('Total Sales', ascending=False)
    
    # Display product performance
    st.write("Product Family Performance")
    st.write(family_sales)
    
    # Visualization
    fig = px.bar(
        family_sales.reset_index(),
        x='family',
        y='Total Sales',
        title='Total Sales by Product Family'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sales trends for top families
    top_families = family_sales.head(5).index
    top_family_trends = filtered_data[filtered_data['family'].isin(top_families)]
    
    fig2 = px.line(
        top_family_trends.groupby(['date', 'family'])['sales'].mean().reset_index(),
        x='date',
        y='sales',
        color='family',
        title='Sales Trends for Top 5 Product Families'
    )
    st.plotly_chart(fig2, use_container_width=True)

elif choice == "Oil Price Impact":
    def analyze_oil_impact(data):
        st.header("Impact of Oil Prices on Sales")
        
        try:
            # Merge sales data with oil data
            oil_sales = pd.merge(
                data['train'].groupby('date')['sales'].sum().reset_index(),
                data['oil'],
                on='date',
                how='left'
            )
            
            # Calculate correlation
            correlation = oil_sales['sales'].corr(oil_sales['dcoilwtico'])
            
            # Calculate average sales for different oil price ranges
            oil_sales['price_range'] = pd.qcut(oil_sales['dcoilwtico'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            price_range_stats = oil_sales.groupby('price_range').agg({
                'dcoilwtico': ['mean', 'min', 'max'],
                'sales': ['mean', 'std', 'count']
            }).round(2)
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Correlation Coefficient",
                    f"{correlation:.3f}",
                    delta=None
                )
            
            with col2:
                # Calculate average sales change between low and high oil prices
                low_sales = price_range_stats.loc['Low', ('sales', 'mean')]
                high_sales = price_range_stats.loc['High', ('sales', 'mean')]
                sales_change = ((high_sales - low_sales) / low_sales) * 100
                st.metric(
                    "Sales Change (Low to High Oil)",
                    f"{sales_change:+.1f}%",
                    delta=None
                )
            
            with col3:
                # Display oil price range
                min_oil = oil_sales['dcoilwtico'].min()
                max_oil = oil_sales['dcoilwtico'].max()
                st.metric(
                    "Oil Price Range",
                    f"${min_oil:.2f} - ${max_oil:.2f}",
                    delta=None
                )
            
            # Create scatter plot with trend line
            fig1 = px.scatter(
                oil_sales,
                x='dcoilwtico',
                y='sales',
                trendline="ols",
                title='Sales vs Oil Price Correlation',
                labels={
                    'dcoilwtico': 'Oil Price ($)',
                    'sales': 'Total Daily Sales ($)'
                }
            )
            
            # Update layout
            fig1.update_layout(
                height=500,
                title_x=0.5,
                showlegend=True
            )
            
            # Add hover template
            fig1.update_traces(
                hovertemplate="<br>".join([
                    "Oil Price: $%{x:.2f}",
                    "Daily Sales: $%{y:,.2f}"
                ])
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Create box plot for sales distribution by oil price range
            fig2 = px.box(
                oil_sales,
                x='price_range',
                y='sales',
                title='Sales Distribution by Oil Price Range',
                labels={
                    'price_range': 'Oil Price Range',
                    'sales': 'Total Daily Sales ($)'
                },
                color='price_range',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Update layout
            fig2.update_layout(
                height=500,
                title_x=0.5,
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display detailed statistics table
            st.subheader("Sales Statistics by Oil Price Range")
            
            stats_table = pd.DataFrame({
                'Price Range': price_range_stats.index,
                'Average Oil Price': [f"${x:,.2f}" for x in price_range_stats[('dcoilwtico', 'mean')]],
                'Price Range': [
                    f"${min_:.2f} - ${max_:.2f}" 
                    for min_, max_ in zip(
                        price_range_stats[('dcoilwtico', 'min')],
                        price_range_stats[('dcoilwtico', 'max')]
                    )
                ],
                'Average Daily Sales': [f"${x:,.2f}" for x in price_range_stats[('sales', 'mean')]],
                'Sales Std Dev': [f"${x:,.2f}" for x in price_range_stats[('sales', 'std')]],
                'Number of Days': price_range_stats[('sales', 'count')].astype(int)
            })
            
            st.dataframe(
                stats_table,
                column_config={
                    "Price Range": st.column_config.TextColumn("Oil Price Range", width="medium"),
                    "Average Oil Price": st.column_config.TextColumn("Average Oil Price", width="medium"),
                    "Price Range": st.column_config.TextColumn("Price Range (Min-Max)", width="medium"),
                    "Average Daily Sales": st.column_config.TextColumn("Average Daily Sales", width="medium"),
                    "Sales Std Dev": st.column_config.TextColumn("Sales Std Dev", width="medium"),
                    "Number of Days": st.column_config.NumberColumn("Number of Days", width="small")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add time series analysis
            st.subheader("Oil Price and Sales Trends Over Time")
            
            # Create dual-axis time series plot
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add sales line
            fig3.add_trace(
                go.Scatter(
                    x=oil_sales['date'],
                    y=oil_sales['sales'],
                    name="Daily Sales",
                    line=dict(color='rgb(55, 83, 109)')
                ),
                secondary_y=True
            )
            
            # Add oil price line
            fig3.add_trace(
                go.Scatter(
                    x=oil_sales['date'],
                    y=oil_sales['dcoilwtico'],
                    name="Oil Price",
                    line=dict(color='rgb(26, 118, 255)')
                ),
                secondary_y=False
            )
            
            # Update layout
            fig3.update_layout(
                title='Oil Price and Sales Trends',
                height=500,
                title_x=0.5,
                hovermode='x unified'
            )
            
            # Update axes
            fig3.update_xaxes(title_text="Date")
            fig3.update_yaxes(title_text="Oil Price ($)", secondary_y=False)
            fig3.update_yaxes(title_text="Total Daily Sales ($)", secondary_y=True)
            
            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing oil impact: {str(e)}")
            print(f"Detailed error: {e}")
    
    analyze_oil_impact(data)

elif choice == "Holiday Impact":
    st.header("Impact of Holidays on Sales")
    
    try:
        # Print available columns for debugging
        print("\nHolidays data columns:", data['holidays'].columns.tolist())
        print("Sample holidays data:")
        print(data['holidays'].head())
        
        # Merge sales data with holidays
        sales_with_holidays = pd.merge(
            data['train'],
            data['holidays'],
            on='date',
            how='left'
        )
        
        print("\nMerged data columns:", sales_with_holidays.columns.tolist())
        print("Sample merged data:")
        print(sales_with_holidays.head())
        
        # Create holiday flag (check if any holiday columns have values)
        holiday_columns = [col for col in sales_with_holidays.columns if col.startswith('holiday_') and col.endswith('_y')]
        if not holiday_columns:
            print("\nAvailable columns:", sales_with_holidays.columns.tolist())
            raise ValueError("No holiday columns found in the data")
            
        print("\nHoliday columns found:", holiday_columns)
        sales_with_holidays['is_holiday'] = (sales_with_holidays[holiday_columns] > 0).any(axis=1)
        
        # Calculate average daily sales for holiday vs non-holiday
        holiday_impact = sales_with_holidays.groupby('is_holiday')['sales'].agg([
            'mean',
            'count',
            'std'
        ]).round(2)
        
        # Calculate percentage difference
        non_holiday_sales = holiday_impact.loc[False, 'mean']
        holiday_sales = holiday_impact.loc[True, 'mean']
        sales_increase = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Sales - Regular Days",
                f"${non_holiday_sales:,.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Average Sales - Holidays",
                f"${holiday_sales:,.2f}",
                delta=f"{sales_increase:+.1f}%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Number of Holiday Days",
                f"{int(holiday_impact.loc[True, 'count']):,}",
                delta=None
            )
        
        # Analyze sales by holiday type
        holiday_stats = []
        
        for holiday_col in holiday_columns:
            # Calculate stats for this holiday type
            holiday_data = sales_with_holidays[sales_with_holidays[holiday_col] > 0]
            if len(holiday_data) > 0:
                stats = holiday_data['sales'].agg(['mean', 'count', 'std']).round(2)
                # Extract holiday type from column name (remove 'holiday_' prefix and '_y' suffix)
                holiday_type = holiday_col.replace('holiday_', '').replace('_y', '')
                holiday_stats.append({
                    'holiday_type': holiday_type,
                    'mean': stats['mean'],
                    'count': stats['count'],
                    'std': stats['std']
                })
        
        # Create DataFrame from collected stats
        holiday_type_sales = pd.DataFrame(holiday_stats)
        if len(holiday_type_sales) == 0:
            st.warning("No holiday data found for analysis")
            st.stop()
            
        holiday_type_sales = holiday_type_sales.sort_values('mean', ascending=False)
        
        # Create bar chart for holiday type impact
        fig = px.bar(
            holiday_type_sales,
            y='holiday_type',
            x='mean',
            title='Average Daily Sales by Holiday Type',
            labels={
                'mean': 'Average Daily Sales ($)',
                'holiday_type': 'Holiday Type'
            },
            color='mean',
            color_continuous_scale='Viridis'
        )
        
        # Customize layout
        fig.update_layout(
            height=500,
            yaxis_title="Holiday Type",
            xaxis_title="Average Daily Sales ($)",
            title_x=0.5,
            showlegend=False
        )
        
        # Format hover template
        fig.update_traces(
            hovertemplate="<br>".join([
                "Holiday Type: %{y}",
                "Average Sales: $%{x:,.2f}",
            ])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed holiday impact table
        st.subheader("Detailed Holiday Impact Analysis")
        
        holiday_impact_table = pd.DataFrame({
            'Holiday Type': holiday_type_sales['holiday_type'],
            'Average Daily Sales': holiday_type_sales['mean'].apply(lambda x: f"${x:,.2f}"),
            'Number of Days': holiday_type_sales['count'].astype(int),
            'Sales Increase': ((holiday_type_sales['mean'] - non_holiday_sales) / non_holiday_sales * 100).apply(lambda x: f"{x:+.1f}%")
        })
        
        st.dataframe(
            holiday_impact_table,
            column_config={
                "Holiday Type": st.column_config.TextColumn("Holiday Type", width="medium"),
                "Average Daily Sales": st.column_config.TextColumn("Average Daily Sales", width="medium"),
                "Number of Days": st.column_config.NumberColumn("Number of Days", width="small"),
                "Sales Increase": st.column_config.TextColumn("vs Regular Days", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error analyzing holiday impact: {str(e)}")
        print(f"Detailed error: {e}")
        print("\nDataFrame info:")
        if 'sales_with_holidays' in locals():
            print("\nMerged data info:")
            print(sales_with_holidays.info())
            print("\nSample merged data:")
            print(sales_with_holidays.head())

# Footer
st.markdown("---")
st.markdown("Store Sales BI Dashboard - Powered by Streamlit")

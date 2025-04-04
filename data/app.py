import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from feature import clean_data, add_features, customer_segmentation,prepare_churn_data,train_churn_model, predict_sales_with_prophet,prepare_sales_data_for_prophet
from visualise import plot_prophet_forecast,plot_sales_heatmap, plot_category_sales, enable_data_download , sales_by_hour,sales_by_Time,sales_by_product_category,product_specific_analysis

st.title("Supermarket Sales Dashboard")
st.header("Initial Data Exploration & Cleaning")

# Use st.cache to avoid reloading data on every app rerun
@st.cache_resource
def load_data():
    # Load the dataset from the data folder
    df = pd.read_csv("D:\supermarket-dashboard\data\supermarket_sales.csv")
    return df

# Load the raw data
data = load_data()

# Display statistical summary of the data
st.subheader("Statistical Summary")
st.write(data.describe())

# Data Cleaning Section
st.header("Data Cleaning")
st.write("Removing duplicates and handling missing values...")

# Remove duplicate rows
data_cleaned = data.drop_duplicates()

# Fill missing values using forward fill (you can choose another method if preferred)
data_cleaned = data_cleaned.fillna(method='ffill')

# Convert the 'Date' column to datetime if it exists
if 'Date' in data_cleaned.columns:
    data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')

st.subheader("Cleaned Data Preview")
st.write(data_cleaned.head())
data_segmented = customer_segmentation(data_cleaned, n_clusters=3)

st.write("Data cleaning is complete.")

# --- NEW CODE: EXPLORATORY DATA ANALYSIS ---
st.header("Exploratory Data Analysis")

# Sidebar for filter controls
st.sidebar.header("Filters")

# Assuming we have these columns in the dataset (adjust based on actual data)
if 'Product line' in data_cleaned.columns:
    product_categories = ['All'] + sorted(data_cleaned['Product line'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Product Category", product_categories, key="product_category_1")

if 'Customer type' in data_cleaned.columns:
    customer_types = ['All'] + sorted(data_cleaned['Customer type'].unique().tolist())
    selected_customer_type = st.sidebar.selectbox("Select Customer Type", customer_types, key="customer_type_1")

if 'Gender' in data_cleaned.columns:
    genders = ['All'] + sorted(data_cleaned['Gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Select Gender", genders,key="gender_1")

if 'Date' in data_cleaned.columns:
    # Create date range selector
    min_date = data_cleaned['Date'].min().date()
    max_date = data_cleaned['Date'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range_1"
    )

# Filter data based on selections
filtered_data = data_cleaned.copy()

if 'Product line' in data_cleaned.columns and selected_category != 'All':
    filtered_data = filtered_data[filtered_data['Product line'] == selected_category]

if 'Customer type' in data_cleaned.columns and selected_customer_type != 'All':
    filtered_data = filtered_data[filtered_data['Customer type'] == selected_customer_type]

if 'Gender' in data_cleaned.columns and selected_gender != 'All':
    filtered_data = filtered_data[filtered_data['Gender'] == selected_gender]

if 'Date' in data_cleaned.columns and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_data = filtered_data[(filtered_data['Date'].dt.date >= start_date) & 
                                 (filtered_data['Date'].dt.date <= end_date)]

# Display count of filtered records
st.subheader(f"Filtered Data: {len(filtered_data)} records")

# --- ANALYSIS BY DISTRIBUTION ---
st.subheader("Sales Distribution Analysis")

# Distribution of sales by product category

sales_by_product_category(filtered_data)


# Distribution of sales by payment method

if 'Payment' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales by Payment Method")
    
    payment_sales = filtered_data.groupby('Payment')['Total'].sum().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        payment_sales,
        x='Payment',
        y='Total',
        title='Total Sales by Payment Method',
        labels={'Payment': 'Payment Method', 'Total': 'Total Sales'},
        color='Payment',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Payment Method',
        yaxis_title='Total Sales',
        plot_bgcolor='white',
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate='Payment Method: %{x}<br>Total Sales: %{y:,.2f}<extra></extra>'
    )
    
    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Distribution by customer demographics
if 'Gender' in filtered_data.columns and 'Customer type' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales by Customer Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender_sales = filtered_data.groupby('Gender')['Total'].sum().reset_index()
        fig_gender = px.pie(
            gender_sales,
            values='Total',
            names='Gender',
            title='Sales by Gender',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3  # Creates a donut chart for modern look
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        fig_gender.update_layout(
            legend_title_text='Gender',
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        customer_type_sales = filtered_data.groupby('Customer type')['Total'].sum().reset_index()
        fig_cust_type = px.pie(
            customer_type_sales,
            values='Total',
            names='Customer type',
            title='Sales by Customer Type',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.3  # Creates a donut chart for modern look
        )
        fig_cust_type.update_traces(textposition='inside', textinfo='percent+label')
        fig_cust_type.update_layout(
            legend_title_text='Customer Type',
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_cust_type, use_container_width=True)

# Time-based analysis
sales_by_Time(filtered_data)

# Hour-based analysis
st.subheader("Sales by Hour of Day")
sales_by_hour(filtered_data) 

# Correlation analysis
st.subheader("Correlation Analysis")

# Select only numeric columns for correlation
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
if len(numeric_cols) > 1:
    corr_matrix = filtered_data[numeric_cols].corr()
        
    # Convert correlation matrix to heatmap format
    fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=numeric_cols,
            y=numeric_cols,
            colorscale="RdBu",
            annotation_text=corr_matrix.round(2).values,
            showscale=True
        )

    fig.update_layout(
            title="📊 Correlation Matrix of Numeric Variables",
            xaxis_title="Features",
            yaxis_title="Features",
            font=dict(family="Arial", size=12),
            width=800, height=700
        )

    st.plotly_chart(fig)

    st.write("""
        ### **Correlation Analysis Insights**
        - 🔵 **Strong positive correlations** (closer to `1.0`) indicate variables increasing together.
        - 🔴 **Strong negative correlations** (closer to `-1.0`) indicate inverse relationships.
        - ⚪ **Values close to `0.0`** suggest little or no correlation.
        """)
# Product-specific analysis
product_specific_analysis(filtered_data)


# Visualizations
if 'Date' in data_cleaned.columns and 'Total' in data_cleaned.columns:
    st.write("### Sales Heatmap by Day and Hour")
    plot_sales_heatmap(data_cleaned)

if 'Product line' in data_cleaned.columns and 'Total' in data_cleaned.columns:
    plot_category_sales(data_cleaned)

# Forecasting
st.subheader("Sales Forecast")
...
prophet_ready_df = prepare_sales_data_for_prophet(data_cleaned, date_column='Date', sales_column='Total')
forecast_df, model = predict_sales_with_prophet(prophet_ready_df, periods=30)
fig = plot_prophet_forecast(forecast_df)
st.plotly_chart(fig)


# Customer Segmentation
st.header("Customer Segmentation Analysis")

# Use Streamlit tabs to organize your segmentation outputs
tab1, tab2 = st.tabs(["Cluster Overview", "Detailed Analysis"])

with tab1:
    # 1. Table preview
    if 'Cluster' in data_segmented.columns:
        st.subheader("Segmented Data Preview")
        st.write(
            data_segmented[
                [
                    'Invoice ID', 'Gender', 'Customer type',
                    'Product line', 'Total', 'Quantity', 'Cluster'
                ]
            ].head(10)
        )

        # 2. Scatter plot (Quantity vs. Total)
        st.subheader("Scatter Plot: Quantity vs. Total by Cluster")
        scatter_data = data_segmented.dropna(subset=['Quantity', 'Total', 'Cluster'])
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=scatter_data, 
            x='Quantity', 
            y='Total', 
            hue='Cluster', 
            palette='viridis', 
            ax=ax_scatter
        )
        ax_scatter.set_title("Clusters Based on Quantity vs. Total")
        st.pyplot(fig_scatter)
        
        # 3. Cluster distribution
        cluster_counts = data_segmented['Cluster'].value_counts()
        st.write("### Cluster Distribution")
        for cluster_label, count in cluster_counts.items():
            st.write(f"Cluster {cluster_label}: {count} rows")

    else:
        st.warning("Customer Segmentation could not be performed or 'Cluster' column not found.")

with tab2:
    # 4. Catplot: Demographic & Product Preferences by Cluster
    if (
        'Cluster' in data_segmented.columns
        and 'Gender' in data_segmented.columns
        and 'Product line' in data_segmented.columns
    ):
        st.subheader("Demographic & Product Preferences by Cluster")
        st.write("How do demographics (e.g., Gender) and product preferences vary across different clusters?")

        catplot_data = data_segmented.dropna(subset=['Gender', 'Product line', 'Cluster'])
        
        g_cat = sns.catplot(
            data=catplot_data,
            x='Product line',
            hue='Gender',
            col='Cluster',
            kind='count',
            col_wrap=3,
            height=4,
            aspect=1.2,
            sharex=False,
            sharey=False
        )
        g_cat.set_xticklabels(rotation=45)
        g_cat.set_titles("Cluster {col_name}")
        st.pyplot(g_cat.fig)
    else:
        st.warning("Required columns for demographic & product cluster analysis are missing.")

    # 5. Box Plot: Distribution of Total Spending by Cluster
    if 'Cluster' in data_segmented.columns and 'Total' in data_segmented.columns:
        st.subheader("Distribution of Total Spending by Cluster")
        box_data = data_segmented.dropna(subset=['Cluster', 'Total'])
        
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=box_data, 
            x='Cluster', 
            y='Total', 
            palette='Spectral', 
            ax=ax_box
        )
        ax_box.set_title("Box Plot: Distribution of Total Spending by Cluster")
        st.pyplot(fig_box)
    else:
        st.warning("Cannot display box plot for clusters—missing 'Cluster' or 'Total' column.")

    # 6. Payment Method by Cluster (Catplot)
    if 'Payment' in data_segmented.columns and 'Cluster' in data_segmented.columns:
        st.subheader("Payment Method by Cluster")
        pay_data = data_segmented.dropna(subset=['Payment', 'Cluster'])
        
        g_pay = sns.catplot(
            data=pay_data,
            x='Payment',
            hue='Gender',  # or 'Customer type'
            col='Cluster',
            kind='count',
            col_wrap=3,
            height=4,
            aspect=1.2,
            sharex=False,
            sharey=False
        )
        g_pay.set_xticklabels(rotation=45)
        g_pay.set_titles("Cluster {col_name}")
        st.pyplot(g_pay.fig)
    else:
        st.warning("Cannot display Payment Method by Cluster—missing 'Payment' or 'Cluster' column.")



# Churn Prediction
st.header("Churn Prediction")

# Prepare data for churn prediction
X, y = prepare_churn_data(data_cleaned)

if X is not None and y is not None:
    churn_model = train_churn_model(X, y)
    if churn_model:
        st.success("Churn Prediction Model Trained Successfully.")
        
        # Predict churn probabilities
        data_cleaned['Churn Probability'] = churn_model.predict_proba(X)[:, 1]
        st.write(data_cleaned[['Invoice ID', 'Total', 'Quantity', 'Churn Probability']].head())

        # Visualization
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(data_cleaned, x='Churn Probability', nbins=20, color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig)
    else:
        st.error("Model training failed.")
else:
    st.warning("Not enough data available for churn prediction.")







# Enable CSV Download
enable_data_download(data_cleaned)

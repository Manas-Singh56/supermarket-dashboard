
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

st.title("Supermarket Sales Dashboard")
st.header("Initial Data Exploration & Cleaning")

# Use st.cache to avoid reloading data on every app rerun
@st.cache(allow_output_mutation=True)
def load_data():
    # Load the dataset from the data folder
    df = pd.read_csv("data/supermarket_sales.csv")
    return df

# Load the raw data
data = load_data()

# Display a preview of the raw data
st.subheader("Raw Data Preview")
st.write(data.head())

# Display information about the dataset
st.subheader("Data Information")
buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

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

st.write("Data cleaning is complete.")

# --- NEW CODE: EXPLORATORY DATA ANALYSIS ---
st.header("Exploratory Data Analysis")

# Sidebar for filter controls
st.sidebar.header("Filters")

# Assuming we have these columns in the dataset (adjust based on actual data)
if 'Product line' in data_cleaned.columns:
    product_categories = ['All'] + sorted(data_cleaned['Product line'].unique().tolist())
    selected_category = st.sidebar.selectbox("Select Product Category", product_categories)

if 'Customer type' in data_cleaned.columns:
    customer_types = ['All'] + sorted(data_cleaned['Customer type'].unique().tolist())
    selected_customer_type = st.sidebar.selectbox("Select Customer Type", customer_types)

if 'Gender' in data_cleaned.columns:
    genders = ['All'] + sorted(data_cleaned['Gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Select Gender", genders)

if 'Date' in data_cleaned.columns:
    # Create date range selector
    min_date = data_cleaned['Date'].min().date()
    max_date = data_cleaned['Date'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
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
if 'Product line' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales by Product Category")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    category_sales = filtered_data.groupby('Product line')['Total'].sum().sort_values(ascending=False)
    sns.barplot(x=category_sales.index, y=category_sales.values, palette='viridis', ax=ax)
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Total Sales')
    ax.set_title('Total Sales by Product Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Distribution of sales by payment method
if 'Payment' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales by Payment Method")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    payment_sales = filtered_data.groupby('Payment')['Total'].sum().sort_values(ascending=False)
    sns.barplot(x=payment_sales.index, y=payment_sales.values, palette='Set2', ax=ax)
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Total Sales')
    ax.set_title('Total Sales by Payment Method')
    plt.tight_layout()
    st.pyplot(fig)

# Distribution by customer demographics
if 'Gender' in filtered_data.columns and 'Customer type' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales by Customer Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        gender_sales = filtered_data.groupby('Gender')['Total'].sum()
        ax.pie(gender_sales, labels=gender_sales.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
        ax.set_title('Sales by Gender')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        customer_type_sales = filtered_data.groupby('Customer type')['Total'].sum()
        ax.pie(customer_type_sales, labels=customer_type_sales.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax.set_title('Sales by Customer Type')
        plt.tight_layout()
        st.pyplot(fig)

# Time-based analysis
if 'Date' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales Trends Over Time")
    
    # Daily sales trend
    daily_sales = filtered_data.groupby(filtered_data['Date'].dt.date)['Total'].sum().reset_index()
    daily_sales.columns = ['Date', 'Total Sales']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=daily_sales, x='Date', y='Total Sales', marker='o', ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Sales')
    ax.set_title('Daily Sales Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Hour-based analysis if available
if 'Time' in filtered_data.columns and 'Total' in filtered_data.columns:
    st.write("### Sales by Hour of Day")
    
    # Extract hour from time column if it's in string format
    if filtered_data['Time'].dtype == 'object':
        try:
            filtered_data['Hour'] = pd.to_datetime(filtered_data['Time']).dt.hour
        except:
            # If above fails, try to extract hour from time string directly
            filtered_data['Hour'] = filtered_data['Time'].str.split(':', expand=True)[0].astype(int)
    
    hourly_sales = filtered_data.groupby('Hour')['Total'].sum().reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=hourly_sales, x='Hour', y='Total', palette='rocket', ax=ax)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Total Sales')
    ax.set_title('Sales by Hour of Day')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h}:00" for h in range(24)])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Correlation analysis
st.subheader("Correlation Analysis")

# Select only numeric columns for correlation
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if len(numeric_cols) > 1:
    corr_matrix = filtered_data[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("""
    ### Correlation Analysis Insights:
    - Strong positive correlations indicate variables that increase together
    - Strong negative correlations indicate when one variable increases, the other decreases
    - Values close to zero indicate little to no relationship between variables
    """)

# Product-specific analysis
if 'Product line' in filtered_data.columns and 'Unit price' in filtered_data.columns and 'Quantity' in filtered_data.columns:
    st.subheader("Product Analysis")
    
    # Average unit price by product category
    avg_price = filtered_data.groupby('Product line')['Unit price'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_price.index, y=avg_price.values, palette='muted', ax=ax)
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Average Unit Price')
    ax.set_title('Average Unit Price by Product Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Total quantity sold by product category
    qty_sold = filtered_data.groupby('Product line')['Quantity'].sum().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=qty_sold.index, y=qty_sold.values, palette='Blues_d', ax=ax)
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Total Quantity Sold')
    ax.set_title('Quantity Sold by Product Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


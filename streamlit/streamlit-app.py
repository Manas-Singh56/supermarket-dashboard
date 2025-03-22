import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime
import streamlit_visualise

# Configure API endpoints
API_URL = "http://localhost:5000/api"

# Set page title and layout
st.set_page_config(page_title="Supermarket Sales Dashboard", layout="wide")
st.title("Supermarket Sales Dashboard")

# Function to fetch data from Flask API
@st.cache_resource
def load_data_from_api():
    try:
        response = requests.get(f"{API_URL}/data")
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            return data
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return pd.DataFrame()

# Function to fetch category data for filters
@st.cache_resource
def load_categories_from_api():
    try:
        response = requests.get(f"{API_URL}/data/categories")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching categories: {response.status_code}")
            return {}
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return {}

# Function to get filtered data based on selected filters
def get_filtered_data(filter_params):
    try:
        response = requests.post(f"{API_URL}/data/filtered", json=filter_params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            return data
        else:
            st.error(f"Error fetching filtered data: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return pd.DataFrame()

# Load data
data = load_data_from_api()
categories = load_categories_from_api()

if data.empty:
    st.error("Failed to load data. Please check if the Flask API is running.")
    st.stop()

# Sidebar for filter controls
st.sidebar.header("Filters")

# Set up filters based on API categories data
filter_params = {}

if 'productCategories' in categories:
    selected_category = st.sidebar.selectbox("Select Product Category", categories['productCategories'])
    filter_params['category'] = selected_category

if 'customerTypes' in categories:
    selected_customer_type = st.sidebar.selectbox("Select Customer Type", categories['customerTypes'])
    filter_params['customerType'] = selected_customer_type

if 'genders' in categories:
    selected_gender = st.sidebar.selectbox("Select Gender", categories['genders'])
    filter_params['gender'] = selected_gender

if 'dateRange' in categories:
    # Create date range selector
    min_date = datetime.strptime(categories['dateRange']['min'], '%Y-%m-%d').date()
    max_date = datetime.strptime(categories['dateRange']['max'], '%Y-%m-%d').date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    # Convert to string format for the API
    filter_params['dateRange'] = [date.strftime('%Y-%m-%d') for date in selected_date_range]

# Add apply button to fetch filtered data
if st.sidebar.button("Apply Filters"):
    filtered_data = get_filtered_data(filter_params)
else:
    # Initial data load
    filtered_data = data

# Display count of filtered records
st.subheader(f"Filtered Data: {len(filtered_data)} records")

# Display statistical summary of the data
with st.expander("Statistical Summary"):
    try:
        response = requests.get(f"{API_URL}/data/summary")
        if response.status_code == 200:
            summary_data = response.json()
            st.write(pd.DataFrame(summary_data))
        else:
            st.error("Failed to load summary statistics")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Data preview
with st.expander("Data Preview"):
    st.write(filtered_data.head(10))

# Now that we have the filtered data, we can pass it to the visualise.py functions
# The visualise module will use this data to render the visualizations
streamlit_visualise.filtered_data = filtered_data

# Show the visualization components from visualise.py
st.header("Visualization Dashboard")
streamlit_visualise.render_visualizations()

# Download button for filtered data
if not filtered_data.empty:
    csv_data = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(label='Download as CSV', data=csv_data, file_name='filtered_data.csv', mime='text/csv')
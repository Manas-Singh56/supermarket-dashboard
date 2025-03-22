import plotly.express as px
import streamlit as st
import pandas as pd

# Function to Plot Heatmap for Sales by Day and Hour
def plot_sales_heatmap(filtered_data):
    try:
        if filtered_data.empty:
            st.warning("No data available to plot the heatmap.")
            return

        # Ensure necessary columns exist
        required_columns = ['Date', 'Time', 'Total']
        if not all(col in filtered_data.columns for col in required_columns):
            st.warning(f"Missing one or more required columns: {', '.join(required_columns)}")
            return

        # Convert 'Date' to day name
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], errors='coerce')
        filtered_data['Day'] = pd.Categorical(
            filtered_data['Date'].dt.day_name(),
            categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            ordered=True
        )

        # Convert 'Time' to hour
        if 'Hour' not in filtered_data.columns:
            filtered_data['Hour'] = pd.to_numeric(filtered_data['Time'].str.split(':').str[0], errors='coerce')

        # Aggregate for Heatmap
        heatmap_data = filtered_data.groupby(['Day', 'Hour'])['Total'].sum().reset_index()

        # Plot Heatmap
        fig = px.density_heatmap(
            heatmap_data,
            x='Hour',
            y='Day',
            z='Total',
            color_continuous_scale='Viridis',
            category_orders={"Day": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        )
        fig.update_layout(title='Sales Heatmap by Day and Hour', xaxis_title='Hour', yaxis_title='Day')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error while generating heatmap: {e}")

# Function to Plot Sales by Product Category
def plot_category_sales(filtered_data):
    try:
        if filtered_data.empty:
            st.warning("No data available to plot product category sales.")
            return

        # Check for necessary columns
        if not all(col in filtered_data.columns for col in ['Product line', 'Total']):
            st.warning("Required columns missing for product category visualization.")
            return
        
        # Aggregate sales data by Product line
        st.write("### Interactive Sales by Product Category")
        category_sales = filtered_data.groupby('Product line')['Total'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=category_sales.index,
            y=category_sales.values,
            labels={'x': 'Product Category', 'y': 'Total Sales'},
            color=category_sales.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(title='Total Sales by Product Category')
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error while generating product category plot: {e}")

# Function to Enable Data Download
def enable_data_download(filtered_data):
    try:
        if filtered_data.empty:
            st.warning("No data available for download.")
            return
        
        csv_data = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download as CSV', data=csv_data, file_name='filtered_data.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Error during CSV export: {e}")

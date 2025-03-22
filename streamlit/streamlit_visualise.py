import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import json

# This variable will be set by app.py before calling functions in this module
filtered_data = None

def render_visualizations():
    """Main function to render all visualizations"""
    if filtered_data is None or filtered_data.empty:
        st.warning("No data available for visualization. Please adjust filters or check API connection.")
        return
    
    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4 = st.tabs(["Sales Distribution", "Time Analysis", "Customer Analysis", "Product Analysis"])
    
    with tab1:
        render_sales_distribution()
    
    with tab2:
        render_time_analysis()
    
    with tab3:
        render_customer_analysis()
    
    with tab4:
        render_product_analysis()

def render_sales_distribution():
    """Render sales distribution visualizations"""
    st.subheader("Sales Distribution Analysis")
    
    # Distribution of sales by product category
    if 'Product line' in filtered_data.columns and 'Total' in filtered_data.columns:
        st.write("### Sales by Product Category")
        
        # Use Plotly for interactive chart
        category_sales = filtered_data.groupby('Product line')['Total'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(category_sales, 
                    x='Product line', 
                    y='Total', 
                    color='Product line',
                    title='Total Sales by Product Category')
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution of sales by payment method
    if 'Payment' in filtered_data.columns and 'Total' in filtered_data.columns:
        st.write("### Sales by Payment Method")
        
        payment_sales = filtered_data.groupby('Payment')['Total'].sum().sort_values(ascending=False).reset_index()
        fig = px.pie(payment_sales, 
                     values='Total', 
                     names='Payment',
                     title='Sales by Payment Method')
        st.plotly_chart(fig, use_container_width=True)

def render_time_analysis():
    """Render time-based analysis visualizations"""
    st.subheader("Time Analysis")
    
    # Time-based analysis
    if 'Date' in filtered_data.columns and 'Total' in filtered_data.columns:
        st.write("### Sales Trends Over Time")
        
        # Daily sales trend
        daily_sales = filtered_data.groupby(filtered_data['Date'].dt.date)['Total'].sum().reset_index()
        fig = px.line(daily_sales, 
                      x='Date', 
                      y='Total',
                      markers=True,
                      title='Daily Sales Trend')
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap for Sales by Day and Hour
        if 'Time' in filtered_data.columns:
            st.write("### Sales Heatmap by Day and Hour")
            
            # Create day of week column
            temp_df = filtered_data.copy()
            temp_df['Day'] = pd.Categorical(
                temp_df['Date'].dt.day_name(),
                categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                ordered=True
            )
            
            # Extract hour from Time column
            if 'Hour' not in temp_df.columns:
                if temp_df['Time'].dtype == 'object':
                    temp_df['Hour'] = temp_df['Time'].str.split(':').str[0].astype(int)
            
            # Aggregate sales data by Day and Hour
            heatmap_data = temp_df.groupby(['Day', 'Hour'])['Total'].sum().reset_index()
            
            # Create a heatmap with correctly ordered days
            fig = px.density_heatmap(
                heatmap_data,
                x='Hour',
                y='Day',
                z='Total',
                color_continuous_scale='Viridis',
                title='Sales Heatmap by Day and Hour'
            )
            
            fig.update_layout(xaxis_title='Hour', yaxis_title='Day')
            st.plotly_chart(fig, use_container_width=True)

def render_customer_analysis():
    """Render customer-focused analysis visualizations"""
    st.subheader("Customer Analysis")
    
    # Create two columns for side by side charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Gender
        if 'Gender' in filtered_data.columns and 'Total' in filtered_data.columns:
            gender_sales = filtered_data.groupby('Gender')['Total'].sum().reset_index()
            fig = px.pie(gender_sales, 
                        values='Total', 
                        names='Gender',
                        title='Sales by Gender',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by Customer Type
        if 'Customer type' in filtered_data.columns and 'Total' in filtered_data.columns:
            customer_type_sales = filtered_data.groupby('Customer type')['Total'].sum().reset_index()
            fig = px.pie(customer_type_sales, 
                        values='Total', 
                        names='Customer type',
                        title='Sales by Customer Type',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
    
    # Average Rating by Customer Type and Gender if available
    if all(col in filtered_data.columns for col in ['Rating', 'Gender', 'Customer type']):
        st.write("### Average Rating Analysis")
        
        fig = px.box(filtered_data, 
                     x='Customer type', 
                     y='Rating', 
                     color='Gender',
                     title='Rating Distribution by Customer Type and Gender')
        st.plotly_chart(fig, use_container_width=True)

def render_product_analysis():
    """Render product-focused analysis visualizations"""
    st.subheader("Product Analysis")
    
    # Average unit price by product category
    if 'Product line' in filtered_data.columns and 'Unit price' in filtered_data.columns:
        st.write("### Average Unit Price by Product Category")
        
        avg_price = filtered_data.groupby('Product line')['Unit price'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_price, 
                     x='Product line', 
                     y='Unit price',
                     color='Product line',
                     title='Average Unit Price by Product Category')
        st.plotly_chart(fig, use_container_width=True)
    
    # Total quantity sold by product category
    if 'Product line' in filtered_data.columns and 'Quantity' in filtered_data.columns:
        st.write("### Quantity Sold by Product Category")
        
        qty_sold = filtered_data.groupby('Product line')['Quantity'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(qty_sold, 
                     x='Product line', 
                     y='Quantity',
                     color='Product line',
                     title='Total Quantity Sold by Product Category')
        st.plotly_chart(fig, use_container_width=True)
        
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Select only numeric columns for correlation
    numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = filtered_data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
        
        st.write("""
        ### Correlation Analysis Insights:
        - Strong positive correlations indicate variables that increase together
        - Strong negative correlations indicate when one variable increases, the other decreases
        - Values close to zero indicate little to no relationship between variables
        """)

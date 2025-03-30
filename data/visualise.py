import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Distribution of sales by product category

def sales_by_product_category(filtered_data):
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



# Sales Trends Over Time
def sales_by_Time(filtered_data):
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


# plot_sales_by_hour
def sales_by_hour(data):
    """Plots total sales distribution by hour of the day."""
    
    if 'Time' not in data.columns or 'Total' not in data.columns:
        st.warning("Time or Total column not found in dataset.")
        return
    
    # Extract hour from time column if it's in string format
    if data['Time'].dtype == 'object':
        try:
            data['Hour'] = pd.to_datetime(data['Time']).dt.hour
        except:
            # If conversion fails, extract hour from time string
            data['Hour'] = data['Time'].str.split(':', expand=True)[0].astype(int)
    
    # Group by hour and sum total sales
    hourly_sales = data.groupby('Hour')['Total'].sum().reset_index()

    # Create the visualization
    fig = px.bar(
        hourly_sales, 
        x='Hour', 
        y='Total', 
        text_auto=True,
        color='Total',
        color_continuous_scale='Viridis',
        labels={'Hour': 'Hour of Day', 'Total': 'Total Sales'},
        title='Sales by Hour of Day'
    )
    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    st.plotly_chart(fig)
    # Product-specific analysis
def product_specific_analysis(filtered_data):
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

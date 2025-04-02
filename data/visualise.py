import plotly.express as px
import streamlit as st
import pandas as pd
import seaborn as sns

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

        # Create Heatmap with better visuals
        fig = px.density_heatmap(
            heatmap_data, x='Hour', y='Day', z='Total',
            color_continuous_scale='Plasma',  # More vibrant colors
            category_orders={"Day": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']},
            title="Sales Heatmap by Day and Hour"
        )

        fig.update_layout(
            xaxis_title="Hour of the Day",
            yaxis_title="Day of the Week",
            font=dict(family="Arial", size=12),
            coloraxis_colorbar=dict(title="Total Sales"),
            template="plotly_white"
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error while generating heatmap: {e}")

# Distribution of sales by product category

def sales_by_product_category(filtered_data):
    if 'Product line' in filtered_data.columns and 'Total' in filtered_data.columns:
        st.write("### Sales by Product Category")

        category_sales = filtered_data.groupby('Product line')['Total'].sum().reset_index()

        fig = px.bar(category_sales, 
                     x='Product line', 
                     y='Total', 
                     color='Total',
                     text='Total',
                     color_continuous_scale='viridis')

        fig.update_layout(
            xaxis_title="Product Category",
            yaxis_title="Total Sales",
            title="Total Sales by Product Category",
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig)



# Sales Trends Over Time
import plotly.express as px

def sales_by_Time(filtered_data):
    if 'Date' in filtered_data.columns and 'Total' in filtered_data.columns:
        st.write("### Sales Trends Over Time")
    
        # Convert 'Date' column to datetime if not already
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    
        # Daily sales trend
        daily_sales = filtered_data.groupby(filtered_data['Date'].dt.date)['Total'].sum().reset_index()
        daily_sales.columns = ['Date', 'Total Sales']

        # Plotly line chart
        fig = px.line(daily_sales, x='Date', y='Total Sales', 
                      markers=True, title='Daily Sales Trend')
        fig.update_layout(xaxis_title='Date', yaxis_title='Total Sales', 
                          xaxis_tickangle=-45, template='plotly_dark')
        
        st.plotly_chart(fig)


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
import plotly.express as px

def product_specific_analysis(filtered_data):
    if 'Product line' in filtered_data.columns and 'Unit price' in filtered_data.columns and 'Quantity' in filtered_data.columns:
        st.subheader("Product Analysis")

        # Average unit price by product category
        avg_price = filtered_data.groupby('Product line')['Unit price'].mean().reset_index()

        fig1 = px.bar(avg_price, 
                      x='Product line', 
                      y='Unit price', 
                      color='Unit price',
                      text='Unit price',
                      title="Average Unit Price by Product Category",
                      color_continuous_scale='viridis')

        fig1.update_layout(xaxis_tickangle=-45, xaxis_title="Product Category", yaxis_title="Average Unit Price")
        st.plotly_chart(fig1)

        # Total quantity sold by product category
        qty_sold = filtered_data.groupby('Product line')['Quantity'].sum().reset_index()

        fig2 = px.bar(qty_sold, 
                      x='Product line', 
                      y='Quantity', 
                      color='Quantity',
                      text='Quantity',
                      title="Quantity Sold by Product Category",
                      color_continuous_scale='blues')

        fig2.update_layout(xaxis_tickangle=-45, xaxis_title="Product Category", yaxis_title="Total Quantity Sold")
        st.plotly_chart(fig2)

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
def plot_prophet_forecast(forecast, title="Sales Forecast with Prophet"):
    """
    Plot the Prophet forecast results.
    
    Parameters:
    -----------
    forecast : pandas.DataFrame
        DataFrame returned from Prophet's predict() method (merged with actual values if available),
        containing at least the columns:
            - 'ds' (dates)
            - 'yhat' (predicted values)
            - 'yhat_lower' (lower prediction bound)
            - 'yhat_upper' (upper prediction bound)
            - optionally, 'y' (actual values)
    title : str, default="Sales Forecast with Prophet"
        Title for the plot.
        
    Returns:
    --------
    fig : plotly.graph_objs._figure.Figure
        A Plotly Figure object.
    """
    # Create the main line plot for predicted sales
    fig = px.line(
        forecast, 
        x='ds', 
        y='yhat', 
        title=title, 
        labels={'ds': 'Date', 'yhat': 'Predicted Sales'}
    )
    
    # If actual sales are available, add them as markers
    if 'y' in forecast.columns:
        fig.add_scatter(
            x=forecast['ds'], 
            y=forecast['y'], 
            mode='markers', 
            name='Actual Sales',
            marker=dict(color='black', size=5)
        )
    
    # Add the lower and upper bounds for prediction intervals
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        line=dict(dash='dash', color='gray')
    )
    
    fig.add_scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(dash='dash', color='gray')
    )
    
    return fig

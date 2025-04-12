import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from feature import clean_data, add_features, customer_segmentation, prepare_churn_data, train_churn_model, predict_sales_with_prophet, prepare_sales_data_for_prophet
from visualise import plot_prophet_forecast, plot_sales_heatmap, plot_category_sales, enable_data_download, sales_by_hour, sales_by_Time, sales_by_product_category, product_specific_analysis

st.title("Supermarket Sales Dashboard")

# Add file uploader to sidebar
st.sidebar.header("Upload Data")
encodings = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]
selected_encoding = st.sidebar.selectbox("Select file encoding", encodings, index=0)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])


# Define function to check required columns
def check_required_columns(df):
    # Define required columns for different visualizations
    required_columns = {
        'Product line': ['Sales by Product Category', 'Category Sales Analysis'],
        'Payment': ['Payment Method Analysis'],
        'Gender': ['Demographic Analysis'],
        'Customer type': ['Customer Type Analysis'],
        'Date': ['Time-based Analysis', 'Sales Forecast', 'Sales Heatmap'],
        'Total': ['Sales Analysis', 'Correlation Analysis', 'Forecasting'],
        'Quantity': ['Product Analysis', 'Correlation Analysis'],
        'Invoice ID': ['Customer Segmentation', 'Churn Prediction']
    }
    
    missing_columns = []
    affected_visualizations = []
    
    # Check for missing columns
    for col, visualizations in required_columns.items():
        if col not in df.columns:
            missing_columns.append(col)
            affected_visualizations.extend(visualizations)
    
    # Display warnings for missing columns
    if missing_columns:
        st.warning("⚠️ Some columns are missing from your dataset:")
        for col in missing_columns:
            st.write(f"- Missing '{col}' column")
        
        st.warning("The following visualizations may not work properly:")
        for viz in set(affected_visualizations):  # Use set to remove duplicates
            st.write(f"- {viz}")
    
    return missing_columns

#load data

@st.cache_resource
def load_data(uploaded_file=None, encoding="utf-8"):
    try:
        if uploaded_file is not None:
            # Try to read the uploaded file with the specified encoding
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.sidebar.success(f"File successfully uploaded using {encoding} encoding!")
        else:
            # If no file is uploaded, load the default dataset
            df = pd.read_csv("D:\supermarket-dashboard\data\supermarket_sales.csv")
            st.sidebar.info("Using default dataset. Upload your own CSV for custom analysis.")
        
        return df
    except UnicodeDecodeError as e:
        # If the specified encoding fails, try other encodings
        if encoding == "utf-8":
            st.sidebar.warning(f"UTF-8 encoding failed. Trying latin1 encoding instead.")
            try:
                # Reset file pointer to beginning of file
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
                st.sidebar.success("File successfully uploaded using latin1 encoding!")
                return df
            except Exception as e2:
                st.sidebar.error(f"Failed with latin1 encoding too: {e2}")
        else:
            st.sidebar.error(f"Error reading the file with {encoding} encoding: {e}")
        
        # Fall back to the default dataset
        try:
            df = pd.read_csv("D:\supermarket-dashboard\data\supermarket_sales.csv")
            st.sidebar.info("Loading default dataset instead.")
            return df
        except Exception as e2:
            st.error(f"Could not load default dataset: {e2}")
            # Create an empty DataFrame with basic columns if all else fails
            return pd.DataFrame(columns=['Invoice ID', 'Date', 'Total', 'Quantity'])
    except Exception as e:
        st.sidebar.error(f"Error reading the file: {e}")
        st.sidebar.info("Loading default dataset instead.")
        # Fall back to the default dataset
        try:
            df = pd.read_csv("D:\supermarket-dashboard\data\supermarket_sales.csv")
            return df
        except Exception as e2:
            st.error(f"Could not load default dataset: {e2}")
            # Create an empty DataFrame with basic columns if all else fails
            return pd.DataFrame(columns=['Invoice ID', 'Date', 'Total', 'Quantity'])

# Load data based on whether a file was uploaded
data = load_data(uploaded_file, selected_encoding)
# Check for required columns after loading the data
missing_columns = check_required_columns(data)

# Display statistical summary of the data
st.header("Initial Data Exploration & Cleaning")
st.subheader("Statistical Summary")
try:
    st.write(data.describe())
except Exception as e:
    st.error(f"Could not generate statistical summary: {e}")

# Data Cleaning Section
st.header("Data Cleaning")
try:
    st.write("Removing duplicates and handling missing values...")
    
    # Remove duplicate rows
    data_cleaned = data.drop_duplicates()
    
    # Fill missing values using forward fill
    data_cleaned = data_cleaned.fillna(method='ffill')
    
    # Convert the 'Date' column to datetime if it exists
    if 'Date' in data_cleaned.columns:
        data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')
    
    st.subheader("Cleaned Data Preview")
    st.write(data_cleaned.head())
    
    # Only run customer segmentation if all required columns are present
    try:
        if all(col in data_cleaned.columns for col in ['Total', 'Quantity', 'Gender', 'Customer type']):
            data_segmented = customer_segmentation(data_cleaned, n_clusters=3)
            st.write("Data cleaning and customer segmentation complete.")
        else:
            st.warning("Cannot perform customer segmentation - required columns are missing.")
            data_segmented = data_cleaned.copy()  # Use cleaned data without segmentation
    except Exception as e:
        st.error(f"Customer segmentation failed: {e}")
        data_segmented = data_cleaned.copy()  # Use cleaned data without segmentation
        
except Exception as e:
    st.error(f"Data cleaning failed: {e}")
    data_cleaned = data.copy()  # Use the original data as fallback
    data_segmented = data.copy()

# --- EXPLORATORY DATA ANALYSIS ---
st.header("Exploratory Data Analysis")

# Sidebar for filter controls
st.sidebar.header("Filters")

# Setup filter options with proper error handling
try:
    # Product category filter
    if 'Product line' in data_cleaned.columns:
        product_categories = ['All'] + sorted(data_cleaned['Product line'].unique().tolist())
        selected_category = st.sidebar.selectbox("Select Product Category", product_categories, key="product_category_1")
    else:
        selected_category = 'All'
        
    # Customer type filter
    if 'Customer type' in data_cleaned.columns:
        customer_types = ['All'] + sorted(data_cleaned['Customer type'].unique().tolist())
        selected_customer_type = st.sidebar.selectbox("Select Customer Type", customer_types, key="customer_type_1")
    else:
        selected_customer_type = 'All'
        
    # Gender filter
    if 'Gender' in data_cleaned.columns:
        genders = ['All'] + sorted(data_cleaned['Gender'].unique().tolist())
        selected_gender = st.sidebar.selectbox("Select Gender", genders, key="gender_1")
    else:
        selected_gender = 'All'
        
    # Date range filter
    if 'Date' in data_cleaned.columns:
        try:
            min_date = data_cleaned['Date'].min().date()
            max_date = data_cleaned['Date'].max().date()
            selected_date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_range_1"
            )
        except Exception as e:
            st.sidebar.warning(f"Date filter not available: {e}")
            selected_date_range = []
    else:
        selected_date_range = []
        
except Exception as e:
    st.sidebar.error(f"Error setting up filters: {e}")
    selected_category = 'All'
    selected_customer_type = 'All'
    selected_gender = 'All'
    selected_date_range = []

# Filter data based on selections
try:
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
    
except Exception as e:
    st.error(f"Error filtering data: {e}")
    filtered_data = data_cleaned.copy()
    st.subheader(f"Showing all data: {len(filtered_data)} records")

# --- ANALYSIS BY DISTRIBUTION ---
st.subheader("Sales Distribution Analysis")

# Distribution of sales by product category
try:
    if 'Product line' in filtered_data.columns and 'Total' in filtered_data.columns:
        sales_by_product_category(filtered_data)
    else:
        st.warning("Cannot display Sales by Product Category - required columns 'Product line' and/or 'Total' are missing.")
except Exception as e:
    st.error(f"Error displaying Sales by Product Category: {e}")
    st.info("Try uploading a dataset with 'Product line' and 'Total' columns to view this visualization.")

# Distribution of sales by payment method
try:
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
    else:
        st.warning("Cannot display Sales by Payment Method - required columns 'Payment' and/or 'Total' are missing.")
except Exception as e:
    st.error(f"Error displaying Sales by Payment Method: {e}")

# Distribution by customer demographics
try:
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
                hole=0.3
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
                hole=0.3
            )
            fig_cust_type.update_traces(textposition='inside', textinfo='percent+label')
            fig_cust_type.update_layout(
                legend_title_text='Customer Type',
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_cust_type, use_container_width=True)
    else:
        st.warning("Cannot display Sales by Customer Demographics - required columns are missing.")
except Exception as e:
    st.error(f"Error displaying Sales by Customer Demographics: {e}")

# Time-based analysis
try:
    if 'Date' in filtered_data.columns and 'Total' in filtered_data.columns:
        sales_by_Time(filtered_data)
    else:
        st.warning("Cannot display Time-based analysis - required columns 'Date' and/or 'Total' are missing.")
except Exception as e:
    st.error(f"Error displaying Time-based analysis: {e}")

# Hour-based analysis
try:
    if 'Time' in filtered_data.columns or ('Date' in filtered_data.columns and filtered_data['Date'].dt.time.nunique() > 1):
        st.subheader("Sales by Hour of Day")
        sales_by_hour(filtered_data)
    else:
        st.warning("Cannot display Sales by Hour - required time data is missing.")
except Exception as e:
    st.error(f"Error displaying Sales by Hour: {e}")

# Correlation analysis
try:
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
    else:
        st.warning("Not enough numeric columns for correlation analysis.")
except Exception as e:
    st.error(f"Error generating correlation analysis: {e}")

# Product-specific analysis
try:
    if 'Product line' in filtered_data.columns and 'Total' in filtered_data.columns:
        product_specific_analysis(filtered_data)
    else:
        st.warning("Cannot display Product-specific analysis - required columns are missing.")
except Exception as e:
    st.error(f"Error displaying Product-specific analysis: {e}")

# Visualizations
try:
    if 'Date' in data_cleaned.columns and 'Total' in data_cleaned.columns:
        st.write("### Sales Heatmap by Day and Hour")
        plot_sales_heatmap(data_cleaned)
    else:
        st.warning("Cannot display Sales Heatmap - required columns 'Date' and/or 'Total' are missing.")
except Exception as e:
    st.error(f"Error displaying Sales Heatmap: {e}")

try:
    if 'Product line' in data_cleaned.columns and 'Total' in data_cleaned.columns:
        plot_category_sales(data_cleaned)
    else:
        st.warning("Cannot display Category Sales - required columns 'Product line' and/or 'Total' are missing.")
except Exception as e:
    st.error(f"Error displaying Category Sales: {e}")


# Forecasting
st.subheader("Sales Forecast")
try:
    if 'Date' in data_cleaned.columns and 'Total' in data_cleaned.columns:
        prophet_ready_df = prepare_sales_data_for_prophet(data_cleaned, date_column='Date', sales_column='Total')
        forecast_df, model = predict_sales_with_prophet(prophet_ready_df, periods=30)
        
        # Calculate reference statistics from data, safely handling edge cases
        avg_sales = prophet_ready_df['y'].mean()
        min_sales = prophet_ready_df['y'].min()
        
        # Create a sidebar section for forecast settings
        st.sidebar.header("Forecast Settings")
        
        # Set safe default values that won't cause widget errors
        min_threshold_value = max(1.0, float(min_sales * 0.5))  # Ensure minimum is at least 1.0
        default_threshold = max(min_threshold_value, round(avg_sales * 0.7, -2))  # 70% of average, at least min value
        max_threshold_value = max(default_threshold * 2, float(avg_sales * 1.5))  # Ensure max is higher than default
        
        # Allow user to set their own threshold with a sensible default value
        critical_threshold = st.sidebar.number_input(
            "Set critical sales threshold (INR)",
            min_value=min_threshold_value,
            max_value=max_threshold_value,
            value=default_threshold,
            step=100.0,
            help="Sales below this amount will trigger an alert"
        )
        
        # Plot the forecast
        fig = plot_prophet_forecast(forecast_df)
        st.plotly_chart(fig)
        
        # Add a horizontal line at the threshold value to the forecast chart
        threshold_fig = px.line(forecast_df, x='ds', y='yhat', 
                               title=f"Sales Forecast with ₹{critical_threshold:,.2f} Threshold")
        threshold_fig.add_hline(y=critical_threshold, line_width=2, line_dash="dash", line_color="red",
                              annotation_text=f"Critical Threshold: ₹{critical_threshold:,.2f}",
                              annotation_position="bottom right")
        threshold_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Sales (INR)",
            hovermode="x unified"
        )
        st.plotly_chart(threshold_fig)
        
        # Check if any forecasted days fall below the threshold
        below_threshold = forecast_df[forecast_df['yhat'] < critical_threshold]
        
        if not below_threshold.empty:
            # Find the first date when sales are predicted to fall below threshold
            first_critical_date = below_threshold.iloc[0]['ds'].strftime('%d %B, %Y')
            
            # Calculate how many days in the forecast period are below threshold
            below_threshold_count = len(below_threshold)
            below_threshold_percent = (below_threshold_count / len(forecast_df)) * 100
            
            # Display warning with the date
            st.warning(f"⚠️ Alert: Forecast predicts that sales may fall below your critical threshold of ₹{critical_threshold:,.2f} by {first_critical_date}. {below_threshold_count} days ({below_threshold_percent:.1f}% of forecast period) show potentially critical sales levels.")
            
            # Show the problematic period in a small table
            st.write("### Critical Sales Periods")
            critical_periods = below_threshold[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(7)
            critical_periods.columns = ['Date', 'Predicted Sales (INR)', 'Lower Bound (INR)', 'Upper Bound (INR)']
            critical_periods['Predicted Sales (INR)'] = critical_periods['Predicted Sales (INR)'].round(2)
            critical_periods['Lower Bound (INR)'] = critical_periods['Lower Bound (INR)'].round(2)
            critical_periods['Upper Bound (INR)'] = critical_periods['Upper Bound (INR)'].round(2)
            st.dataframe(critical_periods)
            
            # Add recommendations
            st.info("""
            **Recommendations:**
            - Consider running promotional campaigns around these dates
            - Analyze historical data to understand if this is a seasonal pattern
            - Plan inventory accordingly to minimize costs during low sales periods
            - Adjust marketing strategy for the affected product categories
            """)
            
            # Add download button for critical periods
            csv_critical = critical_periods.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Critical Periods CSV",
                data=csv_critical,
                file_name="critical_sales_periods.csv",
                mime="text/csv",
            )
        else:
            st.success(f"✅ Sales forecast looks healthy! No periods below your critical threshold of ₹{critical_threshold:,.2f} detected in the next 30 days.")
            
            # Show the minimum forecasted value for reference
            min_forecast = forecast_df['yhat'].min()
            st.info(f"The lowest forecasted sales value is ₹{min_forecast:,.2f}, which is above your threshold.")
    else:
        st.warning("Cannot display Sales Forecast - required columns 'Date' and/or 'Total' are missing.")
except Exception as e:
    st.error(f"Error generating Sales Forecast: {e}")
    st.info("Sales forecasting requires a dataset with 'Date' and 'Total' columns and sufficient historical data.")
    
    # Add debug information if there's an error
    with st.expander("Debug Information"):
        if 'Date' in data_cleaned.columns and 'Total' in data_cleaned.columns:
            st.write("Data sample:")
            st.write(data_cleaned[['Date', 'Total']].head())
            st.write(f"Min Total: {data_cleaned['Total'].min()}")
            st.write(f"Avg Total: {data_cleaned['Total'].mean()}")
            st.write(f"Error details: {str(e)}")

# Customer Segmentation
st.header("Customer Segmentation Analysis")

# Use Streamlit tabs to organize your segmentation outputs
tab1, tab2 = st.tabs(["Cluster Overview", "Detailed Analysis"])

try:
    with tab1:
        # 1. Table preview
        if 'Cluster' in data_segmented.columns:
            st.subheader("Segmented Data Preview")
            preview_columns = [col for col in ['Invoice ID', 'Gender', 'Customer type', 
                                            'Product line', 'Total', 'Quantity', 'Cluster'] 
                            if col in data_segmented.columns]
            
            if preview_columns:
                st.write(data_segmented[preview_columns].head(10))
            else:
                st.warning("No relevant columns for segmentation preview.")

            # 2. Scatter plot (Quantity vs. Total)
            if all(col in data_segmented.columns for col in ['Quantity', 'Total', 'Cluster']):
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
            else:
                st.warning("Cannot display Cluster Scatter Plot - required columns are missing.")
            
            # 3. Cluster distribution
            if 'Cluster' in data_segmented.columns:
                cluster_counts = data_segmented['Cluster'].value_counts()
                st.write("### Cluster Distribution")
                for cluster_label, count in cluster_counts.items():
                    st.write(f"Cluster {cluster_label}: {count} rows")
            else:
                st.warning("Cluster column not found for distribution analysis.")
        else:
            st.warning("Customer Segmentation could not be performed or 'Cluster' column not found.")

    with tab2:
        # 4. Catplot: Demographic & Product Preferences by Cluster
        if all(col in data_segmented.columns for col in ['Cluster', 'Gender', 'Product line']):
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
        if all(col in data_segmented.columns for col in ['Cluster', 'Total']):
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
            st.warning("Cannot display box plot for clusters - missing 'Cluster' or 'Total' column.")

        # 6. Payment Method by Cluster (Catplot)
        if all(col in data_segmented.columns for col in ['Payment', 'Cluster', 'Gender']):
            st.subheader("Payment Method by Cluster")
            pay_data = data_segmented.dropna(subset=['Payment', 'Cluster', 'Gender'])
            
            g_pay = sns.catplot(
                data=pay_data,
                x='Payment',
                hue='Gender',
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
            st.warning("Cannot display Payment Method by Cluster - required columns are missing.")
except Exception as e:
    st.error(f"Error generating Customer Segmentation Analysis: {e}")

# Churn Prediction
st.header("Churn Prediction")

try:
    # Prepare data for churn prediction
    if all(col in data_cleaned.columns for col in ['Total', 'Quantity']):
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
    else:
        st.warning("Cannot perform Churn Prediction - required columns are missing.")
except Exception as e:
    st.error(f"Error during Churn Prediction: {e}")

# Enable CSV Download
try:
    enable_data_download(data_cleaned)
except Exception as e:
    st.error(f"Error enabling data download: {e}")
    
    # Fallback download option
    st.write("### Download Cleaned Data")
    csv = data_cleaned.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="cleaned_supermarket_data.csv",
        mime="text/csv"
    )

import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
import numpy as np

# NEW IMPORTS FOR CUSTOMER SEGMENTATION
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Data Cleaning
def clean_data(df):
    try:
        logging.info(f"Initial Data Shape: {df.shape}")
        
        # Remove Duplicates
        df = df.drop_duplicates().fillna(method='ffill')
        logging.info(f"After Removing Duplicates: {df.shape}")
        
        # Convert Date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Day'] = df['Date'].dt.day_name()

        # Convert Time column to Hour if available
        if 'Time' in df.columns and df['Time'].dtype == 'object':
            df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour

        return df
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        return pd.DataFrame()

# Feature Engineering
def add_features(df):
    try:
        required_columns = ['Total', 'Quantity', 'Unit price']
        
        # Check for required columns
        if all(col in df.columns for col in required_columns):
            df['Profit'] = df['Total'] - (df['Quantity'] * df['Unit price'])
            df['Profit Margin (%)'] = (df['Profit'] / df['Total']) * 100
            logging.info("Profit and Profit Margin calculated.")
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            logging.warning(f"Missing columns for feature engineering: {missing_cols}")
        
        return df
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        return df

# Sales Forecasting
def sales_forecast(df):
    try:
        if 'Date' in df.columns and 'Total' in df.columns:
            df = df.dropna(subset=['Date', 'Total'])
            df['Date_ordinal'] = df['Date'].apply(lambda x: x.toordinal())
            
            X = df[['Date_ordinal']]
            y = df['Total']
            
            # Train Model
            model = LinearRegression().fit(X, y)
            logging.info("Sales Forecast Model Trained Successfully.")

            # Predict for Next 30 Days
            future_dates = pd.date_range(df['Date'].max() + pd.Timedelta(days=1), periods=30)
            future_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
            predictions = model.predict(future_ordinal)

            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predictions})
            return forecast_df
        else:
            logging.warning("Date or Total column missing. Forecasting skipped.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error in sales forecasting: {e}")
        return pd.DataFrame()

# NEW FUNCTION: CUSTOMER SEGMENTATION
def customer_segmentation(df, n_clusters=3):
    """
    Perform K-Means clustering on key purchasing features to segment customers.
    Adds a 'Cluster' column to the DataFrame indicating each row's cluster assignment.
    
    :param df: DataFrame containing supermarket sales data
    :param n_clusters: Number of clusters to form (default is 3)
    :return: DataFrame with a new 'Cluster' column
    """
    try:
        # Ensure required numeric columns exist
        required_cols = ['Total', 'Quantity', 'Unit price']
        if not all(col in df.columns for col in required_cols):
            logging.warning("Missing columns for segmentation. Segmentation skipped.")
            return df
        
        # Create a subset with only the required columns
        segmentation_df = df[required_cols].copy()
        
        # Handle missing values (if any)
        segmentation_df = segmentation_df.dropna()
        
        # Scale the features for better clustering performance
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(segmentation_df)
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add the cluster labels back to the original DataFrame
        # We align on the index because we dropped some rows for NA
        df.loc[segmentation_df.index, 'Cluster'] = cluster_labels
        
        logging.info(f"Customer Segmentation done with {n_clusters} clusters.")
        return df
    except Exception as e:
        logging.error(f"Error in customer segmentation: {e}")
        return df
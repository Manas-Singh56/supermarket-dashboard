
import pandas as pd
import logging
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#Prophet

def predict_sales_with_prophet(sales_data, periods=30, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
    """
    Predict future sales using Facebook Prophet.
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        DataFrame containing date column 'ds' and sales column 'y'
        (if your columns have different names, rename them before passing)
    periods : int, default=30
        Number of periods to forecast into the future
    yearly_seasonality : bool or int, default=True
        Whether to include yearly seasonality
    weekly_seasonality : bool or int, default=True
        Whether to include weekly seasonality
    daily_seasonality : bool or int, default=False
        Whether to include daily seasonality
        
    Returns:
    --------
    forecast : pandas.DataFrame
        DataFrame with the original data and forecast including:
        - ds: dates
        - y: actual values (where available)
        - yhat: predicted values
        - yhat_lower: lower bound of prediction interval
        - yhat_upper: upper bound of prediction interval
    model : Prophet model
        Trained Prophet model
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = sales_data.copy()
    
    # Ensure the data is properly formatted for Prophet
    # Prophet requires columns named 'ds' and 'y'
    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("DataFrame must contain 'ds' (date) and 'y' (sales) columns")
    
    # Initialize the Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    
    # Add additional regressors if needed
    
    # Fit the model to the data
    model.fit(df)
    
    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=periods)
    
    # Generate the forecast
    forecast = model.predict(future)
    
    # Merge with actual values
    forecast = pd.merge(forecast, df, on='ds', how='left')
    
    return forecast, model

def prepare_sales_data_for_prophet(df, date_column, sales_column):
    """
    Prepare sales data for Prophet by renaming columns to 'ds' and 'y'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing sales data
    date_column : str
        Name of the column containing dates
    sales_column : str
        Name of the column containing sales values
        
    Returns:
    --------
    prophet_df : pandas.DataFrame
        DataFrame with columns renamed for Prophet
    """
    prophet_df = df.copy()
    prophet_df = prophet_df.rename(columns={date_column: 'ds', sales_column: 'y'})
    
    # Ensure date column is datetime type
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    return prophet_df



#churn prediction

# Prepare data for churn prediction
def prepare_churn_data(df):
    try:
        if 'Date' not in df.columns or 'Total' not in df.columns or 'Quantity' not in df.columns:
            st.error("Missing required columns for churn prediction: 'Date', 'Total', 'Quantity'")
            return None, None

        # Ensure Date is in datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Remove invalid dates
        df = df.dropna(subset=['Date'])

        # Calculate features
        df['Days Since Last Purchase'] = (pd.to_datetime('today') - df['Date']).dt.days
        df['Average Purchase Value'] = df['Total'] / np.where(df['Quantity'] == 0, 1, df['Quantity'])

        # Handle missing or infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()

        # Check for the Churn column
        if 'Churn' not in df.columns:
            st.error("Missing 'Churn' column for prediction.")
            return None, None
        
        # Features and target
        X = df[['Total', 'Quantity', 'Average Purchase Value', 'Days Since Last Purchase']]
        y = df['Churn'].astype(int)
        return X, y
    except Exception as e:
        logging.error(f"Error preparing churn data: {e}")
        return None, None


def train_churn_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Churn Prediction Model Accuracy: {accuracy * 100:.2f}%")
        
        return model
    except Exception as e:
        st.error(f"Error training churn model: {e}")
        return None  

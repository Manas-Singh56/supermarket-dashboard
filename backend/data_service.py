import pandas as pd
import numpy as np
from flask import send_file
import io
import os
import json
from flask import jsonify
from datetime import datetime

class DataService:
    def __init__(self):
        """Initialize the data service and load the dataset"""
        self.data_path = '../data/supermarket_sales.csv'
        self.load_data()

    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Clean the data
            self.data = self.data.drop_duplicates()
            self.data = self.data.fillna(method='ffill')
            
            # Convert Date column to datetime
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
                
            # Extract hour from Time column if it exists
            if 'Time' in self.data.columns and self.data['Time'].dtype == 'object':
                try:
                    self.data['Hour'] = pd.to_datetime(self.data['Time']).dt.hour
                except:
                    # If above fails, try to extract hour from time string directly
                    self.data['Hour'] = self.data['Time'].str.split(':', expand=True)[0].astype(int)
                    
            print(f"Data loaded successfully: {len(self.data)} records")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Create empty DataFrame with same structure in case of error
            self.data = pd.DataFrame()
            return False

    def get_data(self):
        """Return the entire dataset as a list of dictionaries"""
        return self.data.to_dict(orient='records')

    def filter_data(self, filter_params):
        """Filter data based on provided parameters"""
        filtered_data = self.data.copy()
        
        # Apply category filter
        if 'category' in filter_params and filter_params['category'] != 'All' and 'Product line' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Product line'] == filter_params['category']]
            
        # Apply customer type filter
        if 'customerType' in filter_params and filter_params['customerType'] != 'All' and 'Customer type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Customer type'] == filter_params['customerType']]
            
        # Apply gender filter
        if 'gender' in filter_params and filter_params['gender'] != 'All' and 'Gender' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['Gender'] == filter_params['gender']]
            
        # Apply date range filter
        if 'dateRange' in filter_params and 'Date' in filtered_data.columns:
            start_date = datetime.strptime(filter_params['dateRange'][0], '%Y-%m-%d')
            end_date = datetime.strptime(filter_params['dateRange'][1], '%Y-%m-%d')
            filtered_data = filtered_data[(filtered_data['Date'].dt.date >= start_date.date()) & 
                                         (filtered_data['Date'].dt.date <= end_date.date())]
            
        return filtered_data.to_dict(orient='records')

    def get_summary(self):
        """Return statistical summary of the dataset"""
        summary = self.data.describe().to_dict()
        return summary

    def get_categories(self):
        """Return unique values for categorical columns"""
        categories = {}
        
        if 'Product line' in self.data.columns:
            categories['productCategories'] = ['All'] + sorted(self.data['Product line'].unique().tolist())
            
        if 'Customer type' in self.data.columns:
            categories['customerTypes'] = ['All'] + sorted(self.data['Customer type'].unique().tolist())
            
        if 'Gender' in self.data.columns:
            categories['genders'] = ['All'] + sorted(self.data['Gender'].unique().tolist())
            
        if 'Payment' in self.data.columns:
            categories['paymentMethods'] = sorted(self.data['Payment'].unique().tolist())
            
        if 'Date' in self.data.columns:
            categories['dateRange'] = {
                'min': self.data['Date'].min().strftime('%Y-%m-%d'),
                'max': self.data['Date'].max().strftime('%Y-%m-%d')
            }
            
        return categories

    def get_time_series(self, params):
        """Return time series data based on parameters"""
        filtered_data = self.filter_data(params)
        filtered_df = pd.DataFrame(filtered_data)
        
        if filtered_df.empty or 'Date' not in filtered_df.columns:
            return []
        
        # Group by date and calculate total sales
        daily_sales = filtered_df.groupby(filtered_df['Date'].dt.date)['Total'].sum().reset_index()
        daily_sales.columns = ['Date', 'Total Sales']
        
        # Convert dates to strings for JSON serialization
        daily_sales['Date'] = daily_sales['Date'].astype(str)
        
        return daily_sales.to_dict(orient='records')

    def get_correlation(self):
        """Return correlation matrix of numeric columns"""
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if len(numeric_cols) <= 1:
            return {}
            
        corr_matrix = self.data[numeric_cols].corr().round(2)
        
        # Convert correlation matrix to a format suitable for heatmap visualization
        corr_data = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                corr_data.append({
                    'x': col1,
                    'y': col2,
                    'value': corr_matrix.iloc[i, j]
                })
                
        return {
            'matrix': corr_data,
            'columns': numeric_cols
        }

    def get_product_analysis(self, params):
        """Return product-specific analysis"""
        filtered_data = pd.DataFrame(self.filter_data(params))
        
        if filtered_data.empty:
            return {}
            
        result = {}
        
        # Average unit price by product category
        if 'Product line' in filtered_data.columns and 'Unit price' in filtered_data.columns:
            avg_price = filtered_data.groupby('Product line')['Unit price'].mean().reset_index()
            avg_price.columns = ['Product Category', 'Average Unit Price']
            result['avgPrice'] = avg_price.to_dict(orient='records')
            
        # Total quantity sold by product category
        if 'Product line' in filtered_data.columns and 'Quantity' in filtered_data.columns:
            qty_sold = filtered_data.groupby('Product line')['Quantity'].sum().reset_index()
            qty_sold.columns = ['Product Category', 'Total Quantity']
            result['qtySold'] = qty_sold.to_dict(orient='records')
            
        # Total sales by product category
        if 'Product line' in filtered_data.columns and 'Total' in filtered_data.columns:
            total_sales = filtered_data.groupby('Product line')['Total'].sum().reset_index()
            total_sales.columns = ['Product Category', 'Total Sales']
            result['totalSales'] = total_sales.to_dict(orient='records')
            
        # Sales by payment method
        if 'Payment' in filtered_data.columns and 'Total' in filtered_data.columns:
            payment_sales = filtered_data.groupby('Payment')['Total'].sum().reset_index()
            payment_sales.columns = ['Payment Method', 'Total Sales']
            result['paymentSales'] = payment_sales.to_dict(orient='records')
            
        # Sales by customer demographics
        if 'Gender' in filtered_data.columns and 'Total' in filtered_data.columns:
            gender_sales = filtered_data.groupby('Gender')['Total'].sum().reset_index()
            gender_sales.columns = ['Gender', 'Total Sales']
            result['genderSales'] = gender_sales.to_dict(orient='records')
            
        if 'Customer type' in filtered_data.columns and 'Total' in filtered_data.columns:
            customer_type_sales = filtered_data.groupby('Customer type')['Total'].sum().reset_index()
            customer_type_sales.columns = ['Customer Type', 'Total Sales']
            result['customerTypeSales'] = customer_type_sales.to_dict(orient='records')
            
        return result

    def generate_csv(self, filter_params):
        """Generate CSV file from filtered data"""
        filtered_data = pd.DataFrame(self.filter_data(filter_params))
        
        if filtered_data.empty:
            return jsonify({'error': 'No data to download'})
            
        # Create a CSV string
        csv_data = filtered_data.to_csv(index=False)
        
        # Create a BytesIO object
        buffer = io.BytesIO()
        buffer.write(csv_data.encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='filtered_data.csv'
        )
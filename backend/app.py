from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import json
from data_service import DataService

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Initialize data service
data_service = DataService()

@app.route('/api/data', methods=['GET'])
def get_data():
    """Return the full dataset"""
    return jsonify(data_service.get_data())

@app.route('/api/data/filtered', methods=['POST'])
def get_filtered_data():
    """Return filtered data based on criteria"""
    filter_params = request.json
    return jsonify(data_service.filter_data(filter_params))

@app.route('/api/data/summary', methods=['GET'])
def get_summary():
    """Return statistical summary of the data"""
    return jsonify(data_service.get_summary())

@app.route('/api/data/categories', methods=['GET'])
def get_categories():
    """Return unique categories for filtering"""
    return jsonify(data_service.get_categories())

@app.route('/api/data/time_series', methods=['POST'])
def get_time_series():
    """Return time series data based on parameters"""
    params = request.json
    return jsonify(data_service.get_time_series(params))

@app.route('/api/data/correlation', methods=['GET'])
def get_correlation():
    """Return correlation matrix of numeric columns"""
    return jsonify(data_service.get_correlation())

@app.route('/api/data/product_analysis', methods=['POST'])
def get_product_analysis():
    """Return product-specific analysis"""
    params = request.json
    return jsonify(data_service.get_product_analysis(params))

@app.route('/api/data/download', methods=['POST'])
def download_data():
    """Generate and return filtered data as CSV"""
    filter_params = request.json
    return data_service.generate_csv(filter_params)

# Serve React static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
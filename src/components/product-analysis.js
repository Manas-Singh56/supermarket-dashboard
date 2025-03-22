import React, { useState, useEffect } from 'react';
import apiService from '../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ProductAnalysis = ({ filterParams }) => {
  const [loading, setLoading] = useState(true);
  const [productData, setProductData] = useState({});
  
  useEffect(() => {
    const fetchProductData = async () => {
      setLoading(true);
      try {
        const data = await apiService.getProductAnalysis(filterParams);
        setProductData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching product data:', error);
        setLoading(false);
      }
    };
    
    fetchProductData();
  }, [filterParams]);
  
  if (loading) {
    return <div className="loading">Loading product analysis...</div>;
  }
  
  return (
    <div className="product-analysis">
      <h2>Product Analysis</h2>
      
      {productData.totalSales && (
        <div className="card">
          <h3 className="card-title">Total Sales by Product Category</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={productData.totalSales} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Product Category" angle={-45} textAnchor="end" />
              <YAxis />
              <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
              <Legend />
              <Bar dataKey="Total Sales" fill="#0088FE" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {productData.qtySold && (
        <div className="card">
          <h3 className="card-title">Quantity Sold by Product Category</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={productData.qtySold} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Product Category" angle={-45} textAnchor="end" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Total Quantity" fill="#00C49F" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {productData.avgPrice && (
        <div className="card">
          <h3 className="card-title">Average Unit Price by Product Category</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={productData.avgPrice} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Product Category" angle={-45} textAnchor="end" />
              <YAxis />
              <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Average Unit Price']} />
              <Legend />
              <Bar dataKey="Average Unit Price" fill="#FFBB28" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {productData.paymentSales && (
        <div className="card">
          <h3 className="card-title">Sales by Payment Method</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={productData.paymentSales} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Payment Method" />
              <YAxis />
              <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
              <Legend />
              <Bar dataKey="Total Sales" fill="#FF8042" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      
      {!productData.totalSales && !productData.qtySold && !productData.avgPrice && (
        <div className="no-data-message">
          <p>No product data available for the selected filters.</p>
        </div>
      )}
    </div>
  );
};

export default ProductAnalysis;

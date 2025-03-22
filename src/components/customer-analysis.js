import React, { useState, useEffect } from 'react';
import apiService from '../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const CustomerAnalysis = ({ filterParams }) => {
  const [loading, setLoading] = useState(true);
  const [customerData, setCustomerData] = useState({});
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d'];
  
  useEffect(() => {
    const fetchCustomerData = async () => {
      setLoading(true);
      try {
        const data = await apiService.getProductAnalysis(filterParams);
        setCustomerData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching customer data:', error);
        setLoading(false);
      }
    };
    
    fetchCustomerData();
  }, [filterParams]);
  
  if (loading) {
    return <div className="loading">Loading customer analysis...</div>;
  }
  
  return (
    <div className="customer-analysis">
      <h2>Customer Analysis</h2>
      
      <div className="charts-container">
        {customerData.genderSales && (
          <div className="card">
            <h3 className="card-title">Sales by Gender</h3>
            <div className="chart-row">
              <ResponsiveContainer width="48%" height={300}>
                <PieChart>
                  <Pie
                    data={customerData.genderSales}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="Total Sales"
                    nameKey="Gender"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {customerData.genderSales.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
                </PieChart>
              </ResponsiveContainer>
              
              <ResponsiveContainer width="48%" height={300}>
                <BarChart data={customerData.genderSales} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Gender" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
                  <Legend />
                  <Bar dataKey="Total Sales" fill="#0088FE" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        
        {customerData.customerTypeSales && (
          <div className="card">
            <h3 className="card-title">Sales by Customer Type</h3>
            <div className="chart-row">
              <ResponsiveContainer width="48%" height={300}>
                <PieChart>
                  <Pie
                    data={customerData.customerTypeSales}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="Total Sales"
                    nameKey="Customer Type"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {customerData.customerTypeSales.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
                </PieChart>
              </ResponsiveContainer>
              
              <ResponsiveContainer width="48%" height={300}>
                <BarChart data={customerData.customerTypeSales} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="Customer Type" />
                  <YAxis />
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
                  <Legend />
                  <Bar dataKey="Total Sales" fill="#00C49F" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
      
      {!customerData.genderSales && !customerData.customerTypeSales && (
        <div className="no-data-message">
          <p>No customer data available for the selected filters.</p>
        </div>
      )}
    </div>
  );
};

export default CustomerAnalysis;

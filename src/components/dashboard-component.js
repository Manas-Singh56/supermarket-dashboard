import React, { useState, useEffect } from 'react';
import apiService from '../api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const Dashboard = ({ filterParams }) => {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState({});
  const [productAnalysis, setProductAnalysis] = useState({});
  const [recordCount, setRecordCount] = useState(0);
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d'];

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoading(true);
      try {
        // Fetch data summary
        const summaryData = await apiService.getSummary();
        setSummary(summaryData);
        
        // Fetch product analysis
        const productData = await apiService.getProductAnalysis(filterParams);
        setProductAnalysis(productData);
        
        // Fetch filtered data to get record count
        const filteredData = await apiService.getFilteredData(filterParams);
        setRecordCount(filteredData.length);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, [filterParams]);
  
  if (loading) {
    return <div className="loading">Loading dashboard data...</div>;
  }
  
  return (
    <div className="dashboard">
      <h2>Dashboard Overview</h2>
      <p>Showing data for {recordCount} records based on current filter selections</p>
      
      <div className="summary-stats">
        {summary.Total && (
          <>
            <div className="stat-card">
              <div className="stat-value">${summary.Total.mean?.toFixed(2)}</div>
              <div className="stat-label">Average Sale</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">${summary.Total.max?.toFixed(2)}</div>
              <div className="stat-label">Highest Sale</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">${summary.Total.sum?.toFixed(2) || 'N/A'}</div>
              <div className="stat-label">Total Revenue</div>
            </div>
            {summary.Quantity && (
              <div className="stat-card">
                <div className="stat-value">{summary.Quantity.sum?.toFixed(0) || 'N/A'}</div>
                <div className="stat-label">Items Sold</div>
              </div>
            )}
          </>
        )}
      </div>
      
      {productAnalysis.totalSales && (
        <div className="card">
          <h3 className="card-title">Sales by Product Category</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={productAnalysis.totalSales} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
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
      
      <div className="chart-grid">
        {productAnalysis.paymentSales && (
          <div className="card">
            <h3 className="card-title">Sales by Payment Method</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={productAnalysis.paymentSales}
                  cx="50%"
                  cy="50%"
                  labelLine={true}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="Total Sales"
                  nameKey="Payment Method"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {productAnalysis.paymentSales.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {productAnalysis.customerTypeSales && productAnalysis.genderSales && (
          <div className="card">
            <h3 className="card-title">Customer Demographics</h3>
            <div className="chart-row">
              <ResponsiveContainer width="47%" height={300}>
                <PieChart>
                  <Pie
                    data={productAnalysis.customerTypeSales}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="Total Sales"
                    nameKey="Customer Type"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {productAnalysis.customerTypeSales.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
                </PieChart>
              </ResponsiveContainer>
              
              <ResponsiveContainer width="47%" height={300}>
                <PieChart>
                  <Pie
                    data={productAnalysis.genderSales}
                    cx="50%"
                    cy="50%"
                    labelLine={true}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="Total Sales"
                    nameKey="Gender"
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  >
                    {productAnalysis.genderSales.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

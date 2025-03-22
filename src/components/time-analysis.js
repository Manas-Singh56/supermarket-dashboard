import React, { useState, useEffect } from 'react';
import apiService from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const TimeAnalysis = ({ filterParams }) => {
  const [loading, setLoading] = useState(true);
  const [timeSeriesData, setTimeSeriesData] = useState([]);
  
  useEffect(() => {
    const fetchTimeSeriesData = async () => {
      setLoading(true);
      try {
        const data = await apiService.getTimeSeriesData(filterParams);
        setTimeSeriesData(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching time series data:', error);
        setLoading(false);
      }
    };
    
    fetchTimeSeriesData();
  }, [filterParams]);
  
  if (loading) {
    return <div className="loading">Loading time analysis...</div>;
  }
  
  return (
    <div className="time-analysis">
      <h2>Time Analysis</h2>
      
      {timeSeriesData && timeSeriesData.length > 0 ? (
        <div className="card">
          <h3 className="card-title">Daily Sales Trend</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={timeSeriesData}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="Date" 
                angle={-45} 
                textAnchor="end"
                tickFormatter={(value) => new Date(value).toLocaleDateString()}
              />
              <YAxis />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleDateString()}
                formatter={(value) => [`$${value.toFixed(2)}`, 'Total Sales']}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="Total Sales" 
                stroke="#0088FE" 
                activeDot={{ r: 8 }} 
                strokeWidth={2} 
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="no-data-message">
          <p>No time series data available for the selected filters.</p>
        </div>
      )}
      
      <div className="card">
        <h3 className="card-title">Time Analysis Insights</h3>
        <div className="insights">
          {timeSeriesData && timeSeriesData.length > 0 ? (
            <>
              <p>
                <strong>Date Range:</strong> {new Date(timeSeriesData[0].Date).toLocaleDateString()} to {new Date(timeSeriesData[timeSeriesData.length - 1].Date).toLocaleDateString()}
              </p>
              <p>
                <strong>Total Days:</strong> {timeSeriesData.length}
              </p>
              <p>
                <strong>Average Daily Sales:</strong> ${(timeSeriesData.reduce((sum, day) => sum + day['Total Sales'], 0) / timeSeriesData.length).toFixed(2)}
              </p>
              <p>
                <strong>Highest Sales Day:</strong> {new Date(timeSeriesData.reduce((max, day) => day['Total Sales'] > max.sales ? { date: day.Date, sales: day['Total Sales'] } : max, { date: '', sales: 0 }).date).toLocaleDateString()} (${timeSeriesData.reduce((max, day) => day['Total Sales'] > max.sales ? { date: day.Date, sales: day['Total Sales'] } : max, { date: '', sales: 0 }).sales.toFixed(2)})
              </p>
              <p>
                <strong>Lowest Sales Day:</strong> {new Date(timeSeriesData.reduce((min, day) => day['Total Sales'] < min.sales ? { date: day.Date, sales: day['Total Sales'] } : min, { date: '', sales: Number.MAX_VALUE }).date).toLocaleDateString()} (${timeSeriesData.reduce((min, day) => day['Total Sales'] < min.sales ? { date: day.Date, sales: day['Total Sales'] } : min, { date: '', sales: Number.MAX_VALUE }).sales.toFixed(2)})
              </p>
            </>
          ) : (
            <p>No insights available due to lack of data.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default TimeAnalysis;

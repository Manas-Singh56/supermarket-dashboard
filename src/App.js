import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import apiService from './api';

// Import components
import Dashboard from './components/dashboard-component';
import DataExplorer from './components/data-explorer';
import FilterPanel from './components/filter-panel';
import ProductAnalysis from './components/product-analysis';
import CustomerAnalysis from './components/customer-analysis';
import TimeAnalysis from './components/time-analysis';

function App() {
  const [categories, setCategories] = useState({
    productCategories: ['All'],
    customerTypes: ['All'],
    genders: ['All'],
    dateRange: { min: '', max: '' }
  });
  
  const [filterParams, setFilterParams] = useState({
    category: 'All',
    customerType: 'All',
    gender: 'All',
    dateRange: ['', '']
  });
  
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch categories on component mount
    const fetchCategories = async () => {
      try {
        const categoriesData = await apiService.getCategories();
        setCategories(categoriesData);
        
        // Initialize date range filter with min and max dates
        setFilterParams(prev => ({
          ...prev,
          dateRange: [categoriesData.dateRange.min, categoriesData.dateRange.max]
        }));
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching categories:', error);
        setLoading(false);
      }
    };
    
    fetchCategories();
  }, []);
  
  // Handle filter changes
  const handleFilterChange = (newFilters) => {
    setFilterParams({
      ...filterParams,
      ...newFilters
    });
  };
  
  if (loading) {
    return <div className="loading">Loading application...</div>;
  }
  
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Supermarket Sales Dashboard</h1>
          <nav>
            <ul className="nav-links">
              <li><Link to="/">Dashboard</Link></li>
              <li><Link to="/data">Data Explorer</Link></li>
              <li><Link to="/products">Product Analysis</Link></li>
              <li><Link to="/customers">Customer Analysis</Link></li>
              <li><Link to="/time">Time Analysis</Link></li>
              <li><a href="http://localhost:8501" target="_blank" rel="noopener noreferrer">Streamlit App</a></li>
            </ul>
          </nav>
        </header>
        
        <main className="main-content">
          <aside className="sidebar">
            <FilterPanel 
              categories={categories} 
              filterParams={filterParams} 
              onFilterChange={handleFilterChange} 
            />
          </aside>
          
          <div className="content">
            <Routes>
              <Route path="/" element={<Dashboard filterParams={filterParams} />} />
              <Route path="/data" element={<DataExplorer filterParams={filterParams} />} />
              <Route path="/products" element={<ProductAnalysis filterParams={filterParams} />} />
              <Route path="/customers" element={<CustomerAnalysis filterParams={filterParams} />} />
              <Route path="/time" element={<TimeAnalysis filterParams={filterParams} />} />
            </Routes>
          </div>
        </main>
        
        <footer className="App-footer">
          <p>Supermarket Sales Dashboard &copy; 2025</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;

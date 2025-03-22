import React, { useState, useEffect } from 'react';
import apiService from '../api';

const DataExplorer = ({ filterParams }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [pagination, setPagination] = useState({
    currentPage: 1,
    itemsPerPage: 10,
    totalItems: 0
  });
  
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const filteredData = await apiService.getFilteredData(filterParams);
        setData(filteredData);
        setPagination(prev => ({
          ...prev,
          totalItems: filteredData.length,
          currentPage: 1 // Reset to first page when filter changes
        }));
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        setLoading(false);
      }
    };
    
    fetchData();
  }, [filterParams]);
  
  const handlePageChange = (newPage) => {
    setPagination(prev => ({
      ...prev,
      currentPage: newPage
    }));
  };
  
  const handleItemsPerPageChange = (event) => {
    setPagination(prev => ({
      ...prev,
      itemsPerPage: parseInt(event.target.value, 10),
      currentPage: 1 // Reset to first page when items per page changes
    }));
  };
  
  if (loading) {
    return <div className="loading">Loading data...</div>;
  }
  
  // Calculate pagination
  const indexOfLastItem = pagination.currentPage * pagination.itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - pagination.itemsPerPage;
  const currentItems = data.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(pagination.totalItems / pagination.itemsPerPage);
  
  // Generate table headers from the first data item
  const headers = currentItems.length > 0 ? Object.keys(currentItems[0]) : [];
  
  return (
    <div className="data-explorer">
      <h2>Data Explorer</h2>
      <p>Showing {pagination.totalItems} records based on current filter selections</p>
      
      <div className="pagination-controls">
        <div className="items-per-page">
          <label htmlFor="items-per-page">Items per page:</label>
          <select 
            id="items-per-page" 
            value={pagination.itemsPerPage}
            onChange={handleItemsPerPageChange}
          >
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
        
        <div className="download-btn">
          <button onClick={() => window.open(apiService.downloadFilteredData(filterParams))}>
            Download as CSV
          </button>
        </div>
      </div>
      
      <div className="table-container">
        <table>
          <thead>
            <tr>
              {headers.map(header => (
                <th key={header}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {currentItems.map((item, index) => (
              <tr key={index}>
                {headers.map(header => (
                  <td key={`${index}-${header}`}>
                    {item[header] instanceof Date 
                      ? item[header].toLocaleDateString() 
                      : item[header]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="pagination">
        <button 
          onClick={() => handlePageChange(pagination.currentPage - 1)}
          disabled={pagination.currentPage === 1}
        >
          Previous
        </button>
        
        <span>Page {pagination.currentPage} of {totalPages}</span>
        
        <button 
          onClick={() => handlePageChange(pagination.currentPage + 1)}
          disabled={pagination.currentPage === totalPages}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default DataExplorer;

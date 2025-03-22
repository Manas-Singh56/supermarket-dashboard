import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create an axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

const apiService = {
  // Get the full dataset
  getData: async () => {
    try {
      const response = await api.get('/data');
      return response.data;
    } catch (error) {
      console.error('Error fetching data:', error);
      throw error;
    }
  },

  // Get filtered data based on parameters
  getFilteredData: async (filterParams) => {
    try {
      const response = await api.post('/data/filtered', filterParams);
      return response.data;
    } catch (error) {
      console.error('Error fetching filtered data:', error);
      throw error;
    }
  },

  // Get statistical summary
  getSummary: async () => {
    try {
      const response = await api.get('/data/summary');
      return response.data;
    } catch (error) {
      console.error('Error fetching summary data:', error);
      throw error;
    }
  },

  // Get unique categories for filtering
  getCategories: async () => {
    try {
      const response = await api.get('/data/categories');
      return response.data;
    } catch (error) {
      console.error('Error fetching categories:', error);
      throw error;
    }
  },

  // Get time series data
  getTimeSeriesData: async (params) => {
    try {
      const response = await api.post('/data/time_series', params);
      return response.data;
    } catch (error) {
      console.error('Error fetching time series data:', error);
      throw error;
    }
  },

  // Get correlation data
  getCorrelationData: async () => {
    try {
      const response = await api.get('/data/correlation');
      return response.data;
    } catch (error) {
      console.error('Error fetching correlation data:', error);
      throw error;
    }
  },

  // Get product analysis
  getProductAnalysis: async (params) => {
    try {
      const response = await api.post('/data/product_analysis', params);
      return response.data;
    } catch (error) {
      console.error('Error fetching product analysis:', error);
      throw error;
    }
  },

  // Generate download link for filtered data
  downloadFilteredData: (filterParams) => {
    return `${API_URL}/data/download?params=${encodeURIComponent(JSON.stringify(filterParams))}`;
  }
};

export default apiService;

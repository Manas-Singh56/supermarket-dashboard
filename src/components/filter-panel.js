import React from 'react';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

function FilterPanel({ categories, filterParams, onFilterChange }) {
  const handleCategoryChange = (event) => {
    onFilterChange({ category: event.target.value });
  };

  const handleCustomerTypeChange = (event) => {
    onFilterChange({ customerType: event.target.value });
  };

  const handleGenderChange = (event) => {
    onFilterChange({ gender: event.target.value });
  };

  const handleStartDateChange = (date) => {
    onFilterChange({ 
      dateRange: [
        date ? date.toISOString().split('T')[0] : filterParams.dateRange[0], 
        filterParams.dateRange[1]
      ] 
    });
  };

  const handleEndDateChange = (date) => {
    onFilterChange({ 
      dateRange: [
        filterParams.dateRange[0], 
        date ? date.toISOString().split('T')[0] : filterParams.dateRange[1]
      ] 
    });
  };

  return (
    <div className="filter-panel">
      <h2>Filters</h2>
      
      <div className="filter-group">
        <label htmlFor="category">Product Category:</label>
        <select 
          id="category" 
          value={filterParams.category}
          onChange={handleCategoryChange}
        >
          {categories.productCategories.map(category => (
            <option key={category} value={category}>{category}</option>
          ))}
        </select>
      </div>
      
      <div className="filter-group">
        <label htmlFor="customerType">Customer Type:</label>
        <select 
          id="customerType" 
          value={filterParams.customerType}
          onChange={handleCustomerTypeChange}
        >
          {categories.customerTypes.map(type => (
            <option key={type} value={type}>{type}</option>
          ))}
        </select>
      </div>
      
      <div className="filter-group">
        <label htmlFor="gender">Gender:</label>
        <select 
          id="gender" 
          value={filterParams.gender}
          onChange={handleGenderChange}
        >
          {categories.genders.map(gender => (
            <option key={gender} value={gender}>{gender}</option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label>Date Range:</label>
        <div>
          <DatePicker
            selected={new Date(filterParams.dateRange[0])}
            onChange={handleStartDateChange}
            selectsStart
            startDate={new Date(filterParams.dateRange[0])}
            endDate={new Date(filterParams.dateRange[1])}
            dateFormat="yyyy-MM-dd"
            placeholderText="Start Date"
          />
          <DatePicker
            selected={new Date(filterParams.dateRange[1])}
            onChange={handleEndDateChange}
            selectsEnd
            startDate={new Date(filterParams.dateRange[0])}
            endDate={new Date(filterParams.dateRange[1])}
            minDate={new Date(filterParams.dateRange[0])}
            dateFormat="yyyy-MM-dd"
            placeholderText="End Date"
          />
        </div>
      </div>
    </div>
  );
}

export default FilterPanel;
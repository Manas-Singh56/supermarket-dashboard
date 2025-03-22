# Supermarket Sales Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-17.0.2+-61DAFB.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1+-000000.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10.0+-FF4B4B.svg)](https://streamlit.io/)

An interactive data analytics dashboard for visualizing and analyzing supermarket sales trends, built with Flask, Streamlit, and React.


## 🚀 Features

- **Sales Trend Analysis**: Interactive visualization of sales over time
- **Product Performance**: Compare sales across different product categories
- **Customer Segmentation**: Analyze purchasing patterns by demographic
- **Multi-dimensional Filtering**: Filter by date, product, branch, and more
- **Responsive Design**: Optimized for desktop and mobile devices
- **Dual Interface**: Choose between React dashboard or Streamlit app

## 🔧 Technology Stack

### Backend
- **Flask**: RESTful API serving processed data
- **Streamlit**: Alternative data visualization interface
- **Pandas & NumPy**: Data processing and analysis
- **SQLite/PostgreSQL**: Data storage (configurable)

### Frontend
- **React**: Component-based UI with hooks
- **Chart.js**: Dynamic data visualizations
- **Material UI**: Modern, responsive component library
- **Axios**: API communication

## 📊 Dataset

The project utilizes `supermarketsales.csv`, a comprehensive dataset containing:
- Transaction details
- Product categories
- Customer information
- Payment methods
- Branch data
- Time-series sales data

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

## 🔌 Installation & Setup

### Clone Repository
```bash
git clone https://github.com/yourusername/supermarket-dashboard.git
cd supermarket-dashboard
```

### Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py

# In a new terminal, start Streamlit
streamlit run streamlit_app.py
```

### Frontend Setup
```bash
# Navigate to React app directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 🖥️ Usage

- **React Dashboard**: Access at `http://localhost:3000`
- **Streamlit App**: Access at `http://localhost:8501`
- Use the date picker and dropdown filters to customize your view
- Toggle between different visualization types using the tabs
- Hover over charts for detailed tooltips


```


## 🛠️ Development

### Running Tests
```bash
# Backend tests
pytest

# Frontend tests
cd frontend
npm test
```



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/supermarket-dashboard](https://github.com/yourusername/supermarket-dashboard)

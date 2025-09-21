# StockBot Advisor - AI-Powered Financial Advisory Platform

**Author:** Anthony Winata Salim  
**Student Number:** 230726051  
**Course:** CM3070 Project  
**Supervisor:** Hwa Heng Kan

## 📋 Project Overview

StockBot Advisor is an AI-powered financial advisory platform that combines machine learning with technical analysis to provide transparent, educational investment recommendations for retail investors. The system analyzes 15 stocks across multiple sectors plus 3 market benchmarks, offering real-time insights through a minimalistic, modern web interface.

## 🎯 Key Features

- **Hybrid Analysis Engine**: Combines Random Forest ML with technical indicators (RSI, MACD, SMA)
- **Real-time Market Dashboard**: Live stock prices, portfolio tracking, and market overview
- **AI-Powered Predictions**: Buy/Sell/Hold signals with confidence scores
- **Natural Language Explanations**: Clear reasoning for every recommendation
- **Portfolio Management**: Track holdings, performance, and risk metrics
- **Backtesting System**: Test strategies on historical data
- **Educational Resources**: Learn about indicators and investment strategies
- **Minimalistic Design**: Clean black, white, and gray interface

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/financial-advisor-bot.git
cd financial-advisor-bot
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
financial-advisor-bot/
├── app.py                 # Main application entry
├── config.yaml           # Configuration settings
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── .streamlit/          # Streamlit configuration
│   └── config.toml     # Theme settings
├── pages/              # Application pages
│   ├── 1_📊_Dashboard.py
│   ├── 2_🔍_Analysis.py
│   ├── 3_💼_Portfolio.py
│   ├── 4_📈_Backtesting.py
│   ├── 5_🎓_Education.py
│   └── 6_⚙️_Settings.py
├── utils/              # Utility modules
│   ├── data_processor.py
│   ├── ml_models.py
│   ├── technical_indicators.py
│   ├── llm_explainer.py
│   └── database.py
├── components/         # Reusable UI components
│   ├── charts.py
│   ├── metrics.py
│   ├── sidebar.py
│   └── alerts.py
└── assets/            # Static assets
    ├── style.css
    └── logo.png
```

## 📊 Stock Coverage

### Technology Sector (6 stocks)
- AAPL - Apple Inc.
- MSFT - Microsoft Corporation
- GOOGL - Alphabet Inc.
- NVDA - NVIDIA Corporation
- META - Meta Platforms Inc.
- PLTR - Palantir Technologies

### Financial Sector (5 stocks)
- JPM - JPMorgan Chase & Co.
- BAC - Bank of America
- GS - Goldman Sachs
- MS - Morgan Stanley
- V - Visa Inc.

### Healthcare Sector (4 stocks)
- JNJ - Johnson & Johnson
- PFE - Pfizer Inc.
- MRNA - Moderna Inc.
- UNH - UnitedHealth Group

### Market Benchmarks (3 ETFs)
- SPY - S&P 500 ETF
- QQQ - NASDAQ 100 ETF
- DIA - Dow Jones ETF

## 🔧 Technical Stack

- **Frontend**: Streamlit, Plotly, Custom CSS
- **Backend**: Python, pandas, NumPy
- **Machine Learning**: scikit-learn, XGBoost
- **Data Source**: yfinance API
- **Database**: SQLite/PostgreSQL
- **Technical Analysis**: ta, pandas-ta
- **Deployment**: Docker, AWS/Heroku

## 📈 Features Breakdown

### Dashboard
- Real-time stock prices
- Portfolio performance metrics
- Market overview with indices
- AI recommendations summary
- News feed integration

### Analysis
- Interactive candlestick charts
- 15+ technical indicators
- ML prediction confidence meters
- Fundamental analysis metrics
- Sentiment analysis

### Portfolio Management
- Holdings tracker
- Performance vs benchmarks
- Risk metrics (Sharpe, Beta)
- Transaction history
- Rebalancing suggestions

### Backtesting
- Strategy builder
- Historical performance analysis
- Risk-adjusted returns
- Comparison with buy-and-hold
- Monte Carlo simulations

## 🎨 Design Philosophy

The interface follows a minimalistic design approach with:
- **Color Palette**: Black (#000000), White (#FFFFFF), Gray (#6C757D, #F8F9FA)
- **Typography**: Clean sans-serif (Inter, SF Pro)
- **Layout**: Generous whitespace, subtle shadows, clear hierarchy
- **Interactions**: Smooth transitions, hover states, responsive design

## 📊 Machine Learning Model

### Features Used
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Band Position
- Stochastic Oscillator
- Volume Ratio
- ATR (Average True Range)

### Model Performance
- Algorithm: Random Forest Classifier
- Accuracy: ~45-50% (above baseline)
- Confidence Range: 37-77%
- Feature Importance: Balanced across indicators

## 🔐 Security

- Environment variables for sensitive data
- SQL injection prevention
- XSS protection
- Rate limiting
- Session management
- Input validation

## 📝 Configuration

Edit `config.yaml` to customize:
- Stock universe
- Technical indicator parameters
- ML model settings
- Risk management rules
- UI preferences
- Update frequencies

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 📖 Usage Guide

1. **Dashboard**: Monitor your portfolio and market overview
2. **Analysis**: Deep dive into individual stocks with AI predictions
3. **Portfolio**: Manage your holdings and track performance
4. **Backtesting**: Test strategies before implementation
5. **Education**: Learn about indicators and strategies
6. **Settings**: Customize your risk profile and preferences

## 🤝 Contributing

This is an academic project for CM3070. For educational collaboration:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request
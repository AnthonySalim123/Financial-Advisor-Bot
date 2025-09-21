# StockBot Advisor - AI-Powered Financial Advisory Platform

**Author:** Anthony Winata Salim  
**Student Number:** 230726051  
**Course:** CM3070 Project  
**Supervisor:** Hwa Heng Kan

## 📋 Project Overview

StockBot Advisor is an AI-powered financial advisory platform that combines machine learning with technical analysis to provide transparent, educational investment recommendations for retail investors. The system analyzes 18 stocks across multiple sectors plus 3 market benchmarks, offering real-time insights through a minimalistic, modern web interface.

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
git clone https://github.com/anthonysalim123/financial-advisor-bot.git
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

4. **Initialize the system**
```bash
python main.py init
```

5. **Run the application**
```bash
# Option 1: Using the main coordinator (recommended)
python main.py run

# Option 2: Direct Streamlit
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎮 System Commands

The `main.py` coordinator provides several commands:

```bash
# Initialize system (create directories, database)
python main.py init

# Run full application
python main.py run [--debug]

# Train ML models
python main.py train [--symbols AAPL MSFT GOOGL]

# Run batch analysis
python main.py batch [--symbols AAPL MSFT]

# Monitor portfolio
python main.py monitor [--interval 60]

# Run tests
python main.py test
```

## 📁 Project Structure

```
financial-advisor-bot/
├── main.py               # Main system coordinator (entry point)
├── app.py                # Streamlit application
├── config.yaml           # Configuration settings
├── requirements.txt      # Python dependencies
├── init_db.py           # Database initialization script
├── practical_trading_system.py  # High-confidence trading model trainer
├── test_practical_integration.py # Practical system integration tests
├── .gitignore           # Git ignore file
├── .streamlit/          # Streamlit configuration
│   └── config.toml      # Theme settings
├── pages/               # Application pages
│   ├── dashboard.py     # Market dashboard with practical AI signals
│   ├── portfolio.py     # Portfolio management
│   ├── backtesting.py   # Strategy backtesting
│   ├── education.py     # Educational resources
│   └── settings.py      # User settings
├── utils/               # Utility modules
│   ├── data_processor.py      # Data fetching and processing
│   ├── ml_models.py           # Machine learning models
│   ├── technical_indicators.py # Technical analysis
│   ├── llm_explainer.py       # Natural language explanations
│   ├── database.py            # Database operations
│   ├── sentiment_analyzer.py  # Sentiment analysis
│   └── user_evaluation.py     # User feedback system
├── components/          # Reusable UI components
│   ├── charts.py        # Chart components
│   ├── metrics.py       # Metric displays
│   ├── sidebar.py       # Sidebar navigation
│   └── alerts.py        # Alert system
├── assets/             # Static assets
│   └── style.css       # Custom CSS styling
├── data/               # Data directory (auto-created)
│   └── models/         # Trained ML models
└── logs/               # Log files (auto-created)
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

### Real Estate Sector (3 stocks)
- MA - Mastercard Inc.
- PLD - Prologis Inc.
- O - Realty Income Corporation

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

### Core Technologies
- **Frontend**: Streamlit, Plotly, Custom CSS
- **Backend**: Python 3.9+, pandas, NumPy
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting)
- **Data Source**: yfinance API
- **Database**: SQLite with SQLAlchemy
- **Technical Analysis**: ta library
- **Configuration**: YAML

### Optional Enhancements
- **Advanced ML**: XGBoost, LightGBM (for improved predictions)
- **Explainability**: SHAP (for model interpretability)
- **Sentiment Analysis**: Transformers, TextBlob (for news sentiment)
- **Data Balancing**: imbalanced-learn / SMOTE (for better ML training)

## 📈 Features Breakdown

### Dashboard
- Real-time stock prices with auto-refresh
- Portfolio performance metrics and P&L tracking
- Market overview with major indices
- AI-powered buy/sell/hold recommendations
- **Practical Trading Signals** with confidence filtering
- Watchlist with quick access to favorites

### Analysis
- Interactive candlestick charts with Plotly
- 15+ technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)
- ML prediction confidence meters
- Feature importance visualization
- Natural language explanations of signals

### Portfolio Management
- Holdings tracker with real-time valuations
- Performance comparison vs benchmarks
- Risk metrics (Sharpe Ratio, Beta, Volatility)
- Transaction history and record keeping
- Rebalancing suggestions based on targets

### Backtesting
- Multiple strategy options:
  - Moving Average Crossover
  - RSI Mean Reversion
  - MACD Momentum
  - Bollinger Bands
- Historical performance analysis
- Risk-adjusted returns calculation
- Win rate and profit factor metrics
- Detailed trade-by-trade analysis

### Education Center
- Interactive indicator tutorials
- Strategy explanations with examples
- Risk management best practices
- Investment terminology glossary
- Progress tracking system

### Settings
- User profile management
- Risk tolerance configuration
- Notification preferences
- Theme customization
- API configuration

## 🎨 Design Philosophy

The interface follows a minimalistic design approach with:
- **Color Palette**: Black (#000000), White (#FFFFFF), Gray (#6C757D, #F8F9FA)
- **Typography**: Clean sans-serif (Inter, SF Pro Display)
- **Layout**: Generous whitespace, subtle shadows, clear hierarchy
- **Interactions**: Smooth transitions, hover states, responsive design
- **Accessibility**: High contrast, clear labels, keyboard navigation

## 📊 Machine Learning Model

### Features Used
- Technical Indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Band Position
  - Stochastic Oscillator
  - ATR (Average True Range)
- Price-based Features:
  - SMA/EMA Crossovers
  - Volume Ratio
  - Price momentum

### Model Performance
- **Algorithm**: Ensemble (Random Forest + Gradient Boosting)
- **Accuracy**: ~45-50% (above random baseline of 33%)
- **Confidence Range**: 37-77%
- **Feature Importance**: Balanced across technical indicators
- **Validation**: Time series cross-validation

### Practical Trading System
The system includes a specialized **high-confidence trading model** that:
- Focuses only on trades with >50% confidence
- Achieves 51-60% accuracy on filtered trades
- Trades only 10-20% of the time (patient approach)
- Prioritizes risk management over frequency
- Can be trained using `python practical_trading_system.py`

## 🔐 Security

- Environment variables for sensitive configuration
- SQL injection prevention via SQLAlchemy ORM
- Input validation and sanitization
- Session state management
- No hardcoded credentials or API keys
- Secure data storage practices

## 📝 Configuration

Edit `config.yaml` to customize:
- Stock universe selection
- Technical indicator parameters
- ML model hyperparameters
- Risk management rules
- UI preferences and themes
- Data update frequencies

## 🧪 Testing

Initialize and test the system:

```bash
# Initialize database and directories
python main.py init

# Run system tests (if test_setup.py exists)
python main.py test

# Train ML models
python main.py train

# Train practical trading model (high-confidence trades)
python practical_trading_system.py

# Test practical trading integration
python test_practical_integration.py
```

## 📖 Usage Guide

1. **Initialize**: Run `python main.py init` to set up the system
2. **Dashboard**: Monitor your portfolio and market overview
3. **Analysis**: Deep dive into individual stocks with AI predictions
4. **Portfolio**: Add transactions and track performance
5. **Backtesting**: Test strategies on historical data before implementation
6. **Education**: Learn about technical indicators and strategies
7. **Settings**: Customize your risk profile and preferences

## 📚 Academic Context

This project is developed as part of the CM3070 Final Project course at the University of London International Programmes. It builds upon the CM3020 Artificial Intelligence template for a Financial Advisor Bot, extending it with:

- Enhanced ML models using ensemble methods
- Real estate sector coverage for diversification
- Comprehensive backtesting system
- User evaluation and feedback collection
- Natural language explanations via LLM integration
- Professional-grade UI/UX design
- **Practical Trading System** focusing on high-confidence trades only

The project demonstrates practical application of:
- Machine Learning for financial prediction
- Technical analysis implementation
- Software engineering best practices
- User-centered design principles
- Academic research methodologies


## 🙏 Acknowledgments

- **University**: University of London International Programmes
- **Supervisor**: Hwa Heng Kan
- **Course**: CM3070 Final Project
- **Template**: Based on CM3020 Artificial Intelligence - Financial Advisor Bot
- **Libraries**: Thanks to the open-source community for excellent tools:
  - Streamlit team for the web framework
  - yfinance for market data access
  - scikit-learn for ML capabilities
  - Plotly for interactive visualizations


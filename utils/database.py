"""
Database Module
Handles all database operations for user data, portfolios, and transactions
Using SQLite for simplicity (can be upgraded to PostgreSQL)
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import hashlib
import secrets

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for the StockBot Advisor"""
    
    def __init__(self, db_path: str = 'data/stockbot.db'):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        # Create data directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _create_tables(self):
        """Create all necessary tables if they don't exist"""
        
        # Users table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                profile_data TEXT
            )
        ''')
        
        # User profiles table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                risk_tolerance TEXT DEFAULT 'Moderate',
                investment_horizon TEXT DEFAULT '1-3 years',
                initial_capital REAL DEFAULT 100000,
                currency TEXT DEFAULT 'USD',
                preferred_sectors TEXT,
                notification_settings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Portfolios table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                portfolio_name TEXT DEFAULT 'Main Portfolio',
                total_value REAL DEFAULT 0,
                cash_balance REAL DEFAULT 0,
                total_return REAL DEFAULT 0,
                daily_return REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Holdings table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_purchase_price REAL NOT NULL,
                current_price REAL,
                current_value REAL,
                unrealized_pnl REAL,
                unrealized_pnl_pct REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
        ''')
        
        # Transactions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                total_amount REAL NOT NULL,
                commission REAL DEFAULT 0,
                notes TEXT,
                transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
        ''')
        
        # Watchlists table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlists (
                watchlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                watchlist_name TEXT DEFAULT 'Main Watchlist',
                symbols TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Alerts table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                condition TEXT NOT NULL,
                threshold_value REAL,
                is_active BOOLEAN DEFAULT 1,
                triggered_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Predictions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                signal TEXT,
                confidence REAL,
                predicted_price REAL,
                current_price REAL,
                indicators_data TEXT,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Backtest results table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                backtest_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                strategy_name TEXT NOT NULL,
                symbols TEXT NOT NULL,
                start_date DATE,
                end_date DATE,
                initial_capital REAL,
                final_capital REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                num_trades INTEGER,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    # User Management Methods
    
    def create_user(self, username: str, email: str = None, password: str = None) -> int:
        """Create a new user"""
        try:
            password_hash = self._hash_password(password) if password else None
            
            self.cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            user_id = self.cursor.lastrowid
            
            # Create default profile
            self.cursor.execute('''
                INSERT INTO user_profiles (user_id)
                VALUES (?)
            ''', (user_id,))
            
            # Create default portfolio
            self.cursor.execute('''
                INSERT INTO portfolios (user_id, cash_balance, total_value)
                VALUES (?, 100000, 100000)
            ''', (user_id,))
            
            # Create default watchlist
            self.cursor.execute('''
                INSERT INTO watchlists (user_id, symbols)
                VALUES (?, ?)
            ''', (user_id, json.dumps(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'])))
            
            self.conn.commit()
            logger.info(f"Created user: {username} (ID: {user_id})")
            return user_id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed: {e}")
            return -1
    
    def get_user(self, user_id: int = None, username: str = None) -> Optional[Dict]:
        """Get user by ID or username"""
        if user_id:
            self.cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        elif username:
            self.cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        else:
            return None
        
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def update_user_profile(self, user_id: int, profile_data: Dict) -> bool:
        """Update user profile"""
        try:
            self.cursor.execute('''
                UPDATE user_profiles
                SET risk_tolerance = ?, investment_horizon = ?, 
                    initial_capital = ?, preferred_sectors = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (
                profile_data.get('risk_tolerance', 'Moderate'),
                profile_data.get('investment_horizon', '1-3 years'),
                profile_data.get('initial_capital', 100000),
                json.dumps(profile_data.get('preferred_sectors', [])),
                user_id
            ))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Profile update failed: {e}")
            return False
    
    # Portfolio Management Methods
    
    def get_portfolio(self, user_id: int) -> Optional[Dict]:
        """Get user's portfolio"""
        self.cursor.execute('''
            SELECT * FROM portfolios 
            WHERE user_id = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id,))
        
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def get_holdings(self, portfolio_id: int) -> List[Dict]:
        """Get all holdings in a portfolio"""
        self.cursor.execute('''
            SELECT * FROM holdings
            WHERE portfolio_id = ?
            ORDER BY current_value DESC
        ''', (portfolio_id,))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    def add_transaction(self, portfolio_id: int, transaction_data: Dict) -> int:
        """Add a new transaction"""
        try:
            self.cursor.execute('''
                INSERT INTO transactions 
                (portfolio_id, symbol, transaction_type, quantity, price, total_amount, commission, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                portfolio_id,
                transaction_data['symbol'],
                transaction_data['type'],  # BUY or SELL
                transaction_data['quantity'],
                transaction_data['price'],
                transaction_data['quantity'] * transaction_data['price'],
                transaction_data.get('commission', 0),
                transaction_data.get('notes', '')
            ))
            
            transaction_id = self.cursor.lastrowid
            
            # Update holdings
            self._update_holdings(portfolio_id, transaction_data)
            
            self.conn.commit()
            return transaction_id
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            self.conn.rollback()
            return -1
    
    def _update_holdings(self, portfolio_id: int, transaction: Dict):
        """Update holdings based on transaction"""
        symbol = transaction['symbol']
        quantity = transaction['quantity']
        price = transaction['price']
        
        # Check if holding exists
        self.cursor.execute('''
            SELECT * FROM holdings
            WHERE portfolio_id = ? AND symbol = ?
        ''', (portfolio_id, symbol))
        
        existing = self.cursor.fetchone()
        
        if transaction['type'] == 'BUY':
            if existing:
                # Update existing holding
                new_quantity = existing['quantity'] + quantity
                new_avg_price = ((existing['quantity'] * existing['avg_purchase_price']) + 
                               (quantity * price)) / new_quantity
                
                self.cursor.execute('''
                    UPDATE holdings
                    SET quantity = ?, avg_purchase_price = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE holding_id = ?
                ''', (new_quantity, new_avg_price, existing['holding_id']))
            else:
                # Create new holding
                self.cursor.execute('''
                    INSERT INTO holdings
                    (portfolio_id, symbol, quantity, avg_purchase_price, current_price, current_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (portfolio_id, symbol, quantity, price, price, quantity * price))
        
        elif transaction['type'] == 'SELL' and existing:
            new_quantity = existing['quantity'] - quantity
            
            if new_quantity <= 0:
                # Remove holding
                self.cursor.execute('DELETE FROM holdings WHERE holding_id = ?', 
                                   (existing['holding_id'],))
            else:
                # Update holding
                self.cursor.execute('''
                    UPDATE holdings
                    SET quantity = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE holding_id = ?
                ''', (new_quantity, existing['holding_id']))
    
    def get_transactions(self, portfolio_id: int, limit: int = 50) -> List[Dict]:
        """Get transaction history"""
        self.cursor.execute('''
            SELECT * FROM transactions
            WHERE portfolio_id = ?
            ORDER BY transaction_date DESC
            LIMIT ?
        ''', (portfolio_id, limit))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    # Watchlist Methods
    
    def get_watchlist(self, user_id: int) -> List[str]:
        """Get user's watchlist"""
        self.cursor.execute('''
            SELECT symbols FROM watchlists
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id,))
        
        row = self.cursor.fetchone()
        return json.loads(row['symbols']) if row else []
    
    def update_watchlist(self, user_id: int, symbols: List[str]) -> bool:
        """Update user's watchlist"""
        try:
            self.cursor.execute('''
                UPDATE watchlists
                SET symbols = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (json.dumps(symbols), user_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Watchlist update failed: {e}")
            return False
    
    # Predictions Methods
    
    def save_prediction(self, prediction_data: Dict) -> int:
        """Save a prediction to database"""
        try:
            self.cursor.execute('''
                INSERT INTO predictions
                (symbol, prediction_type, signal, confidence, predicted_price, 
                 current_price, indicators_data, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data['symbol'],
                prediction_data.get('type', 'classification'),
                prediction_data.get('signal'),
                prediction_data.get('confidence'),
                prediction_data.get('predicted_price'),
                prediction_data.get('current_price'),
                json.dumps(prediction_data.get('indicators', {})),
                prediction_data.get('model_version', '1.0')
            ))
            
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            logger.error(f"Save prediction failed: {e}")
            return -1
    
    def get_predictions(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get recent predictions"""
        if symbol:
            self.cursor.execute('''
                SELECT * FROM predictions
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (symbol, limit))
        else:
            self.cursor.execute('''
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
        
        return [dict(row) for row in self.cursor.fetchall()]
    
    # Backtest Methods
    
    def save_backtest_result(self, user_id: int, result_data: Dict) -> int:
        """Save backtest results"""
        try:
            self.cursor.execute('''
                INSERT INTO backtest_results
                (user_id, strategy_name, symbols, start_date, end_date,
                 initial_capital, final_capital, total_return, sharpe_ratio,
                 max_drawdown, win_rate, num_trades, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                result_data.get('strategy_name', 'Random Forest'),
                json.dumps(result_data.get('symbols', [])),
                result_data.get('start_date'),
                result_data.get('end_date'),
                result_data.get('initial_capital', 100000),
                result_data.get('final_capital'),
                result_data.get('total_return'),
                result_data.get('sharpe_ratio'),
                result_data.get('max_drawdown'),
                result_data.get('win_rate'),
                result_data.get('num_trades'),
                json.dumps(result_data.get('parameters', {}))
            ))
            
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            logger.error(f"Save backtest failed: {e}")
            return -1
    
    def get_backtest_results(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get user's backtest results"""
        self.cursor.execute('''
            SELECT * FROM backtest_results
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = []
        for row in self.cursor.fetchall():
            result = dict(row)
            result['symbols'] = json.loads(result['symbols'])
            result['parameters'] = json.loads(result['parameters'])
            results.append(result)
        
        return results
    
    # Utility Methods
    
    def _hash_password(self, password: str) -> str:
        """Hash password for storage"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                           password.encode('utf-8'),
                                           salt.encode('utf-8'),
                                           100000)
        return salt + password_hash.hex()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        salt = password_hash[:32]
        stored_hash = password_hash[32:]
        
        new_hash = hashlib.pbkdf2_hmac('sha256',
                                       password.encode('utf-8'),
                                       salt.encode('utf-8'),
                                       100000)
        
        return new_hash.hex() == stored_hash
    
    def export_portfolio_to_csv(self, portfolio_id: int, filepath: str):
        """Export portfolio data to CSV"""
        holdings = self.get_holdings(portfolio_id)
        df = pd.DataFrame(holdings)
        df.to_csv(filepath, index=False)
        logger.info(f"Portfolio exported to {filepath}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()

# Create a singleton instance
_db_instance = None

def get_database() -> DatabaseManager:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
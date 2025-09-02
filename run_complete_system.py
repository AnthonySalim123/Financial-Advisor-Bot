#!/usr/bin/env python
"""
Complete System Runner
Ensures all components are properly initialized and running
"""

import os
import sys
import subprocess
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemRunner:
    """Manages the complete Financial Advisor Bot system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / 'requirements.txt'
        self.app_file = self.project_root / 'app.py'
        self.data_dir = self.project_root / 'data'
        self.logs_dir = self.project_root / 'logs'
    
    def check_environment(self):
        """Check if environment is properly set up"""
        logger.info("Checking environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 9:
            logger.error(f"Python 3.9+ required, found {sys.version}")
            return False
        
        # Check directories
        for directory in [self.data_dir, self.logs_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def initialize_database(self):
        """Initialize the database"""
        logger.info("Initializing database...")
        
        try:
            from utils.database import get_database
            db = get_database()
            
            # Create demo user if not exists
            user_id = db.create_user("demo_user", "demo@stockbot.com", "demo123")
            if user_id > 0:
                logger.info(f"Created demo user with ID: {user_id}")
            else:
                logger.info("Demo user already exists")
            
            return True
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def train_initial_models(self):
        """Train initial ML models"""
        logger.info("Training initial models...")
        
        try:
            from utils.ml_models import create_prediction_model
            from utils.data_processor import get_data_processor
            from utils.technical_indicators import TechnicalIndicators
            
            data_processor = get_data_processor()
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            for symbol in symbols:
                logger.info(f"Training model for {symbol}...")
                
                # Fetch data
                df = data_processor.fetch_stock_data(symbol, period='2y')
                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # Calculate indicators
                df = TechnicalIndicators.calculate_all_indicators(df)
                
                # Train model
                model = create_prediction_model('classification')
                metrics = model.train(df)
                
                if 'error' not in metrics:
                    # Save model
                    model_path = self.data_dir / f'models/{symbol}_model.pkl'
                    model_path.parent.mkdir(exist_ok=True)
                    model.save_model(str(model_path))
                    logger.info(f"Model for {symbol} trained successfully: Accuracy={metrics['accuracy']:.2%}")
                else:
                    logger.error(f"Failed to train model for {symbol}: {metrics['error']}")
            
            return True
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def run_tests(self):
        """Run system tests"""
        logger.info("Running system tests...")
        
        tests_passed = 0
        tests_failed = 0
        
        # Test imports
        test_modules = [
            'streamlit',
            'pandas',
            'numpy',
            'yfinance',
            'plotly',
            'sklearn'
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module} imported successfully")
                tests_passed += 1
            except ImportError:
                logger.error(f"❌ Failed to import {module}")
                tests_failed += 1
        
        # Test data fetching
        try:
            from utils.data_processor import get_data_processor
            processor = get_data_processor()
            df = processor.fetch_stock_data('AAPL', period='1mo')
            if not df.empty:
                logger.info("✅ Data fetching working")
                tests_passed += 1
            else:
                logger.error("❌ Data fetching returned empty")
                tests_failed += 1
        except Exception as e:
            logger.error(f"❌ Data fetching failed: {e}")
            tests_failed += 1
        
        logger.info(f"Tests completed: {tests_passed} passed, {tests_failed} failed")
        return tests_failed == 0
    
    def launch_application(self):
        """Launch the Streamlit application"""
        logger.info("Launching StockBot Advisor...")
        
        try:
            # Launch Streamlit
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(self.app_file)],
                check=False
            )
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Failed to launch application: {e}")
    
    def run(self):
        """Run the complete system"""
        logger.info("=" * 50)
        logger.info("StockBot Advisor - System Startup")
        logger.info(f"Time: {datetime.now()}")
        logger.info("=" * 50)
        
        # Step 1: Check environment
        if not self.check_environment():
            logger.error("Environment check failed")
            return False
        
        # Step 2: Install dependencies
        if not self.install_dependencies():
            logger.error("Dependency installation failed")
            return False
        
        # Step 3: Initialize database
        if not self.initialize_database():
            logger.error("Database initialization failed")
            return False
        
        # Step 4: Train initial models (optional)
        train_models = input("Train initial models? (y/n): ").lower() == 'y'
        if train_models:
            if not self.train_initial_models():
                logger.warning("Model training failed, continuing anyway...")
        
        # Step 5: Run tests
        if not self.run_tests():
            continue_anyway = input("Tests failed. Continue anyway? (y/n): ").lower() == 'y'
            if not continue_anyway:
                return False
        
        # Step 6: Launch application
        logger.info("System ready! Launching application...")
        self.launch_application()
        
        return True

def main():
    """Main entry point"""
    runner = SystemRunner()
    success = runner.run()
    
    if success:
        logger.info("System shutdown complete")
    else:
        logger.error("System startup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
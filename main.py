#!/usr/bin/env python
"""
Main System Coordinator for StockBot Advisor
This is the main entry point that coordinates all components
Author: Anthony Winata Salim
Student Number: 230726051
"""

import sys
import os
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import time
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockBotSystem:
    """Main system coordinator for StockBot Advisor"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.app_file = self.project_root / 'app.py'
        self.init_db_file = self.project_root / 'init_db.py'
        self.test_file = self.project_root / 'test_setup.py'
        self.data_dir = self.project_root / 'data'
        self.logs_dir = self.project_root / 'logs'
        self.models_dir = self.data_dir / 'models'
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for directory in [self.data_dir, self.logs_dir, self.models_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("Checking dependencies...")
        
        required_modules = [
            'streamlit',
            'pandas',
            'numpy',
            'yfinance',
            'plotly',
            'sklearn',
            'yaml',
            'sqlite3'
        ]
        
        missing = []
        for module in required_modules:
            try:
                if module == 'sklearn':
                    __import__('sklearn')
                else:
                    __import__(module)
                logger.info(f"✅ {module} is installed")
            except ImportError:
                missing.append(module)
                logger.error(f"❌ {module} is not installed")
        
        if missing:
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            logger.info("Run: pip install -r requirements.txt")
            return False
        
        return True
    
    def initialize_database(self):
        """Initialize the database"""
        logger.info("Initializing database...")
        
        try:
            # Run init_db.py
            result = subprocess.run(
                [sys.executable, str(self.init_db_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Database initialized successfully")
                return True
            else:
                logger.error(f"Database initialization failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False
    
    def run_tests(self):
        """Run system tests"""
        logger.info("Running system tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.test_file)],
                capture_output=True,
                text=True
            )
            
            print(result.stdout)
            
            if "Setup test complete!" in result.stdout:
                logger.info("✅ All tests passed")
                return True
            else:
                logger.warning("⚠️ Some tests may have failed")
                return False
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def train_models(self, symbols=None):
        """Train ML models for specified symbols"""
        logger.info("Training ML models...")
        
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        try:
            # Add project root to path
            sys.path.insert(0, str(self.project_root))
            
            from utils.ml_models import create_prediction_model
            from utils.data_processor import get_data_processor
            from utils.technical_indicators import TechnicalIndicators
            
            data_processor = get_data_processor()
            results = {}
            
            for symbol in symbols:
                logger.info(f"Training model for {symbol}...")
                
                # Fetch data
                df = data_processor.fetch_stock_data(symbol, period='2y')
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Calculate indicators
                df = TechnicalIndicators.calculate_all_indicators(df)
                
                # Create and train model
                model = create_prediction_model('classification')
                metrics = model.train(df)
                
                if 'error' not in metrics:
                    # Save model
                    model_path = self.models_dir / f'{symbol}_model.pkl'
                    model.save_model(str(model_path))
                    
                    results[symbol] = {
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1_score': metrics.get('f1_score', 0)
                    }
                    
                    logger.info(f"✅ {symbol} model trained: Accuracy={metrics['accuracy']:.2%}")
                else:
                    logger.error(f"Failed to train {symbol}: {metrics['error']}")
            
            # Save results summary
            if results:
                results_file = self.models_dir / 'training_results.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Training results saved to {results_file}")
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def start_streamlit(self, debug=False):
        """Start the Streamlit application"""
        logger.info("Starting StockBot Advisor application...")
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(self.app_file),
                "--server.port", "8501",
                "--server.address", "localhost"
            ]
            
            if not debug:
                cmd.extend([
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"
                ])
            
            logger.info(f"Launching: {' '.join(cmd)}")
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Error starting Streamlit: {e}")
    
    def run_batch_analysis(self, symbols=None):
        """Run batch analysis on multiple stocks"""
        logger.info("Running batch analysis...")
        
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            from utils.data_processor import get_data_processor
            from utils.technical_indicators import TechnicalIndicators
            from utils.ml_models import create_prediction_model
            
            data_processor = get_data_processor()
            results = []
            
            for symbol in symbols:
                logger.info(f"Analyzing {symbol}...")
                
                # Fetch data
                df = data_processor.fetch_stock_data(symbol, period='1mo')
                
                if df.empty:
                    continue
                
                # Calculate indicators
                df = TechnicalIndicators.calculate_all_indicators(df)
                
                # Load or create model
                model_path = self.models_dir / f'{symbol}_model.pkl'
                
                if model_path.exists():
                    model = create_prediction_model('classification')
                    model.load_model(str(model_path))
                else:
                    model = create_prediction_model('classification')
                    model.train(df)
                
                # Get prediction
                prediction = model.predict_next(df)
                
                if 'error' not in prediction:
                    results.append({
                        'symbol': symbol,
                        'signal': prediction.get('signal', 'HOLD'),
                        'confidence': prediction.get('confidence', 0),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"  {symbol}: {prediction['signal']} ({prediction['confidence']*100:.1f}%)")
            
            # Save results
            if results:
                results_file = self.data_dir / f'batch_analysis_{datetime.now():%Y%m%d_%H%M%S}.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Batch analysis results saved to {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return []
    
    def monitor_portfolio(self, interval=300):
        """Monitor portfolio and generate alerts"""
        logger.info(f"Starting portfolio monitor (interval: {interval}s)...")
        
        try:
            sys.path.insert(0, str(self.project_root))
            
            from utils.data_processor import get_data_processor
            from components.alerts import AlertComponents
            
            data_processor = get_data_processor()
            
            while True:
                logger.info("Checking portfolio...")
                
                # Get current prices for watchlist
                watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
                quotes = data_processor.fetch_batch_quotes(watchlist)
                
                if not quotes.empty:
                    # Check for significant moves
                    for _, row in quotes.iterrows():
                        if abs(row['Change%']) > 5:
                            logger.warning(f"Alert: {row['Symbol']} moved {row['Change%']:.2f}%")
                
                logger.info(f"Next check in {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Portfolio monitor stopped by user")
        except Exception as e:
            logger.error(f"Error in portfolio monitor: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='StockBot Advisor System')
    parser.add_argument('command', choices=['run', 'test', 'train', 'init', 'batch', 'monitor'],
                       help='Command to execute')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to process')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--interval', type=int, default=300, help='Monitor interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize system
    system = StockBotSystem()
    
    logger.info("=" * 60)
    logger.info("StockBot Advisor - Financial Advisory System")
    logger.info(f"Author: Anthony Winata Salim (230726051)")
    logger.info(f"Time: {datetime.now()}")
    logger.info("=" * 60)
    
    # Execute command
    if args.command == 'init':
        # Initialize system
        if system.check_dependencies():
            system.initialize_database()
            logger.info("System initialized successfully")
        else:
            logger.error("Please install missing dependencies")
    
    elif args.command == 'test':
        # Run tests
        system.run_tests()
    
    elif args.command == 'train':
        # Train models
        symbols = args.symbols or ['AAPL', 'MSFT', 'GOOGL']
        system.train_models(symbols)
    
    elif args.command == 'run':
        # Run the full application
        if system.check_dependencies():
            system.initialize_database()
            system.start_streamlit(debug=args.debug)
        else:
            logger.error("Cannot start application - missing dependencies")
    
    elif args.command == 'batch':
        # Run batch analysis
        symbols = args.symbols
        results = system.run_batch_analysis(symbols)
        
        # Print summary
        if results:
            logger.info("\n" + "=" * 40)
            logger.info("BATCH ANALYSIS SUMMARY")
            logger.info("=" * 40)
            for r in results:
                logger.info(f"{r['symbol']}: {r['signal']} ({r['confidence']*100:.1f}%)")
    
    elif args.command == 'monitor':
        # Start portfolio monitor
        system.monitor_portfolio(interval=args.interval)
    
    logger.info("=" * 60)
    logger.info("System shutdown complete")
    logger.info("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("StockBot Advisor - Usage Examples:")
        print("  python main.py init      # Initialize system")
        print("  python main.py test      # Run tests")
        print("  python main.py train     # Train ML models")
        print("  python main.py run       # Start application")
        print("  python main.py batch     # Run batch analysis")
        print("  python main.py monitor   # Start portfolio monitor")
        print("\nFor more options: python main.py -h")
    else:
        main()
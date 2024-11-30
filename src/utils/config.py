import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    #Twitter API Credentials
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    
    STOCK_KEYWORDS = [
        'stock market', 
        'NASDAQ', 
        'market analysis',
        'bull market',
        '$AAPL',
        '$GOOGL',
        '$MSFT', 
        '$TSLA',
        'stock trading',
    ]
    
    TRADING_KEYWORDS = [
        'buy stocks',
        'sell stocks',
        'stock price',
        'market trend',
        'trading volume',
        'market sentiment',
        'stock analysis',
        'technical analysis',
        'fundamental analysis'
    ]
    
    #data storage paths
    RAW_DATA_PATH = 'data/raw/'
    PROCESSED_DATA_PATH = 'data/processed/'
    MODELS_PATH = 'data/models/'
    
    #logging path & configuration
    LOG_PATH = 'logs/'
    LOG_LEVEL = 'INFO'
    
    TARGET_STOCKS = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'MSFT': 'Microsoft Corporation',
        'TSLA': 'Tesla Inc.',
        'AMZN': 'Amazon.com Inc.'
    }
    
    SENTIMENT_CATEGORIES = ['positive', 'neutral', 'negative']
    MA_WINDOWS = [3, 7, 14]  #moving average windows(days)
    PRICE_MOVEMENT_THRESHOLD = 0.01  #1% threshold
    
    #model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    #hyperparams
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced'
        }
    }

    SENTIMENT_THRESHOLDS = {
        'strongly_positive': (0.5, 1.0),
        'positive': (0.1, 0.5),
        'neutral': (-0.1, 0.1),
        'negative': (-0.5, -0.1),
        'strongly_negative': (-1.0, -0.5)
    }
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    #Twitter API Credentials
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    
    STOCK_KEYWORDS = [
        'stock market', 
        'NASDAQ', 
        'NYSE', 
        'trading stocks',
        '$AAPL',
        '$GOOGL',
        '$MSFT', 
        '$TSLA',
        '$AMZN',
        'bull market',
        'bear market',
        'market analysis',
        'stock trading',
        'investment strategy'
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
    
    #Twitter API Settings (just incase)
    MAX_TWEETS_PER_REQUEST = 100
    MAX_TOTAL_TWEETS = 1000
    TWEET_LANG = 'en'
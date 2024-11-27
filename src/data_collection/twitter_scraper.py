import tweepy
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os
import csv
from textblob import TextBlob
import re

from src.utils.config import Config
from src.utils.logger import setup_logger

class TwitterScraper:
    def __init__(self):
        self.logger = setup_logger('twitter_scraper') #twitter_scraper's logger.
        
        try:
            # using only bearer_token for now (read-only permissions, coz thats all we need)
            self.client = tweepy.Client(
                bearer_token=Config.TWITTER_BEARER_TOKEN,
                wait_on_rate_limit=True
            )
            self.logger.info("Twitter API authentication successful")
        
        except Exception as e:
            self.logger.error(f"Twitter API authentication failed: {e}")
            raise

    def clean_tweet(self, tweet: str) -> str:
        """
        cleans tweet text by removing links, special characters.
        
        args: 
            tweet: Raw tweet text
        """
        cleaned_text = ' '.join(
            re.sub(
                r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", 
                " ", 
                str(tweet)
            ).split()
        )
        return cleaned_text

    def get_tweet_sentiment(self, tweet: str) -> str:
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def collect_tweets(
        self,
        keywords: Optional[List[str]] = None,
        max_results: int = 100,
        limit: int = 1000
    ) -> List[Dict]:
        """
        main function to collect tweets using twitter API.
        returns list of collected tweets with metadata
        """
        if keywords is None:
            keywords = Config.STOCK_KEYWORDS

        collected_tweets = []
        
        try:
            for keyword in keywords:
                search_query = f"{keyword} lang:en -is:retweet"  #for english tweets only & no retweets
                self.logger.info(f"Searching tweets for query: {search_query}")
                
                tweet_paginator = tweepy.Paginator(
                    self.client.search_recent_tweets,
                    query=search_query,
                    tweet_fields=['created_at', 'public_metrics', 'lang'],
                    max_results=max_results
                ).flatten(limit=limit)

                for tweet in tweet_paginator:
                    try:
                        #processing each tweet
                        cleaned_text = self.clean_tweet(tweet.text)
                        sentiment = self.get_tweet_sentiment(cleaned_text)
                        
                        tweet_data = {
                            'id': tweet.id,
                            'created_at': tweet.created_at,
                            'text': tweet.text,
                            'cleaned_text': cleaned_text,
                            'sentiment': sentiment,
                            'keyword': keyword,
                            'likes': tweet.public_metrics['like_count'] if hasattr(tweet, 'public_metrics') else 0,
                            'retweets': tweet.public_metrics['retweet_count'] if hasattr(tweet, 'public_metrics') else 0,
                        }
                        collected_tweets.append(tweet_data)
                    
                    except tweepy.TooManyRequests:
                        self.logger.error("Rate limit exceeded. Stopping collection early.")
                        raise  # Propagate the exception to trigger partial return
                
                self.logger.info(f"Collected {len(collected_tweets)} tweets for keyword: {keyword}")

        except tweepy.TooManyRequests:
            self.logger.error("Rate limit exceeded while collecting tweets.")
        except Exception as e:
            self.logger.error(f"Error collecting tweets: {e}")
        
        if collected_tweets:
            self.logger.info("Returning partially collected tweets")
        return collected_tweets

    def save_tweets(
        self, 
        tweets: List[Dict], 
        filename: Optional[str] = None,
        include_sentiment_stats: bool = True
    ):
        """
        takes in the 'collected tweets' & saves them in csv file
        """
        if not tweets:
            self.logger.warning("No tweets to save")
            return
            
        if filename is None:
            filename = f'tweets_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
        filepath = os.path.join(Config.RAW_DATA_PATH, filename)
        
        try:
            df = pd.DataFrame(tweets)
            df.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.info(f"Tweets saved to {filepath}")
            
            #calculating sentiment stats (just to print and understand better)
            if include_sentiment_stats:
                total_tweets = len(tweets)
                sentiment_counts = df['sentiment'].value_counts()
                
                self.logger.info("\nSentiment Analysis Results:")
                self.logger.info(f"Total Tweets Analyzed: {total_tweets}")
                for sentiment, count in sentiment_counts.items():
                    percentage = (count / total_tweets) * 100
                    self.logger.info(f"{sentiment.capitalize()} tweets: {count} ({percentage:.2f}%)")
                
        except Exception as e:
            self.logger.error(f"Error saving tweets: {e}")
            raise

def main():
    try:
        scraper = TwitterScraper()
        
        stock_keywords = [
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
        
        tweets = scraper.collect_tweets(
            keywords=stock_keywords,
            max_results=100,  #tweets per request
            limit=100         #total tweets to collect
        )
        
        scraper.save_tweets(tweets, include_sentiment_stats=True)
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()

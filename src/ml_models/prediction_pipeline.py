import joblib
import numpy as np
import pandas as pd
import glob
import os

from src.data_collection.data_preprocessor import DataPreprocessor
from src.data_collection.feature_extractor import FeatureExtractor
from src.sentiment_analysis.sentiment_analyzer import AdvancedSentimentAnalyzer
from src.utils.config import Config
from src.utils.logger import setup_logger

class StockSentimentPredictor:
    def __init__(self):
        self.logger = setup_logger('prediction_pipeline')
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        
        #using the model and label encoder (saved from model training)
        self.model = joblib.load(Config.MODELS_PATH + '/stock_sentiment_model.joblib')
        self.label_encoder = joblib.load(Config.MODELS_PATH + '/label_encoder.joblib')

    def predict_sentiment(self, new_tweets_df):
        #to predict sentiment for new tweets

        new_tweets_df = self.preprocessor.process_raw_data()  # Ensure required columns are created
        
        text_features = self.feature_extractor.extract_text_features(new_tweets_df)
        metadata_features = self.feature_extractor.extract_metadata_features(new_tweets_df)
        
        new_tweets_df = self.sentiment_analyzer.enhance_sentiment_features(new_tweets_df)  # Update DataFrame
        sentiment_features = new_tweets_df[['vader_compound', 'textblob_polarity']].values
        
        combined_features = np.hstack([
            text_features, 
            metadata_features,
            sentiment_features
        ])
        
        predictions = self.model.predict(combined_features)
        pred_labels = self.label_encoder.inverse_transform(predictions)
        
        new_tweets_df['predicted_sentiment'] = pred_labels
        
        return new_tweets_df


def main():
    predictor = StockSentimentPredictor()
    
    #loading recent tweets for prediction
    recent_tweets = pd.read_csv(max(
        glob.glob('data/raw/tweets_*.csv'), 
        key=os.path.getctime
    ))
    
    results = predictor.predict_sentiment(recent_tweets)
    
    os.makedirs('data/predictions', exist_ok=True)
    
    #save predictions
    results.to_csv('data/predictions/sentiment_predictions.csv', index=False)

if __name__ == "__main__":
    main()  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data_collection.feature_extractor import FeatureExtractor
from src.sentiment_analysis.sentiment_analyzer import AdvancedSentimentAnalyzer

class StockSentimentModelTrainer:
    def __init__(self):
        self.logger = setup_logger('model_trainer')
        self.feature_extractor = FeatureExtractor()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()

    def prepare_training_data(self):

        #extract features
        features_dict = self.feature_extractor.extract_features()
        
        #adding sentiment analysis features
        processed_df = features_dict['original_df']
        enhanced_df = self.sentiment_analyzer.enhance_sentiment_features(processed_df)
        
        #combine features
        combined_features = np.hstack([
            features_dict['combined_features'],
            enhanced_df[['vader_compound', 'textblob_polarity']].values
        ])
        
        #prepare labels
        le = LabelEncoder()
        labels = le.fit_transform(enhanced_df['advanced_sentiment'])
        
        return combined_features, labels, le

    def train_model(self):
        #using Random Forest Classifier for stock sentiment pred.

        X, y, label_encoder = self.prepare_training_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE, 
            random_state=Config.RANDOM_STATE
        )
        
        model = RandomForestClassifier(**Config.MODEL_PARAMS['random_forest'])
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        self.logger.info("Model Evaluation Metrics:")
        self.logger.info("\nClassification Report:\n" + 
            classification_report(y_test, y_pred, 
                target_names=label_encoder.classes_)
        )
        
        os.makedirs(Config.MODELS_PATH, exist_ok=True)
        
        model_path = os.path.join(Config.MODELS_PATH, 'stock_sentiment_model.joblib')
        encoder_path = os.path.join(Config.MODELS_PATH, 'label_encoder.joblib')
        
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Label encoder saved to {encoder_path}")
        
        return model, label_encoder

def main():
    trainer = StockSentimentModelTrainer()
    trainer.train_model()

if __name__ == "__main__":
    main()
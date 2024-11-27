import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data_collection.data_preprocessor import DataPreprocessor

class FeatureExtractor:
    def __init__(self):
        self.logger = setup_logger('feature_extractor')
        self.preprocessor = DataPreprocessor()
    
    # we are going to extract text features from preprocessed text using TF-IDF
    # we will instantiate vectorizer with params, and then "fit_transform" it with "processed_text" column
    def extract_text_features(
        self, 
        processed_df: pd.DataFrame, 
        max_features: int = 100
    ) -> np.ndarray:
        
        # args:
            # processed_df: loading preprocessed DataFrame
            # max_features: aaximum number of TF-IDF features
        
        # output: TF-IDF feature matrix

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        
        text_features = vectorizer.fit_transform(processed_df['processed_text']).toarray()
        self.logger.info(f"Extracted {text_features.shape[1]} text features")
        
        return text_features

    # metadata extraction (generic cols like text_len, likes, retweets)
    # we will instantiate standardscaler, and then "fit_transform" it with metadata columns
    def extract_metadata_features(
        self, 
        processed_df: pd.DataFrame
    ) -> np.ndarray:
        
        # args:
            # processed_df: Preprocessed DataFrame
        
        # output: scaled metadata feature matrix

        metadata_features = processed_df[['text_length', 'likes', 'retweets']].values
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(metadata_features)
        
        self.logger.info(f"Scaled {scaled_features.shape[1]} metadata features")
        
        return scaled_features

    #combining the two - text features and metadata features (from previous two functions)
    def combine_features(
        self, 
        text_features: np.ndarray, 
        metadata_features: np.ndarray
    ) -> np.ndarray:
        
        # args:
            # text_features: TF-IDF text features
            # metadata_features: scaled metadata features
        
        # output: Combined feature matrix

        combined_features = np.hstack([text_features, metadata_features])
        self.logger.info(f"Combined feature matrix shape: {combined_features.shape}")
        
        return combined_features

    #IMP fn.. MAIN function that ties all the above functions together in pipeline flow
    #it also ties data_preprocessor code, thus executing that file too first
    def extract_features(
        self, 
        processed_df: Optional[pd.DataFrame] = None,
        max_text_features: int = 100
    ) -> Dict[str, np.ndarray]:
        
        # output: dictionary of extracted features

        if processed_df is None:
            processed_df = self.preprocessor.process_raw_data()
        
        # Check if DataFrame is empty
        if processed_df.empty:
            self.logger.warning("No data to extract features from")
            return {}
        
        #extract_text_features
        text_features = self.extract_text_features(
            processed_df, 
            max_features=max_text_features
        )
        
        #extract_metadata_features
        metadata_features = self.extract_metadata_features(processed_df)
        
        #combine_features
        combined_features = self.combine_features(text_features, metadata_features)
        
        #preparing labels (for model training)
        labels = processed_df['sentiment_encoded'].values
        
        #& save features for later use (for model training)
        features_dict = {
            'text_features': text_features,
            'metadata_features': metadata_features,
            'combined_features': combined_features,
            'labels': labels,
            'original_df': processed_df
        }
        
        return features_dict

def main():
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features()

if __name__ == "__main__":
    main()
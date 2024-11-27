import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import re
import nltk

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"NLTK download warning: {e}")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from src.utils.config import Config
from src.utils.logger import setup_logger

class DataPreprocessor:
    def __init__(self):
        self.logger = setup_logger('data_preprocessor')
        
        #set up stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.logger.warning("Could not load stopwords. Using minimal stop words list.")
            self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves'}

    # to load all CSV files from the raw data directory.
    # returns concatenated DataFrame of all raw tweet data.
    def load_raw_data(self, directory: Optional[str] = None) -> pd.DataFrame:

        if directory is None:
            directory = Config.RAW_DATA_PATH
        
        all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
        
        if not all_files:
            self.logger.warning("No CSV files found in the raw data directory")
            return pd.DataFrame()
        
        try:
            df_list = [pd.read_csv(file) for file in all_files]
            combined_df = pd.concat(df_list, ignore_index=True)
            
            self.logger.info(f"Loaded {len(all_files)} files with total {len(combined_df)} tweets")
            return combined_df
        
        except Exception as e:
            self.logger.error(f"Error loading raw data: {e}")
            return pd.DataFrame()

    # main preprocessing fn. with fallback tokenization
    def preprocess_text(self, text: str) -> str:
        # args:
            # text: input tweet text
        
        # output: cleaned and preprocessed text

        # Handle potential NaN or None values
        if not isinstance(text, str):
            return ""
        
        #converting to lowercase
        text = text.lower()
        
        #removing URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        #removing user mentions (since there are many, in the tweets)
        text = re.sub(r'@\w+', '', text)
        
        #remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        #simple split-based tokenization as a fallback
        try:
            #first trying with nltk word tokenizer
            tokens = word_tokenize(text)
        except Exception:
            #if fails, fallback to simple splitting
            tokens = text.split()
        
        #removing stopwords
        tokens = [token for token in tokens if token and token not in self.stop_words]
        
        return ' '.join(tokens)

    #implements drop duplicates fillna, and then uses "safe_preprocess_text" to process raw tweet data
    def process_raw_data(self, save_processed: bool = True) -> pd.DataFrame:        
        # args:
            # save_processed: whether to save processed data to csv
        # output: processed DataFrame

        raw_df = self.load_raw_data()
        
        if raw_df.empty:
            self.logger.warning("No data to process")
            return raw_df
        
        #dropping duplicates
        raw_df.drop_duplicates(subset=['id'], inplace=True)
        
        if 'text' not in raw_df.columns:
            self.logger.error("No 'text' column found in the DataFrame")
            return pd.DataFrame()
        
        #filling NaN values in text column
        raw_df['text'] = raw_df['text'].fillna('')
        
        #using "safe_preprocess_text" fn.
        raw_df['processed_text'] = raw_df['text'].apply(self.safe_preprocess_text)
        
        raw_df['text_length'] = raw_df['text'].str.len()
        
        #encode sentiment (to -1,0,1)
        sentiment_mapping = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        raw_df['sentiment_encoded'] = raw_df['sentiment'].map(sentiment_mapping)
        
        self.logger.info(f"Preprocessing complete. Total tweets: {len(raw_df)}")
        self.logger.info("Sentiment distribution:")
        self.logger.info(raw_df['sentiment'].value_counts())
        
        if save_processed:
            os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
            processed_file = os.path.join(
                Config.PROCESSED_DATA_PATH, 
                f'processed_tweets_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            raw_df.to_csv(processed_file, index=False)
            self.logger.info(f"Processed data saved to {processed_file}")
        
        return raw_df
    
    # just implements "preprocess_text" function, but with exception/error handling (acts as a wrapper function)
    def safe_preprocess_text(self, text: str) -> str:
        # args:
            # text: Input tweet text
        # output:
            # preprocessed text or empty string

        try:
            return self.preprocess_text(text)
        except Exception as e:
            self.logger.warning(f"Error preprocessing text: {e}. Returning empty string.")
            return ""

def main():
    #downloading nltk resources
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"NLTK download error: {e}")
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_raw_data()
    
    if not processed_data.empty:
        print(f"Processed {len(processed_data)} tweets")
    else:
        print("No data processed")

if __name__ == "__main__":
    main()
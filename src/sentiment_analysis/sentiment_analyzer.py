import pandas as pd
import numpy as np
from typing import Dict, List
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# I researched online, found out "vader" can be of good use to improve sentiment analysis for our tweet data
# I referred (to understand the basics): https://medium.com/@rslavanyageetha/vader-a-comprehensive-guide-to-sentiment-analysis-in-python-c4f1868b0d2e
class AdvancedSentimentAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    # using vader to analyze sentiments - output: dict with sentiment scores
    def analyze_sentiment(self, text: str) -> Dict[str, float]:

        # textBlob sentiment
        blob_sentiment = TextBlob(text).sentiment
        
        # VADER sentiment
        # takes a piece of text as input and returns a dictionary containing the sentiment scores for the text. 
        # The dictionary contains four keys: neg, neu, pos, and compound.
        vader_sentiment = self.sia.polarity_scores(text)
        
        return {
            'textblob_polarity': blob_sentiment.polarity,
            'textblob_subjectivity': blob_sentiment.subjectivity,
            'vader_pos': vader_sentiment['pos'],
            'vader_neg': vader_sentiment['neg'],
            'vader_neu': vader_sentiment['neu'],
            'vader_compound': vader_sentiment['compound']
        }

    #IMP fn.. will use in other files
    # to apply adv sentiment analysis on input dataframe (particularly "processed_text" column) - using prev fn.
    # and then add the sentiment features to same dataframe (output)
    def enhance_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:

        sentiment_features = df['processed_text'].apply(self.analyze_sentiment).apply(pd.Series)
        
        enhanced_df = pd.concat([df, sentiment_features], axis=1)
        
        # Create categorical sentiment based on multiple metrics
        enhanced_df['advanced_sentiment'] = enhanced_df.apply(
            lambda row: self._categorize_sentiment(
                row['vader_compound'], 
                row['textblob_polarity']
            ), 
            axis=1
        )
        
        return enhanced_df

    def _categorize_sentiment(self, vader_score: float, textblob_score: float) -> str:

        #categorizing the sentiment based on the value of "polarity_scores"
        if vader_score >= 0.05 and textblob_score > 0:
            return 'strongly_positive'
        elif vader_score > 0 and textblob_score > 0:
            return 'positive'
        elif vader_score >= -0.05 and vader_score <= 0.05:
            return 'neutral'
        elif vader_score < 0 and textblob_score < 0:
            return 'negative'
        else:
            return 'strongly_negative'

def main():
    from src.data_collection.data_preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.process_raw_data()
    
    sentiment_analyzer = AdvancedSentimentAnalyzer()
    enhanced_df = sentiment_analyzer.enhance_sentiment_features(processed_df)
    
    print(enhanced_df['advanced_sentiment'].value_counts())

if __name__ == "__main__":
    main()
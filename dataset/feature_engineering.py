import re
import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

class FeatureExtractor:
    """
    Feature extractor for extracting VADER sentiment scores and first-person pronoun usage frequency
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # List of first-person singular pronouns
        self.first_person_singular_pronouns = ['i', 'me', 'my', 'mine', 'myself']
        
    def extract_features(self, texts):
        """
        Extract features from a list of texts
        
        Args:
            texts: List of texts
            
        Returns:
            features_df: DataFrame containing extracted features
        """
        features = []
        
        for text in texts:
            # Extract VADER sentiment scores
            sentiment_scores = self.extract_sentiment(text)
            
            # Extract first-person pronoun usage frequency
            pronoun_freq = self.extract_pronoun_frequency(text)
            
            # Combine features
            text_features = {**sentiment_scores, **pronoun_freq}
            features.append(text_features)
            
        return pd.DataFrame(features)
    
    def extract_sentiment(self, text):
        """
        Extract sentiment scores using VADER
        
        Args:
            text: Input text
            
        Returns:
            dict: Dictionary containing sentiment scores
        """
        scores = self.vader.polarity_scores(text)
        return {
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
            'vader_compound': scores['compound']
        }
    
    def extract_pronoun_frequency(self, text):
        """
        Extract first-person singular pronoun usage frequency
        
        Args:
            text: Input text
            
        Returns:
            dict: Dictionary containing pronoun usage frequency
        """
        # Preprocess text: lowercase and tokenize
        text = text.lower()
        
        # 简单分词，避免使用nltk.word_tokenize
        tokens = re.findall(r'\b\w+\b', text)
        
        # Calculate pronoun frequency
        total_words = len(tokens)
        if total_words == 0:
            return {'first_person_singular_freq': 0.0}
        
        # Count the occurrences of first-person pronouns
        fp_count = sum(1 for token in tokens if token in self.first_person_singular_pronouns)
        
        # Calculate frequency
        fp_freq = fp_count / total_words
        
        return {'first_person_singular_freq': fp_freq}
    
    def preprocess_text(self, text):
        """
        Preprocess text: lowercase, remove URLs and special characters
        
        Args:
            text: Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert text to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters, but keep punctuation
        text = re.sub(r'[^\w\s\.,!?]', '', text)
        
        return text
    
    def process_dataframe(self, df, text_column='text'):
        """
        Process DataFrame, extract features and add to original DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
            
        Returns:
            DataFrame: DataFrame with added features
        """
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in df[text_column]]
        
        # Extract features
        features_df = self.extract_features(preprocessed_texts)
        
        # Add features to original DataFrame
        result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        return result_df

if __name__ == "__main__":
    # Test feature extractor
    extractor = FeatureExtractor()
    test_texts = [
        "I am feeling very sad and depressed today. I hate myself.",
        "Today is a good day! I enjoyed my time with friends."
    ]
    features = extractor.extract_features(test_texts)
    print(features) 
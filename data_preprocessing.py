import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the raw data
raw_data = pd.read_csv('insurance_claims.csv')

# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    
    return text

# Apply preprocessing to the text columns
raw_data['original_text_cleaned'] = raw_data['original_text'].apply(preprocess_text)
raw_data['updated_text_cleaned'] = raw_data['updated_text'].apply(preprocess_text)

# Save the cleaned data
raw_data.to_csv('insurance_claims_cleaned.csv', index=False)

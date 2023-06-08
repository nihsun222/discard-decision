import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import numpy as np

# Load the cleaned training data
train_data = pd.read_csv('insurance_claims_cleaned.csv')

# Convert cleaned text to list of word tokens
original_documents = train_data['original_text_cleaned'].apply(lambda x: word_tokenize(x))
updated_documents = train_data['updated_text_cleaned'].apply(lambda x: word_tokenize(x))

# Train Word2Vec model on the original and updated documents
model = Word2Vec(original_documents + updated_documents, size=200, window=5, min_count=1, workers=4)

# Encode the target variable
encoder = LabelEncoder()
train_data['discard_decision_encoded'] = encoder.fit_transform(train_data['discard_decision'])

# Generate Word2Vec embeddings for the original and updated documents
original_embeddings = train_data['original_text_cleaned'].apply(lambda x: model[x.split()].mean(axis=0))
updated_embeddings = train_data['updated_text_cleaned'].apply(lambda x: model[x.split()].mean(axis=0))

# Perform additional feature engineering
train_data['original_text_length'] = train_data['original_text_cleaned'].apply(lambda x: len(x))
train_data['updated_text_length'] = train_data['updated_text_cleaned'].apply(lambda x: len(x))
train_data['text_length_difference'] = train_data['updated_text_length'] - train_data['original_text_length']

# Calculate cosine similarity between original and updated embeddings
cosine_similarities = original_embeddings.apply(lambda x: x.dot(updated_embeddings.T) / (np.linalg.norm(x) * np.linalg.norm(updated_embeddings, axis=1)))

# Save the features and embeddings
train_data[['original_text_length', 'updated_text_length', 'text_length_difference']].to_csv('text_features.csv', index=False)
pd.DataFrame(cosine_similarities.tolist()).to_csv('cosine_similarities.csv', index=False)

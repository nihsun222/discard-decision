import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load the cleaned data
cleaned_data = pd.read_csv('insurance_claims_cleaned.csv')

# Convert cleaned text to list of word tokens
original_documents = cleaned_data['original_text_cleaned'].apply(lambda x: word_tokenize(x))
updated_documents = cleaned_data['updated_text_cleaned'].apply(lambda x: word_tokenize(x))

# Train Word2Vec model on the original and updated documents
model = Word2Vec(original_documents + updated_documents, size=200, window=5, min_count=1, workers=4)

# Save the trained Word2Vec model
model.save('word2vec.model')

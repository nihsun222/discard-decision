import pandas as pd
from gensim.models import Word2Vec

# Load the cleaned training data
train_data = pd.read_csv('insurance_claims_cleaned.csv')

# Load the Word2Vec model
model = Word2Vec.load('word2vec.model')

# Fill in missing factors and importance rankings
for index, row in train_data.iterrows():
    if pd.isnull(row['factors']):
        # Fill in missing factors based on other information in the row
        factors = row['original_text_cleaned'] + ' ' + row['updated_text_cleaned']
        train_data.at[index, 'factors'] = factors
    
    if pd.isnull(row['importance_rankings']):
        # Fill in missing importance rankings using the trained model
        importance_ranking = model.predict([row['updated_text_cleaned']])[0]
        train_data.at[index, 'importance_rankings'] = importance_ranking

# Save the updated data
train_data.to_csv('insurance_claims_updated.csv', index=False)

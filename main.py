import pandas as pd
from data_preprocessing import preprocess_text
from word2vec_training import train_word2vec_model
from feature_engineering import perform_feature_engineering
from fill_missing_data import fill_missing_data

def main(dataset_path):
    # Load the dataset
    dataset = pd.read_csv(dataset_path)
    
    # Preprocess the text data
    dataset['original_text_cleaned'] = dataset['original_text'].apply(preprocess_text)
    dataset['updated_text_cleaned'] = dataset['updated_text'].apply(preprocess_text)
    
    # Train the Word2Vec model
    train_word2vec_model(dataset)
    
    # Perform feature engineering
    perform_feature_engineering(dataset)
    
    # Fill in missing factors and importance rankings
    fill_missing_data(dataset)
    
    # Save the processed dataset
    processed_dataset_path = dataset_path.replace('.csv', '_processed.csv')
    dataset.to_csv(processed_dataset_path, index=False)
    print(f"Processed dataset saved to: {processed_dataset_path}")

if __name__ == '__main__':
    dataset_path = 'path/to/your/dataset.csv'
    main(dataset_path)

# Insurance Claims Processing

This project is a machine learning model to automate the processing of insurance claim documents. The goal is to compare updated claim documents with the original documents and determine whether to discard the updated version or keep it based on textual and semantic differences.

This project involves uses a training and test dataset consisting of labeled original documents and updated documents. The dataset incorporates human-generated labels. The first label indicates whether the human reviewer decided to discard the updated document or keep it. In cases where the reviewer chose to discard the updated document, the reasons behind their decision are elaborated upon in the next label. Lastly, the dataset includes the reviewer's rankings, ranging from 0 to 10, which assess the significance of each contributing factor in making the decision to discard. If the reviewer decided to keep the document, the reasons and rankings labels are left null.

The project is organized into the following files:

- `data_preprocessing.py`: Performs data cleaning and preprocessing on the raw insurance claims data.
- `word2vec_training.py`: Trains a Word2Vec model on the preprocessed insurance claims data.
- `feature_engineering.py`: Performs feature engineering, including generating embeddings and additional features for the machine learning model.
- `fill_missing_data.py`: Fills in missing factors and importance rankings in the training data by analyzing other information in the row.

Install required dependencies by running pip install -r requirements.txt

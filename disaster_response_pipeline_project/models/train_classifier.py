import sys
# import libraries

import re
import os
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


def load_data(database_filepath):
    """
    Load and preprocess data from a SQLite database.

    Args:
        database_filepath (str): The file path to the SQLite database.

    Returns:
        X (Series): A pandas Series containing the feature data (messages).
        y (DataFrame): A pandas DataFrame containing the label data.
        category (Index): An Index object containing the names of the output categories (labels) in the dataset.
        
    """
  
    #Get the base name of the file (last part of the path)
    base_name = os.path.basename(database_filepath)
    # Split the base name to remove the file extension
    table_name, _ = os.path.splitext(base_name)
    # read in file
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, engine)  
    #clean data
    df = df.dropna()
    #load to database
    df.to_sql('table_name', engine, if_exists='replace', index=False)
    #define features and label arrays
    X = df['message']
    y = df[df.columns[4:]]
    category = y.columns
    return X, y, category

def tokenize(text):
    """
    Tokenize and lemmatize the input text.
    
    Args:
        text(str): The input text to be tokenized and lemmatized.
    
    Returns:
        clean_tokens(list): The list of cleaned and lemmatized tokens.
    """
    tokens =word_tokenize(text)
    lemmatizer =WordNetLemmatizer()
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model():
    """Build a machine learning model pipeline for text classification.

     Args:
        None: The function does not take any arguments.

    Returns:
        model (GridSearchCV): A configured GridSearchCV object for model fitting and 
        hyperparameter tuning.
    """
    
    #model pipeline
    pipeline = Pipeline([
       ('vect', CountVectorizer()),
       ('tfidf', TfidfTransformer()),
       ('clf', MultiOutputClassifier(RandomForestClassifier()))
       ])
    #define parameters for GridSearchCV
    parameters = {
                      
               
                'clf__estimator__max_depth': [5, 10, 15],
                'clf__estimator__min_samples_split': [2, 5, 10]
                              
    }
 
    # Create the grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=10)
  
    model=cv
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """Report the f1 score, precision and recall for each output category of the dataset
   
    Args:
        model (Pipeline): A trained machine learning pipeline that includes preprocessing and model steps.
        X_test (DataFrame): A DataFrame containing the features of the test dataset.
        y_test (DataFrame): A DataFrame containing the true labels for the test dataset, with each column 
                        representing a different target variable.

    Returns:
    None: The function prints the classification report for each target variable in the test set.
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # output model test results
    for i, column in enumerate(category_names):
        # Make predictions on the test set
        y_pred_column = y_pred[:, i]
        # Calculate the classification report
        print(f"**Classification report for column '{column}':**")
        print(f"")
        print(classification_report(Y_test[column], y_pred_column))
        print(f"```")
        print()


def save_model(model, model_filepath):
    """
    Save the trained model to a specified file path as a pickle file.

    Args:
        model (object): The trained machine learning model to be saved.
        model_filepath (str): The file path where the model will be saved, including the filename and .pkl extension.

    Returns:
        None: The function saves the model to the specified file path.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

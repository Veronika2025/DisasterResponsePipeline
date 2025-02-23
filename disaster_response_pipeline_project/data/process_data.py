import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from CSV files and merges them.

    Args:
    messages_filepath (str): The file path to the CSV file containing messages.
    categories_filepath (str): The file path to the CSV file containing categories.

    Returns:
    DataFrame: A pandas DataFrame containing the merged data from messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages.merge(categories_df, how="outer",on=["id"])
    
    return df



def clean_data(df):
    """
    Clean and transform the input DataFrame containing disaster response data.

    
    Parameters:
    df (DataFrame): A pandas DataFrame containing disaster response messages and categories.

    Returns:
    DataFrame: A cleaned pandas DataFrame with the original data and individual category columns,
                where each category column contains binary values.
    """
      
    
    # create a dataframe of the 36 individual category columns
    categories_df = df["categories"].str.split(";", expand = True)
    row = categories_df.values[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    # rename the columns of `categories`
    categories_df.columns = category_colnames
    
    for column in categories_df:
    # set each value to be the last character of the string
        categories_df[column] = categories_df[column].astype(str).str[-1]
    # convert column from string to numeric
        categories_df[column] = categories_df[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop("categories", axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_df],axis=1)
    # drop duplicates
    df = df.drop_duplicates()
        
    return df


def save_data(df, database_filename):
    # save a dataframe to sqlite database
    print('Save {} to {} database: '.format(df, database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

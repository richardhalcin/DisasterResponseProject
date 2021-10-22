import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads dataframe from messages and categories files

    Args:
        messages_filepath: path to messages file
        categories_filepath: path to categories file
    Returns:
        df: dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath) 
    
    # merge datasets into dataframe
    df = pd.merge(messages, categories, on=['id'], how='inner')

    return df

def clean_data(df):
    """Cleans dataframe, drop NaN values and remove duplicates

        Args:
            df: pandas dataframe on which we are operating
        Returns:
            df: cleaned dataframe
        """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    # get list of category names
    category_colnames = []
    for col in row:
        category_colnames.append(col[0:-2])
    # rename categories columns
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    ### remove non binary values from related column
    df = df[df['related'] != 2]
    
    return df

def save_data(df, database_filename):
    """Saves dataframe to database with specified name

        Args:
            df: pandas dataframe on which we are operating
            database_filename: name of database
        Returns:
            None
        """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('dataframe', engine, index=False, if_exists='replace')


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
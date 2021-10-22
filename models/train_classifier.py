import sys
import nltk
nltk.download(['punkt','wordnet'])
nltk.download('stopwords')

import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """Loads dataframe from database

    Args:
        database_filepath: path to database
    Returns:
        X: dataframe of messages
        Y: dataframe of categories
        category_names: list of category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('dataframe', con=engine)
    print(df.groupby('genre').count()['message'])
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    category_names = []
    for col_name in Y.columns: 
        category_names.append(col_name)
    
    return X, Y, category_names


def tokenize(text):
    """Tokenize and lemmatize and normalize input text

    Args:
        text: input text
    Returns:
        clean_tokens: list of tokenized, lemmatized, normalized tokens
    """
    tokens = word_tokenize(text)
    # remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds model. Pipeline of CountVectorizer, TFidf transformer and MultiOutputClassifier of Random Forest

    Returns:
        model: Model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__max_df': (0.5, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': (50, 100),
    }

    model = GridSearchCV(pipeline, param_grid= parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints models stats

        Args:
            y_test: category values of test dataset
            y_pred: predicted values
        Returns:
            None
        """

    y_pred = model.predict(X_test)

    i = 0
    for col in Y_test:
        report = classification_report(Y_test[col], y_pred[:, i])
        print("Classification report:\n", report)
        i = i + 1
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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
# DisasterResponseProject
Project for DataScience training, which uses Machine learning for disaster messages classification.

## ETL part:
- Loads data from 2 csv files. One of them consists of messages and the second one has categories. The next step is cleaning data which consist dropping duplicates, converting values to binary 1, 0. At the end dataframe is saved in database.0

## ML part:
- Loads data from database file. Then data are tokenized, normalized and removed stop words. The next step is lemmatization and stemming.
- I am using TF-idf algorithm for counting weights of words. For classification is used Random forest classificator.
- Then trained model is evaluated and saved locally.

## Web part:
- In this part user can see 2 graphs. The first one is showing number of messages in each category. The second one is showing number of medical help messages vs other.
- User can write his own input and see in which category it belongs.
- There is also a link to project's github repository.

## Requirments:
- Python 3.x
- Libraries: numpy, nltk, pandas, sqlalchemy, sklearn, json, plotly, flask, joblib.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
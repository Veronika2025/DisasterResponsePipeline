# DisasterResponsePipeline

### Project Summary:
This project is part of the Udacity Data Science Nonadegree Class. The aim is to classifiy disaster messages using a machine learning pipeline. The project result will be a web app, which emergency workers can use to classify messages into several given categories. The web app will also provide data visualization.  


### Data:
The project data is provided by Appen. 
The dataset contains two data files: (1) disaster messages and (2) disaster categories.  


### Project Files:
#### process_data.py:
The ETL script takes the file paths of the two datasets, cleanse the datasets and stores it into a SQLite database in the specified database file path.

#### train_classifier.py: 
The machine learning script loads the data from the SQLite database, builds a text processing machine learning pipeline to classify the messages into 36 categories and exports the final model as a pickle file. 

#### run.py: 
This is the main file, which runs the Flask app. The app includes data visualization and the option to inpute messages for classification. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

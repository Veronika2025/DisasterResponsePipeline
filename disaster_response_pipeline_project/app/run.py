import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # Graphic 2: Sorted Count of Categories
    category_counts = df.iloc[:,4:].sum()
    # Sort the category counts in descending order
    sorted_category_counts = category_counts.sort_values(ascending=False)
    category_names = list(sorted_category_counts.index)
    # list of sorted counts
    sorted_counts = list(sorted_category_counts)
    
    
    # Graphic 3: percentage of top 5 categories 
    # Get the top 5 categories and respective counts
    top_5_categories = sorted_category_counts.head(5).index.tolist() 
    top_5_counts = sorted_category_counts.head(5).values
    # Ensure the counts are numeric
    top_5_counts = pd.to_numeric(top_5_counts, errors='coerce')
    # Calculate the total count of all categories
    total_count = category_counts.sum()
    # Ensure total_count is numeric
    total_count = pd.to_numeric(total_count, errors='coerce')
    # Calculate the percentage for each of the top 5 categories
    top_5_percentages = (top_5_counts / total_count) * 100



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Graphic 2: 
        {
            'data': [
                Bar(
                    x=category_names,
                    y=sorted_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        # Graphic 3: 
        {
            'data': [
                Bar(
                    x=top_5_categories,
                    y=top_5_percentages,
                    text=[f'{percentage:.2f}%' for percentage in top_5_percentages],  # Format percentages
            textposition='auto'  # Automatically position the text on the bars
                )
            ],

            'layout': {
                'title': 'Percentage of top 5 Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        } 

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
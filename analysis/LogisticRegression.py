import csv
import json
import pandas as pd 
import sys, getopt, pprint
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

client = MongoClient(uri)
db = client.get_database()
db_cases = db['cases']

def mongo_to_df():
    """
    Converts db_cases to a pandas df
    :return: pandas df
    """
    df = pd.DataFrame(list(db_cases.find()))
    return df

    """
    # helpful other commands to look at data
    print(db_cases.find_one()) # prints one case from the mongo db
    print(df.iloc[0]) # prints the 0th row from the pandas df
    """

def return_statute_code_features():
    statute_codes = set()
    for case in db_cases.find():
        statute_code = case["Statute Code"]
        split = case["Statute Code"].split("-")
        # statute_codes.add(statute_code[:5])
        statute_codes.add(split[0]) 

    print(statute_codes)
    print(len(statute_codes))
    return statute_codes

"""
TODOs:
0. Convert mongo db_cases into a pandas dataframe - done [Kevin]
1. Get the right features (month, hour, weekday vs weekend/weekdays, statute code, reporting officer, counts)
    a. convert categories into columns
        i. date -> day of the week -> [0, ..., 1]/[0, 1] (if weekend/weekday)
        ii. statute code -> first 5 chars -> [0, ..., 1]
        iii. reporting officer -> [0, ..., 1]
2. for case in cases:
        # somehow convert case into a row with these features
        # when considering arrests, experiment with how number of arrests 
        #  per case affects our input data
    output: data = num_cases x (num_features + 1) matrix 

    - Jeff 
    
3. LogisticRegression.train(data)
    a. class_weight = balanced
4. Get new data, put in db as db_test_cases, db_test_arrests
    - Kevin
        - instead, use train_test_split method from sklearn
5. LogisticRegression.train(test_data)
6. Analyze results:
    a. Make precision-recall graph

"""


def train_and_test(X, y):
    """
    Method to train and test on our data!
    :param X: pandas df of the features
    :param y: some form of a list/nparray/df of the labels, ordered in the same way as X
    :return: the accuracy/score of the model!
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        train_size=0.9,
        random_state=0,
        shuffle=True
    )

    model = LogisticRegression().fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Score:", score)
    return score

def main():
    df = mongo_to_df()

if __name__ == "__main__":
    main()
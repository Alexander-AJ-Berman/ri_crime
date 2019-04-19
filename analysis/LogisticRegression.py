import csv
import json
import pandas as pd 
import sys, getopt, pprint
import pymongo
from pymongo import MongoClient
import datetime
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

def convert_data(df):
    data = []
    columns = df.columns;
    for index, row in df.iterrows():
        tmp = {}
        for col in columns:
            if col != 'lat' and col != 'lon':
                tmp[col] = row[col]
        data.append(tmp)
    return data 

def clean_data(data):
    ml_data = []
    labels = []
    officers = get_officers(data)
    statute_codes = get_statute_codes()
    for row in data:
        tmp = []
        tmp.append(row['Month'])
        tmp.append(convert_to_hour(row['Reported Date']))
        tmp.append(convert_to_day_of_the_week(row['Reported Date']))
        tmp.append(convert_statute_code(row['Statute Code'],statute_codes))
        tmp.append(convert_reporting_officer(row['Reporting Officer'],officers))
        tmp.append(row['Counts'])
        tmp.append(row['latitude'])
        tmp.append(row['longitude'])
        ml_data.append(tmp)
        if row['Arrests'] != []:
            labels.append(1)
            print(row['Arrests'])
        else:
            labels.append(0)

    return ml_data, labels

def get_officers(data):
    officers = set()
    for row in data:
        officers.add(row['Reporting Officer'])
    officer_dict = {}
    num = 0
    for officer in list(officers):
        officer_dict[officer] = num
        num += 1
    return officer_dict

def convert_to_hour(date):
    split_date = date.split()
    hour = int(split_date[1][:2])
    if split_date[2] == 'PM':
        hour += 12
    return hour

def convert_to_day_of_the_week(date):
    one_hot_vector = [0] * 7
    split_date = date.split()[0]
    split_date = split_date.split('/')
    day_of_the_week = datetime.datetime(int(split_date[2]),int(split_date[0]),int(split_date[1])).weekday()
    one_hot_vector[day_of_the_week] = 1
    return one_hot_vector

def convert_reporting_officer(officer,officers):
    one_hot_vector = [0] * len(officers.keys())
    one_hot_vector[officers[officer]] = 1
    return one_hot_vector

def convert_statute_code(statute,statutes):
    one_hot_vector = [0] * len(statutes.keys())
    one_hot_vector[statutes[statute[:4]]] = 1
    return one_hot_vector

def get_statute_codes():
    statute_codes = set()
    for case in db_cases.find():
        statute_code = case["Statute Code"]
        # split = case["Statute Code"].split("-")
        statute_codes.add(statute_code[:4])
        # statute_codes.add((split[0].strip(),split[1].strip() if len(split) > 1 else '')) 
    statute_dict = {}
    num = 0
    for code in statute_codes:
        statute_dict[code] = num
        num += 1
    return statute_dict

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
    data = convert_data(df)
    ml_data, labels = clean_data(data)

if __name__ == "__main__":
    main()
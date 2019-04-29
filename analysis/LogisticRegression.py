import csv
import json
import pandas as pd 
import sys, getopt, pprint
import pymongo
from pymongo import MongoClient
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import confusion_matrix
import math

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
    print(columns)
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
    types = get_types()
    districts = get_districts()
    numArrests = 0
    numCases = 0
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
        tmp.append(convert_type(row['type'],types))
        tmp.append(convert_district(row['zillow_district_id'],districts))
        ml_data.append(tmp)
        if row['Arrests'] != []:
            labels.append(1)
            numArrests += 1
        else:
            labels.append(0)
            numCases += 1
    print(numArrests)
    print(numCases)

    return np.array(ml_data), np.array(labels)

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
    # one_hot_vector[day_of_the_week] = 1
    # return np.array(one_hot_vector)
    return day_of_the_week

def convert_reporting_officer(officer,officers):
    one_hot_vector = [0] * len(officers.keys())
    one_hot_vector[officers[officer]] = 1
    # return np.array(one_hot_vector)
    return officers[officer]

def convert_statute_code(statute,statutes):
    one_hot_vector = [0] * len(statutes.keys())
    one_hot_vector[statutes[statute[:4]]] = 1
    # return np.array(one_hot_vector)
    return statutes[statute[:4]]

def convert_type(property_type,types):
    return types[property_type]

def convert_district(district,districts):
    if math.isnan(float(district)):
        return -1
    else: 
        return districts[district]

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

def get_types():
    types = set()
    for case in db_cases.find():
        types.add(case['type'])
    types_dict = {}
    num = 0
    for t in types:
        types_dict[t] = num
        num += 1
    return types_dict

def get_districts():
    districts = set()
    for case in db_cases.find():
        if 'zillow_district_id' in case:
            districts.add(case['zillow_district_id'])
    districts_dict = {}
    num = 0
    for district in districts:
        districts_dict[district] = num
        num += 1
    return districts_dict

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
    print(type_1_2_errors(model, X_test, y_test))
    precision_recall(model, X_test, y_test)

    return score

def precision_recall(model, X_test, y_test):
    y_score = model.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))

def type_1_2_errors(model, X_test, y_test):
    preds = model.predict(X_test)
    FP = confusion_matrix(y_test, preds)[0][1]
    FN = confusion_matrix(y_test, preds)[1][0]

    return {"False positive": FP, "False negative": FN}

def main():
 #   for i in range(10):
    df = mongo_to_df()
    data = convert_data(df)
    ml_data, labels = clean_data(data)
    train_and_test(ml_data, labels)

if __name__ == "__main__":
    main()
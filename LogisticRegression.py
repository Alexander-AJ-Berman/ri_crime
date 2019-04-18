import csv
import json
import pandas as pd 
import sys, getopt, pprint
from pymongo import MongoClient

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

client = MongoClient(uri)

db = client.get_database()
db_cases = db['cases']
db_arrests = db['arrests']

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
3. LogisticRegression.train(data)
    a. class_weight = balanced
4. Get new data, put in db as db_test_cases, db_test_arrests
5. LogisticRegression.train(test_data)
6. Analyze results:
    a. Make precision-recall graph

"""
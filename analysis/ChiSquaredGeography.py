import csv
import json
import pandas as pd
import sys, getopt, pprint
import pymongo
from pymongo import MongoClient
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from scipy.stats import chisquare

import os
uri = os.environ['DB']

client = MongoClient(uri)
db = client.get_database()
db_cases = db['cases']
db_arrests = db['arrests']
def num_zip_codes(db_cases):
    zc_set = set()
    for case in db_cases.find():
        zc_set.add(case["postcode"])
    return len(zc_set)

def percent_data_thrown_out(db_cases):
    zipcode_to_freqs = dict()
    CASES_INDEX = 0
    ARRESTS_INDEX = 1
    total = 0
    count = 0
    for case in db_cases.find():
        total += 1
        zipcode = case["postcode"]
        is_arrest = (case["Arrests"] != [])
        count += is_arrest
        if zipcode in zipcode_to_freqs:
            zipcode_to_freqs[zipcode][0] += 1
        else:
            zipcode_to_freqs[zipcode] = [1, 0]

        if is_arrest:
            zipcode_to_freqs[zipcode][1] += 1
    print(zipcode_to_freqs)
    kept_cases = 0
    kept_arrests = 0
    for k in zipcode_to_freqs:
        if zipcode_to_freqs[k][0] / total * count > 5:
            kept_cases += zipcode_to_freqs[k][0]
            kept_arrests += zipcode_to_freqs[k][1]
    return kept_cases/total, kept_arrests/count


def get_cases_arrests_frequencies(db_cases):
    """
    Given a db of cases, return a dict of {zipcode: frequency}
    :param db_cases: cases db
    :return: dict of ({zipcode: [case_frequency, arrest_frequency]}
    """
    zipcode_to_freqs = dict()
    CASES_INDEX = 0
    ARRESTS_INDEX = 1
    count = 0
    total = 0
    for case in db_cases.find():
        zipcode = case["postcode"]
        if True or zipcode[:3] in ["029", "028"]:
            total += 1
            is_arrest = (case["Arrests"] != [])
            count += is_arrest
            if zipcode in zipcode_to_freqs:
                zipcode_to_freqs[zipcode][CASES_INDEX] += 1
            else:
                zipcode_to_freqs[zipcode] = [1, 0]

            if is_arrest:
                zipcode_to_freqs[zipcode][ARRESTS_INDEX] += 1
    print(count, total)
    return zipcode_to_freqs, \
           [zipcode_to_freqs[k][CASES_INDEX]/total * count for k in zipcode_to_freqs if zipcode_to_freqs[k][CASES_INDEX]/total * count > 5],\
           [zipcode_to_freqs[k][ARRESTS_INDEX] for k in zipcode_to_freqs if zipcode_to_freqs[k][CASES_INDEX]/total * count > 5],\
           [k for k in zipcode_to_freqs if zipcode_to_freqs[k][CASES_INDEX]/total * count > 5]

# print(db_cases.find_one({"CaseNumber": "2018-00124416"} ))
print(num_zip_codes(db_cases))
print(percent_data_thrown_out(db_cases))

zipcode_to_freqs, cases_dist, arrests_dist, zipcodes_used = get_cases_arrests_frequencies(db_cases)
# ax1 = plt.subplot(1, 2, 1)
# ax2 = plt.subplot(1, 2, 2)
# # ax1.set_yscale('log')
# # ax2.set_yscale('log')
#
# ax1.tick_params(labelrotation=90)
#
# ax1.bar(zipcode_to_freqs.keys(), cases_dist)
# print(zipcode_to_freqs)
# ax2.bar(zipcode_to_freqs.keys(), arrests_dist)
# ax2.tick_params(labelrotation=90)
print(chisquare(cases_dist, arrests_dist))
print(len(cases_dist))
print(len(arrests_dist))
print(zipcode_to_freqs.keys())
print(cases_dist)
print(arrests_dist)
print(zipcodes_used)

plt.show()
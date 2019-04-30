import csv
import json
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient
import os
uri = os.environ['DB']

def get_all_case_nums(db_cases):
    cases = db_cases.find()
    case_set = set()
    j = 0
    for j, case in enumerate(cases):
        case_set.add(case["CaseNumber"])
    return case_set

def main():
    client = MongoClient(uri)
    db = client.get_database()
    db_cases = db['cases']
    cases = db_cases.find()
    arrests = db_arrests.find()

    all_case_nums = get_all_case_nums(db_cases)
    print(all_case_nums)

if __name__ == "__main__":
    main()

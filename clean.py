import csv
import json
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

def get_all_case_nums(db_cases):
    cases = db_cases.find()
    case_set = set()
    j = 0
    for j, case in enumerate(cases):
        case_set.add(case["CaseNumber"])
    print(j)
    return case_set

def main():
    client = MongoClient(uri)
    db = client.get_database()
    # print(db)
    db_cases = db['cases']
    db_arrests = db['arrests']
    cases = db_cases.find()
    arrests = db_arrests.find()

    all_case_nums = get_all_case_nums(db_cases)
    # print("num cases:", len(cases))
    # print("num arrests:", len(arrests))

    i = 0
    for i, arrest in enumerate(arrests):
        if (arrest["Case Number"] not in all_case_nums):
            print(arrest["Case Number"])
    print(i)
    print(len(all_case_nums))

    # j = 0
    # for j, arrest in enumerate(cases):
    #     pass
    # print(j)
    # print(db_cm.find())
    # print(get_all_case_nums(db_cm))


    # header = [
    #     "CaseNumber",
    #     "Location",
    #     "Reported Date",
    #     "Month",
    #     "Year",
    #     "Offense Desc", s
    #     "Statute Code",
    #     "Statute Desc",
    #     "Counts",
    #     "Reporting Officer"
    # ]





if __name__ == "__main__":
    main()

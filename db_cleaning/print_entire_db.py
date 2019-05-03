import csv
import json
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient
import os
uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

def main():
    client = MongoClient(uri)
    db = client.get_database()
    db_cases = db['cases']
    cases = db_cases.find()

    with open("../data/allCaseData.csv", 'w') as f:
        count = 0
        header = cases[0].keys()
        f.write(",".join(header) + "\n")
        count += 1
        for case in cases:
            s = ""
            for item in list(case.values()):
                s = s + "," + str(item)
            print(s)
            f.write(s[1:] + "\n")


if __name__ == "__main__":
    main()

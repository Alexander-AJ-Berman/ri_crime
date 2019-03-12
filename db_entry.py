import csv
import json
import pandas as pd 
import sys, getopt, pprint
from pymongo import MongoClient

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

def main():
    client = MongoClient(uri)

    db = client.get_database()
    db_cm = db['cases']

    data = pd.read_csv("./PPCL_180days.csv")
    data_json = json.loads(data.to_json(orient='records'))
    db_cm.insert_many(data_json)

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
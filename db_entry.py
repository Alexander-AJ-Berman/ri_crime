import csv
import json
import pandas as pd 
import sys, getopt, pprint
from pymongo import MongoClient

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'


def add_arrest_to_case(case_num: str, statute_code: str, arrest, db_cases):
    """
    Method to combine an arrest into a case.
    Updates the case doc in db_cases with this case_num and statute_code.
    :param case_num: the case number
    :param statute_code: the statute code
    :param arrest: the dict representing an arrest made with this case_num and statute_code
    :param db_cases: the table of cases

    Note: this method assumes every case in db_cases is uniquely identified by (case_num, statute_code)
    """
    query = {"CaseNumber": case_num,
             "Statute Code": statute_code}
    db_cases.update_one(query, {'$addToSet': {"Arrests": arrest}})


def print_case(case_num: str, statute_code: str, db_cases):
    query = {"CaseNumber": case_num,
             "Statute Code": statute_code}
    print(db_cases.find_one(query))


def combine_arrests_cases_db(db_arrests, db_cases):
    """
    Given db_arrests in which each each row contains a "Case Number" and "Statute Code",
    add the row to the "Arrests" list of the corresponding case in db_cases.
    :param db_arrests: db for arrests
    :param db_cases: db for cases
    """
    for arrest in db_arrests.find():
        case_num = arrest["Case Number"]
        statute_code = arrest["Statute Code"]
        add_arrest_to_case(case_num, statute_code, arrest, db_cases)


def main():
    client = MongoClient(uri)

    db = client.get_database()
    db_cases = db['cases']
    db_arrests = db['arrests']

    case_data = pd.read_csv("./PPCL_180days.csv")
    arrest_data = pd.read_csv("./PPAL_60days.csv")

    case_json = json.loads(case_data.to_json(orient='records'))
    arrest_json = json.loads(arrest_data.to_json(orient='records'))

    # CREATE ARRESTS TABLE
    # db_arrests.insert_many(arrest_json)

    # Appending a new attribute to case_json
    # for case in case_json:
    #     case['Arrests'] = []

    # CREATE CASES TABLE
    # db_cases.insert_many(case_json)

    # COMBINE ARRESTS AND CASES
    # combine_arrests_cases_db(db_arrests, db_cases)


    # TODO: write an assert to make sure cases are actually UID'd by case number and statute code


if __name__ == "__main__":
    main()
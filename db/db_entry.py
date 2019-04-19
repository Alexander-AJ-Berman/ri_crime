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


def del_all_arrests_from_cases(db_cases):
    for case in db_cases.find({"Arrests": {"$exists": False}}):
        db_cases.update_one(case, {'$set': {"Arrests": []}})
    # print(db_cases.find_one({"CaseNumber": case_num, "Statute Code": sc}))

    print(get_num_arrests_in_cases(db_cases))


def print_case(case_num: str, statute_code: str, db_cases):
    """

    :param case_num:
    :param statute_code:
    :param db_cases:
    :return:
    """
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


def get_num_arrests_in_cases(db_cases):
    num_total_cases = db_cases.find().count()
    num_not_arrests = db_cases.find({"Arrests": []}).count()
    num_arrests = num_total_cases - num_not_arrests
    return {"num_total_cases": num_total_cases,
            "num_not_arrests": num_not_arrests,
            "num_arrests": num_arrests}


def assert_case_num_statute_code_UID(db):
    """
    Assertion to make sure cases are actually UID'd by case number and statute code
    :param db: probably want this to be db_cases
    :return: Number of cases
    """
    case_num_statute_code_set = set()
    for entry in db.find():
        id = (entry["CaseNumber"], entry["Statute Code"])
        if id in case_num_statute_code_set:
            # print(entry)
            # DELETE ENTRY
            db.delete_one(entry)
        case_num_statute_code_set.add(id)

    assert len(case_num_statute_code_set) == db.find().count()
    return len(case_num_statute_code_set)


def add_case_data_to_db(db_cases, csv_path):
    cases_data = pd.read_csv(csv_path)
    json_data = json.loads(cases_data.to_json(orient='records'))
    for case in json_data:
        if db_cases.find_one({"CaseNumber": case["CaseNumber"],
                          "Statute Code": case["Statute Code"]}) is None:
            db_cases.insert_one(case)


def add_arrest_data_to_db(db_arrests, csv_path):
    arrests_data = pd.read_csv(csv_path)
    json_data = json.loads(arrests_data.to_json(orient='records'))
    for arrest in json_data:
        if db_arrests.find_one({"Case Number": arrest["Case Number"],
                          "Statute Code": arrest["Statute Code"]}) is None:
            db_arrests.insert_one(arrest)

def main():
    client = MongoClient(uri)

    db = client.get_database()
    db_cases = db['cases']
    db_arrests = db['arrests']


    # add new case, arrest data
    # print("Adding cases")
    # add_case_data_to_db(db_cases, "../data/cases_new.csv")
    #
    # print("Adding arrests")
    # add_arrest_data_to_db(db_arrests, "../data/arrests_new.csv")

    # print("combining arrests, cases")
    # combine arrest, case data

    # combine_arrests_cases_db(db_arrests, db_cases)

    # reset arrests for each case
    # del_all_arrests_from_cases(db_cases)
    print(get_num_arrests_in_cases(db_cases))
    # print(db_cases.find_one({"Arrests": {"$exists": False}}))

    # CREATE ARRESTS TABLE
    # db_arrests.insert_many(arrest_json)

    # Appending a new attribute to case_json
    # for case in case_json:
    #     case['Arrests'] = []

    # CREATE CASES TABLE
    # db_cases.insert_many(case_json)

    # COMBINE ARRESTS AND CASES
    # combine_arrests_cases_db(db_arrests, db_cases)

    # print cases:
    # a = db_cases.find({"CaseNumber": "2018-00133531"})
    # for i in a:
    #     print(i)

    # print(get_num_arrests_in_cases(db_cases))
    # print(assert_case_num_statute_code_UID(db_cases))
    # for i in db_cases.find():
    #     print(i)

if __name__ == "__main__":
    main()
import csv
import json
import pandas as pd
import sys, getopt, pprint
from pymongo import MongoClient
import googlemaps

import os
uri = os.environ['DB']

def get_case_addresses(db_cases):
    cases = db_cases.find()
    case_set = set()
    for case in cases:
        if case["Location"]:
            str = case["Location"] + ", Providence, RI"
            case_set.add(str)
    return case_set

def main():
    client = MongoClient(uri)
    db = client.get_database()
    # print(db)
    db_cases = db['cases']
    db_arrests = db['arrests']
    cases = db_cases.find()
    arrests = db_arrests.find()

    all_case_addresses = get_case_addresses(db_cases)

    gmaps = googlemaps.Client(key='AIzaSyAvj4vtNocwxBpiB-UCo2TIL_yOgCvCr6E')

    # Geocoding an address
    geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')
    print(geocode_result)

    # Look up an address with reverse geocoding
    # reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))






if __name__ == "__main__":
    main()

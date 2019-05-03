import csv
import json
import pandas as pd
import sys
import getopt
import pprint
from pymongo import MongoClient
import requests
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
import urllib.parse
from scipy import spatial
import numpy as np

import os
uri = os.environ['DB']

def main():
    client = MongoClient(uri)
    db = client.get_database()
    db_cases = db['cases']
    db_postdata = db['postcode_data']

    with open("../data/postcode_data.txt") as f:
        s = f.readlines()
        megaS = ""
        for item in s:
            megaS = megaS + item
        arr = megaS.split("STARTING A NEW ZIP CODE\n")[1:]

        for zip_data in arr:
            internal_arr = zip_data.split("\n")
            zipcode = internal_arr[0]
            for i in db_postdata.find():
                print(i["Zillow Home Value Index"])
                if i['Postcode'] == zipcode:
                    id = i["_id"]
                    internal_data = {}
                    for index, datapoint in enumerate(internal_arr):
                        if datapoint == "Zillow Home Value Index":
                            if internal_arr[index+1].replace('.','',1).isdigit() and internal_arr[index+2].replace('.','',1).isdigit():
                                internal_data["Average home value"] = (float(internal_arr[index+1]) + float(internal_arr[index+2])) / 2
                                homevalue = (float(internal_arr[index+1]) + float(internal_arr[index+2])) / 2
                                db_postdata.update({"_id" :id },{"$set" : {"Zillow Home Value Index": homevalue}})

                    d[zipcode] = internal_data
                    if not internal_data:
                        db_postdata.update({"_id" :id },{"$set" : {"Zillow Home Value Index": None}})

if __name__ == "__main__":
    main()

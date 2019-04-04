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

case_data = pd.read_csv("./PPCL_180days.csv")
arrest_data = pd.read_csv("./PPAL_60days.csv")

case_json = json.loads(case_data.to_json(orient='records'))
arrest_json = json.loads(arrest_data.to_json(orient='records'))

all = db_cases.find({})

pairs = []
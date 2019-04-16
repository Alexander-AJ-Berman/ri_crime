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

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

lst = []

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        region_id = ""
        region_name = ""
        if tag == 'region':
            for attr in attrs:
                if attr[0] == 'id':
                    region_id = attr[1]
                if attr[0] == 'name':
                    region_name = attr[1]
        # if region_id:
            lst.append(region_id)
            lst.append(region_name)


parser = MyHTMLParser()


def main():
    client = MongoClient(uri)

    db = client.get_database()
    db_cases = db['cases']
    db_arrests = db['arrests']

    case_data = pd.read_csv("./PPCL_180days.csv")
    arrest_data = pd.read_csv("./PPAL_60days.csv")

    case_json = json.loads(case_data.to_json(orient='records'))
    arrest_json = json.loads(arrest_data.to_json(orient='records'))

    key = "X1-ZWz1h0tux20p3f_29gz1"

    encoded_citystate = urllib.parse.quote("Providence, RI".encode('utf-8'))

    count = 0
    for i in db_cases.find():
        print("case number", count)
        address = (i['Location'])
        encoded_address = urllib.parse.quote(address.encode('utf-8'))
        URL = "http://www.zillow.com/webservice/GetSearchResults.htm?zws-id=" + \
            key + "&address=" + encoded_address + "&citystatezip=" + encoded_citystate
        r = requests.get(url=URL)

        lst.append(i['_id'])
        parser.feed(r.text)
        count = count + 1
        # if count > 100:
        #     break

    final_list = []
    for index, item in enumerate(lst):
        if not isinstance(item, str):
            if index+1 < len(lst) and isinstance(lst[index+1], str):
                new_thing = [item, lst[index+1], lst[index+2]]
                final_list.append(new_thing)
    # print(final_list)


if __name__ == "__main__":
    main()

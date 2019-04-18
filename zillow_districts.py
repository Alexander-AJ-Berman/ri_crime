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

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'


def main():
    client = MongoClient(uri)

    db = client.get_database()
    db_cases = db['cases']
    db_arrests = db['arrests']
    db_postdata = db['postcode_data']

    case_data = pd.read_csv("./PPCL_180days.csv")

    case_json = json.loads(case_data.to_json(orient='records'))

    key = "X1-ZWz1h0tux20p3f_29gz1"

    encoded_citystate = urllib.parse.quote("Providence, RI".encode('utf-8'))

    count = 0
    num_searching = db_cases.count({"Arrests": {"$ne": []}})
    zips = []
    print(client.list_database_names())


    # db.collection.insert("postcode_data", {"Avg_commute_time": "<float>", "Avg_household_size": "<float>", "Median_age": "<float>", "Median_household_income": "<float>", "Percent_renters": "<float>", "Percent_owners": "<float>"})

    # for i in db_cases.find():
    #     if "postcode" in i.keys():
    #         zips.append(i["postcode"])
    #
    # new_zips = list(set(zips))
    # with open('data/postcode_data.txt', 'w') as f:
    #     class MyHTMLParser(HTMLParser):
    #         def handle_data(self, data):
    #             # if flag:
    #             #     f.write("%s\n" % str(data))
    #             #     flag = 0
    #             # if data == "Average Commute Time (Minutes)" or data == "Average Household Size" or data == "Median Age" or data == "Median Household Income" or data == "Renters":
    #             #     flag = 1
    #             f.write("%s\n" % str(data))
    #
    #     parser = MyHTMLParser()
    #     for zip in new_zips:
    #         if str(zip)[:2] == "02":
    #             print(zip)
    #             f.write("%s\n" % "STARTING A NEW ZIP CODE")
    #             URL = "http://www.zillow.com/webservice/GetDemographics.htm?zws-id=" + \
    #                 key + "&zip=" + str(zip)
    #             r = requests.get(url=URL)
    #             parser.feed(r.text)

    # data = pd.read_csv("data/case_address_lat_lng.csv", encoding='utf8')
    # accuracy = data["accuracy"].tolist()
    # database_address = data["database_address"].tolist()
    # new_database_address = [str(address).strip().upper() for address in database_address]
    #
    # formatted_address = data["formatted_address"].tolist()
    # google_place_id = data["google_place_id"].tolist()
    # latitude = data["latitude"].tolist()
    # longitude = data["longitude"].tolist()
    # postcode = data["postcode"].tolist()
    # type = data["type"].tolist()

    # location_vectors = []
    # zillow_neighborhood = []
    # zillow_district = []
    #
    # zips = []
    # count = 0
    # print(new_database_address)
    # for i in db_cases.find():
    #     print(i)
    #     if "latitude" in i.keys() and "zillow_neighborhood_name" in i.keys():
    #         if not np.isnan(float(i["latitude"])) and not np.isnan(float(i["longitude"])):
    #             vec = [float(i["latitude"]), float(i["longitude"])]
    #             location_vectors.append(vec)
    #             zillow_neighborhood.append(i["zillow_neighborhood_name"])
    #             zillow_district.append(i["zillow_district_id"])
    #
    # tree = spatial.KDTree(location_vectors)

    # for i in db_cases.find():
    #     if "latitude" in i.keys() and "zillow_neighborhood_name" in i.keys():
    #         print(i["zillow_district_id"])
    #         break
    #         count = count+ 1
    #         print(count)
    # vec = np.asarray([float(i["latitude"]), float(i["longitude"])])
    # idx = tree.query([vec])[1][0]
    # if idx < len(zillow_district):
    #     db_cases.update({"_id" :id },{"$set" : {"zillow_neighborhood_name": zillow_neighborhood[idx], "zillow_district_id": zillow_district[idx]}})

    #         zips.append(i["postcode"])
    # print(set(zips))
    # id = i["_id"]
    # if i['Location'] and 'longitude' not in i.keys():
    #     print(count)
    #     count += 1
    #     addy = i['Location'].strip().upper()
    #     list_index = new_database_address.index(addy)
    #     db_cases.update({"_id" :id },{"$set" : {"formatted_address": formatted_address[list_index],"google_place_id":google_place_id[list_index],
    #     "latitude": latitude[list_index], "longitude": longitude[list_index], "postcode": postcode[list_index], "type": type[list_index]} })

    # if arrests['From Address']:
    #     addy = arrests['From Address'].strip().upper()
    #     if addy in new_database_address:
    #         list_index = new_database_address.index(addy)
    #
    #         arrests["formatted_address"] = formatted_address[list_index]
    #         arrests["google_place_id"] = google_place_id[list_index]
    #         arrests["latitude"] = latitude[list_index]
    #         arrests["longitude"] = longitude[list_index]
    #         arrests["postcode"] = postcode[list_index]
    #         arrests["type"] = type[list_index]
    #     else:
    #         if addy in new_database_address1:
    #             list_index = new_database_address1.index(addy)
    #
    #             arrests["formatted_address"] = formatted_address1[list_index]
    #             arrests["google_place_id"] = google_place_id1[list_index]
    #             arrests["latitude"] = latitude1[list_index]
    #             arrests["longitude"] = longitude1[list_index]
    #             arrests["postcode"] = postcode1[list_index]
    #             arrests["type"] = type1[list_index]
    #
    #         else:
    #             print("OH NO IT BROKE")
    #         # results.append(get_google_results(addy, 'AIzaSyAvj4vtNocwxBpiB-UCo2TIL_yOgCvCr6E', False))
    #     # arrests["new_field"] = "test"
    #
    #
    #     print(arrests)
    #     db_cases.update({"_id" :id },{"$set" : {"Arrests":[arrests]}})

    # for i in db_cases.find():
    #     print("case " + str(count) +" out of " + str(num_searching))
    #     count = count + 1
    #
    #     if not i['Arrests']:
    #         continue
    #
    #     arrest = i['Arrests'][0]
    #     address = arrest['From Address']
    #     old_address = address
    #
    #     if old_address in lst or not old_address:
    #         continue
    #     if "&" in address:
    #         add_to_use = address[:address.find("&")]
    #         if len(address[address.find("&"):]) > len(address[:address.find("&")]):
    #             add_to_use = address[address.find("&") + 1:]
    #         address = "1 " + add_to_use
    #     # print("address used ", address)
    #
    # encoded_address = urllib.parse.quote(address.encode('utf-8'))
    # URL = "http://www.zillow.com/webservice/GetSearchResults.htm?zws-id=" + \
    #     key + "&zip=" + "02906"
    # r = requests.get(url=URL)
    # print(r.text)
    #
    #     lst.append("&&&")
    #     lst.append(old_address)
    #     parser.feed(r.text)
    #     # if count > 100:
    #     #     break
    #
    # final_count = 0
    # with open('test.txt', 'w') as f:
    #     for index, item in enumerate(lst):
    #         if item == "&&&":
    #             if index+1 < len(lst) and lst[index+2] != "&&&":
    #                 new_thing = lst[index+1] +";" + lst[index+2] +";" + lst[index+3]
    #                 f.write("%s\n" % str(new_thing))


if __name__ == "__main__":
    main()

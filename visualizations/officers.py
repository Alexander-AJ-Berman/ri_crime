"""
Multiple pie charts where
slice of X = (# of arrests of ethnicity X)/(total # of arrests),
where each officer has their own chart
"""

from pymongo import MongoClient
from collections import Counter
import itertools
import matplotlib.pyplot as plt

uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'

client = MongoClient(uri)
db = client.get_database()
db_arrests = db['arrests']
arrests = db_arrests.find()

officer_lst = []
race_lst = []
ethnicity_lst = []

for arrest in arrests:
        officers = arrest["Arresting Officers"]
        race = arrest["Race"]
        ethnicity = arrest["Ethnicity"]
        
        if officers != None:
                # Remove leading and trailing whitespace
                officers = officers.split(',')
                
                for officer in officers:
                        if officer != None and race != None and ethnicity != None:
                                officer_lst.append(officer.strip())
                                race_lst.append(race.strip())
                                ethnicity_lst.append(ethnicity.strip())


labels = ['White', 'Black', 'Unknown', 'Asian/Pacific Islander', 'American Indian/Alaskan Native', 'ZHispanic (FD only)']
sizes = [Counter(race_lst)[el] for el in Counter(race_lst)]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels)
ax1.axis('equal')
plt.legend(loc="upper left")


plt.show()

# Racial arrests by total, normalize pie chart
        
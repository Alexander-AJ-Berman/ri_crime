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



#c = Counter([x.encode('ascii') for x in ethnicity_lst])

# print(len(officer_lst))
# print(len(race_lst))





# labels = ['Hispanic', 'NonHispanic', 'Unknown']
# sizes = [c[label] for label in labels]
# colors = ['#ff9999','#66b3ff','#99ff99']
# explode = [.02, .02, .01]

# fig1, ax1 = plt.subplots()
# patches, texts, autotexts = ax1.pie(sizes,colors=colors, labels=labels, autopct='%1.1f%%',explode=explode, pctdistance=.85, startangle=-45)
# center_circle = plt.Circle((0,0), 0.70, fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(center_circle)

# for text in texts:
#         text.set_weight('bold')
# for autotext in autotexts:
#         autotext.set_weight('bold')
# ax1.axis('equal')
# plt.tight_layout()
# plt.title('Ethnic Breakdown of Arrests as Reported by PPD')
# plt.show()
"""
Multiple pie charts where
slice of X = (# of arrests of ethnicity X)/(total # of arrests),
where each officer has their own chart
"""

from pymongo import MongoClient
from collections import Counter
import itertools
import matplotlib.pyplot as plt
import operator
import numpy as np

import os
uri = os.environ['DB']

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

# Maps officers to a list counting the race breakdown
# 0Black, 1White, 2Unknown, 3Asian/Pacific Islander, 4American Indian/Alaskan Native
lst = zip([x.encode('ascii') for x in officer_lst], [x.encode('ascii') for x in race_lst])

officers = {}

for tuple in lst:
        if tuple[0] not in officers:
                officers[tuple[0]] = [tuple[1]]
        else:
                officers[tuple[0]].append(tuple[1])

sorted_officers = sorted(officers.items(), key=lambda x: len(x[1]), reverse=True)[1:21]

officer_counts = []

for el in sorted_officers:
        c = Counter(el[1])
        officer_counts.append((el[0], c))




races = ['Black', 'White', 'Asian/Pacific Islander', 'American Indian/Alaskan Native', 'Unknown']
labels = [str(n) for n in range(1, 21)]
black_size = np.array([el[1]['Black'] for el in officer_counts])
white_size = np.array([el[1]['White'] for el in officer_counts])
unknown_size = np.array([el[1]['Unknown'] for el in officer_counts])
api_size = np.array([el[1]['Asian/Pacific Islander'] for el in officer_counts])
aian_size = np.array([el[1]['American Indian/Alaskan Native'] for el in officer_counts])
colors = ['#596f91', '#917d59', '#916259', '#b3a3d1', '#a3d1c3']

x_pos = [i for i, _ in enumerate(labels)]
# '#ff9999','#66b3ff','#99ff99', '#ffcc99','#9400D3'
plt.bar(x_pos, black_size, width = 0.8, label='Black', color=colors[0])
plt.bar(x_pos, white_size, width = 0.8, label='White', color=colors[1], bottom=black_size)
plt.bar(x_pos, api_size, width = 0.8, label='Asian/Pacific Islander', color=colors[2], bottom=black_size+white_size)
plt.bar(x_pos, aian_size, width = 0.8, label='American Indian/Alaskan Native', color=colors[3], bottom=black_size+white_size+api_size)
plt.bar(x_pos, unknown_size, width = 0.8, label='Unknown', color=colors[4], bottom=black_size+white_size+api_size+aian_size)
plt.xticks(x_pos, labels)
plt.xlabel("Officer # (Names Redacted)", fontdict = {'fontsize' : 12})
plt.ylabel("Total Arrests", fontdict = {'fontsize' : 12})
plt.title("Racial Breakdown of Individual Arrests by Officer (Jan 2019 - April 2019)", fontdict = {'fontsize' : 24})
plt.legend(races, loc='best')
plt.show()

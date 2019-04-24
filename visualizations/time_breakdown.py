import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient

def plot_month(db_cases):
    """
    x-axis: each month
    y-axis: num_cases, num_arrests
    :param db_cases: cases
    :return: stacked bar chart
    """
    months = {}

    for i in db_cases.find():
        month = i['Month']
        caseNumber = i['CaseNumber']

        if month in months.keys():
            d = months[month]
            newCases = d['Cases']
            newCases.append(caseNumber)
            d['Cases'] = newCases
        else:
            d = {}
            d['Cases'] = [caseNumber]
            d['Arrests'] = []
            months[month] = d

        if i['Arrests'] != []:
            arrestMonth = i['Arrests'][0]['Month']
            d = months[month]
            newArrest = d['Arrests']
            newArrest.append(caseNumber)
            d['Arrests'] = newArrest
            months[month] = d

    numCases = []
    numArrests = []
    monthNames = ['Sep, 2018', 'Oct', 'Nov', 'Dec', 'Jan, 2019', 'Feb', 'Mar', 'Apr']

    for month, monthData in months.items():
        numCases.append(len(set(monthData['Cases'])))
        numArrests.append(len(set(monthData['Arrests'])))

    numCases.reverse()
    numArrests.reverse()

    N = len(months.keys())
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, numCases, width)
    p2 = plt.bar(ind, numArrests, width,
                 bottom=numCases)

    plt.ylabel('Number of Cases')
    plt.title('Number of Cases and Arrests By Month')
    plt.xticks(ind, monthNames)
    plt.legend((p1[0], p2[0]), ('Cases', 'Arrests'))

    plt.savefig("../visualization_output/arrests_cases_by_month.png")


def plot_day_of_week(db_cases):
    """
    x-axis: each day of week
    y-axis: num_cases, num_arrests (2 separate lines, double y-axis)
    :param db_cases: cases
    :return: histogram
    """
    pass


def plot_time_of_day(db_cases):
    """
    x-axis: each hour
    y-axis: num_cases, num_arrests (2 separate lines, double y-axis)
    :param db_cases: cases
    :return: histogram
    """
    pass


def plot_every_day(db_cases):
    """
    x-axis: each day from december to april
    y-axis: num_cases, num_arrests (2 separate lines, double y-axis)
    :param db_cases: cases
    :return: line plot or histogram
    """
    pass

def main():
    uri = 'mongodb://user:password1@ds159025.mlab.com:59025/ri_crime_data'
    client = MongoClient(uri)
    db = client.get_database()
    db_cases = db['cases']
    plot_month(db_cases)

if __name__ == "__main__":
    main()

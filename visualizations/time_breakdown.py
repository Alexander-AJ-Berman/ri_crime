import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import matplotlib.patches as mpatches
import math

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
            d = months[arrestMonth]
            newArrest = d['Arrests']
            newArrest.append(caseNumber)
            d['Arrests'] = newArrest
            months[arrestMonth] = d

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

def reported_date_to_time_of_day(reportedDate):
    timeOfDay = ""
    arr = reportedDate.split(' ')
    hour = int(arr[1].split(':')[0])
    am_pm = arr[2]
    if am_pm == 'AM':
        if hour < 6:
            timeOfDay = "Night"
        elif hour < 10:
            timeOfDay = "Early Morning"
        else:
            timeOfDay = "Late Morning"
    else:
        if hour < 6:
            timeOfDay = "Afternoon"
        elif hour < 10:
            timeOfDay = "Evening"
        else:
            timeOfDay = "Night"
    return timeOfDay

def plot_time_of_day(db_cases):
    """
    x-axis: each hour
    y-axis: num_cases, num_arrests (2 separate lines, double y-axis)
    :param db_cases: cases
    :return: stacked bar chart
    """
    timesofday = {}

    for i in db_cases.find():
        timeOfDay = reported_date_to_time_of_day(i['Reported Date'])
        caseNumber = i['CaseNumber']

        if timeOfDay in timesofday.keys():
            d = timesofday[timeOfDay]
            newCases = d['Cases']
            newCases.append(caseNumber)
            d['Cases'] = newCases
        else:
            d = {}
            d['Cases'] = [caseNumber]
            d['Arrests'] = []
            timesofday[timeOfDay] = d

        if i['Arrests'] != []:
            arrestTimeOfDay = reported_date_to_time_of_day(i['Arrests'][0]['Arrest Date'])
            d = timesofday[arrestTimeOfDay]
            newArrest = d['Arrests']
            newArrest.append(caseNumber)
            d['Arrests'] = newArrest
            timesofday[arrestTimeOfDay] = d

    numCases = []
    numArrests = []
    dayTimes = []

    for dayTime, dayData in timesofday.items():
        dayTimes.append(dayTime)
        numCases.append(len(set(dayData['Cases'])))
        numArrests.append(len(set(dayData['Arrests'])))

    N = len(timesofday.keys())
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, numCases, width)
    p2 = plt.bar(ind, numArrests, width,
                 bottom=numCases)

    plt.ylabel('Number of Cases')
    plt.title('Number of Cases and Arrests By Time of Day')
    plt.xticks(ind, dayTimes)
    plt.legend((p1[0], p2[0]), ('Cases', 'Arrests'))

    plt.savefig("../visualization_output/arrests_cases_by_time_of_day.png")

def plot_median_income(db_cases, db_postcodes):
    """
    x-axis: each hour
    y-axis: num_cases, num_arrests (2 separate lines, double y-axis)
    :param db_cases: cases
    :return: stacked bar chart
    """
    home_values = {}
    zipcodes_to_incomes = {}

    for i in db_postcodes.find():
        zipcodes_to_incomes[i["Postcode"]] = i["Zillow Home Value Index"]

    for i in db_cases.find():
        if i["postcode"] in zipcodes_to_incomes.keys():
            home_value = zipcodes_to_incomes[i["postcode"]]
        if home_value is not None:
            rounded_home_value = "<$210k"
            if home_value > 250000:
                rounded_home_value = "$250k+"
            elif home_value > 225000:
                rounded_home_value = "$225k - $255k"
            elif home_value > 210000:
                rounded_home_value = "$210k - $225k"
            # rounded_home_value = math.floor(home_value / 100000.0) * 100000.0
            caseNumber = i['CaseNumber']

            if rounded_home_value in home_values.keys():
                d = home_values[rounded_home_value]
                newCases = d['Cases']
                newCases.append(caseNumber)
                d['Cases'] = newCases
                home_values[rounded_home_value] = d
            else:
                d = {}
                d['Cases'] = [caseNumber]
                d['Arrests'] = []
                home_values[rounded_home_value] = d

            if i['Arrests'] != []:
                d = home_values[rounded_home_value]
                newArrest = d['Arrests']
                newArrest.append(caseNumber)
                d['Arrests'] = newArrest
                home_values[rounded_home_value] = d

    numCases = []
    numArrests = []
    homevals = []

    for homeval, dayData in home_values.items():
        homevals.append(homeval)
        numCases.append(len(set(dayData['Cases'])))
        numArrests.append(len(set(dayData['Arrests'])))

    together = zip(homevals, numCases)
    sorted_together = sorted(together)
    homevalssorted = [x[0] for x in sorted_together]
    numCasesSorted = [x[1] for x in sorted_together]

    together1 = zip(homevals, numArrests)
    sorted_together1 = sorted(together1)
    numArrestsSorted = [x[1] for x in sorted_together1]

    homevalssorted = [homevalssorted[-1]] + homevalssorted[:-1]
    numArrestsSorted = [numArrestsSorted[-1]] + numArrestsSorted[:-1]
    numCasesSorted = [numCasesSorted[-1]] + numCasesSorted[:-1]

    N = len(home_values.keys())
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, numCasesSorted, width)
    p2 = plt.bar(ind, numArrestsSorted, width,
                 bottom=numCasesSorted)

    plt.ylabel('Number of Cases')
    plt.title('Number of Cases and Arrests By Average Area Home Value')
    plt.xticks(ind, homevalssorted)
    blue_patch = mpatches.Patch(color='blue', label='Cases')
    orange_patch = mpatches.Patch(color='orange', label='Arrests')
    plt.legend(handles=[blue_patch, orange_patch])

    plt.savefig("../visualization_output/median_home_values.png")


def plot_every_day(db_cases):
    """
    x-axis: each day
    y-axis: num_cases, num_arrests (2 separate lines, double y-axis)
    :param db_cases: cases
    :return: line plot
    """
    days = {}

    for i in db_cases.find():
        day = i['Reported Date'].split(' ')[0]
        caseNumber = i['CaseNumber']

        if day in days.keys():
            d = days[day]
            newCases = d['Cases']
            newCases.append(caseNumber)
            d['Cases'] = newCases
        else:
            d = {}
            d['Cases'] = [caseNumber]
            d['Arrests'] = []
            days[day] = d

        if i['Arrests'] != []:
            arrestDay = i['Arrests'][0]['Arrest Date'].split(' ')[0]
            if arrestDay not in days.keys():
                d = {}
                d['Cases'] = []
                d['Arrests'] = []
                days[arrestDay] = d
            d = days[arrestDay]
            newArrest = d['Arrests']
            newArrest.append(caseNumber)
            d['Arrests'] = newArrest
            days[arrestDay] = d

    numCases = []
    numArrests = []
    allDays = []
    allDaysByDate = []
    count = 0
    for day, dayData in days.items():
        allDaysByDate.append(day[6:] + "/" + day[:5])
        allDays.append(count)
        count += 1
        numCases.append(len(set(dayData['Cases'])))
        numArrests.append(len(set(dayData['Arrests'])))

    together = zip(allDaysByDate, numCases)
    sorted_together = sorted(together)
    allDaysByDateSorted = [x[0] for x in sorted_together]
    numCasesSorted = [x[1] for x in sorted_together]

    together1 = zip(allDaysByDate, numArrests)
    sorted_together1 = sorted(together1)
    numArrestsSorted = [x[1] for x in sorted_together1]

    # calculate polynomial
    z = np.polyfit(allDays, numCasesSorted, 3)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(allDays[0], allDays[-1], 50)
    y_new = f(x_new)

    plt.plot(allDaysByDateSorted,numCasesSorted,'y', x_new, y_new, 'r')
    plt.scatter(allDaysByDateSorted,numCasesSorted)
    plt.scatter(allDaysByDateSorted,numArrestsSorted)
    plt.plot(allDaysByDateSorted,numArrestsSorted,'y')

    plt.xticks(np.asarray([0, 22, 53, 85, 117, 149, 177, 208]), ('Sep, 2018', 'Oct', 'Nov', 'Dec', 'Jan, 2019', 'Feb', 'Mar', 'Apr'))
    plt.ylabel('Number of Cases')
    plt.title('Number of Cases and Arrests Each Day')
    blue_patch = mpatches.Patch(color='blue', label='Cases')
    orange_patch = mpatches.Patch(color='orange', label='Arrests')
    plt.legend(handles=[blue_patch, orange_patch])
    plt.savefig("../visualization_output/arrests_cases_each_day.png")

def main():
    import os
uri = os.environ['DB']
    client = MongoClient(uri)
    db = client.get_database()
    db_cases = db['cases']
    db_postcodes = db['postcode_data']

    # plot_month(db_cases) #done
    # plot_time_of_day(db_cases) #done
    # plot_every_day(db_cases)
    plot_median_income(db_cases, db_postcodes)

if __name__ == "__main__":
    main()

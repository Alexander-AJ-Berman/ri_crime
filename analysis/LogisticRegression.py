import pandas as pd
from pymongo import MongoClient
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import confusion_matrix
import seaborn as sns
import calendar

from sklearn.neural_network import MLPClassifier

import math

import os
uri = os.environ['DB']

client = MongoClient(uri)
db = client.get_database()
db_cases = db['cases']

def mongo_to_df():
    """
    Converts db_cases to a pandas df
    :return: pandas df
    """
    df = pd.DataFrame(list(db_cases.find()))
    # print(df.iloc[10000]) # prints the 0th row from the pandas df

    return df

    """
    # helpful other commands to look at data
    print(db_cases.find_one()) # prints one case from the mongo db
    print(df.iloc[0]) # prints the 0th row from the pandas df
    """

def convert_data(df):
    data = []
    columns = df.columns
    for index, row in df.iterrows():
        tmp = {}
        for col in columns:
            if col != 'lat' and col != 'lon':
                tmp[col] = row[col]
        data.append(tmp)
    return data 

def clean_data(data):
    ml_data = []
    labels = []
    months = get_months(data)
    officers = get_officers(data)
    statute_codes = get_statute_codes()
    zip_codes = get_zip_codes(data)
    types = get_types()
    for row in data:
        tmp = []
        print(row)
        tmp.append(convert_to_hour(row['Reported Date']))
        tmp.append(row['Counts'])
        tmp.append(row['latitude'])
        tmp.append(row['longitude'])
        tmp.append(convert_type(row['type'], types))
        tmp += (convert_to_month(row['Month'], months))
        tmp += (convert_to_day_of_the_week(row['Reported Date']))
        tmp += (convert_statute_code(row['Statute Code'], statute_codes))
        tmp += (convert_reporting_officer(row['Reporting Officer'], officers))
        tmp += (convert_zip_code(row['postcode'], zip_codes))
        ml_data.append(tmp)
        if row['Arrests'] != []:
            labels.append(1)
        else:
            labels.append(0)
    feature_list = []
    feature_dict = dict()
    feature_list.append("hour")
    feature_list.append("counts")
    feature_list.append("lat")
    feature_list.append("lon")
    feature_list.append("type")
    for month in months:
        feature_list.append(calendar.month_name[month])
    for i in range(7):
        feature_list.append(calendar.day_abbr[i])
    for sc in statute_codes:
        feature_list.append(f"sc:{sc}")
    for i in range(len(officers.keys())):
        feature_list.append(f"officer{i}")
    for zc in zip_codes:
        feature_list.append(zc)

    # (start_ind, end_ind)
    feature_dict["hour"] = (0, 1)
    feature_dict["counts"] = (1, 2)
    feature_dict["lat"] = (2, 3)
    feature_dict["lon"] = (3, 4)
    feature_dict["type"] = (4, 5)
    feature_dict["months"] = (feature_dict["type"][1], feature_dict["type"][1] + len(months.keys()))
    feature_dict["day"] = (feature_dict["months"][1], feature_dict["months"][1] + 7)
    feature_dict["sc"] = (feature_dict["day"][1], feature_dict["day"][1] + len(statute_codes.keys()))
    feature_dict["officers"] = (feature_dict["sc"][1], feature_dict["sc"][1] + len(officers.keys()))
    feature_dict["zc"] = (feature_dict["officers"][1], feature_dict["officers"][1] + len(zip_codes.keys()))

    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    reduced_data = svd.fit_transform(np.array(ml_data))
    return np.array(ml_data), np.array(labels), feature_list, feature_dict

def get_officers(data):
    officers = set()
    for row in data:
        officers.add(row['Reporting Officer'])
    officer_dict = {}
    num = 0
    for officer in list(officers):
        officer_dict[officer] = num
        num += 1
    return officer_dict

def get_zip_codes(data):
    zip_codes = set()
    for row in data:
        zip_codes.add(row['postcode'])
    zip_code_dict = {}
    num = 0
    for zipcode in list(zip_codes):
        zip_code_dict[zipcode] = num
        num += 1
    return zip_code_dict

def get_months(data):
    months = set()
    for row in data:
        months.add(row['Month'])
    month_dict = {}
    num = 0
    for month in list(months):
        month_dict[month] = num
        num += 1
    return month_dict

def convert_to_hour(date):
    split_date = date.split()
    hour = int(split_date[1][:2])
    if split_date[2] == 'PM':
        hour += 12
    return hour

def convert_to_month(month, months):
    one_hot_vector = [0] * len(months.keys())
    one_hot_vector[months[month]] = 1
    # print("rep off:", np.array(one_hot_vector).shape, one_hot_vector)
    return (one_hot_vector)

def convert_to_day_of_the_week(date):
    one_hot_vector = [0] * 7
    split_date = date.split()[0]
    split_date = split_date.split('/')
    day_of_the_week = datetime.datetime(int(split_date[2]),int(split_date[0]),int(split_date[1])).weekday()
    one_hot_vector[day_of_the_week] = 1
    return (one_hot_vector)
    # return day_of_the_week

def convert_reporting_officer(officer,officers):
    one_hot_vector = [0] * len(officers.keys())
    one_hot_vector[officers[officer]] = 1
    # print("rep off:", np.array(one_hot_vector).shape, one_hot_vector)
    return (one_hot_vector)
    # return officers[officer]

def convert_zip_code(zipcode, zipcodes):
    one_hot_vector = [0] * len(zipcodes.keys())
    one_hot_vector[zipcodes[zipcode]] = 1
    # print("rep off:", np.array(one_hot_vector).shape, one_hot_vector)
    return (one_hot_vector)

def convert_statute_code(statute,statutes):
    one_hot_vector = [0] * len(statutes.keys())
    one_hot_vector[statutes[statute[:4]]] = 1
    return (one_hot_vector)
    # return statutes[statute[:4]]

def convert_type(property_type,types):
    return types[property_type]

def convert_district(district,districts):
    if math.isnan(float(district)):
        return -1
    else: 
        return districts[district]

def get_statute_codes():
    statute_codes = set()
    for case in db_cases.find():
        statute_code = case["Statute Code"]
        # split = case["Statute Code"].split("-")
        statute_codes.add(statute_code[:4])
        # statute_codes.add((split[0].strip(),split[1].strip() if len(split) > 1 else '')) 
    statute_dict = {}
    num = 0
    for code in statute_codes:
        statute_dict[code] = num
        num += 1
    return statute_dict

def get_types():
    types = set()
    for case in db_cases.find():
        types.add(case['type'])
    types_dict = {}
    num = 0
    for t in types:
        types_dict[t] = num
        num += 1
    return types_dict

def get_districts():
    districts = set()
    for case in db_cases.find():
        if 'zillow_district_id' in case:
            districts.add(case['zillow_district_id'])
    districts_dict = {}
    num = 0
    for district in districts:
        districts_dict[district] = num
        num += 1
    return districts_dict

"""
TODOs:
0. Convert mongo db_cases into a pandas dataframe - done [Kevin]
1. Get the right features (month, hour, weekday vs weekend/weekdays, statute code, reporting officer, counts)
    a. convert categories into columns
        i. date -> day of the week -> [0, ..., 1]/[0, 1] (if weekend/weekday)
        ii. statute code -> first 5 chars -> [0, ..., 1]
        iii. reporting officer -> [0, ..., 1]
2. for case in cases:
        # somehow convert case into a row with these features
        # when considering arrests, experiment with how number of arrests 
        #  per case affects our input data
    output: data = num_cases x (num_features + 1) matrix 

    - Jeff 
    
3. LogisticRegression.train(data)
    a. class_weight = balanced
4. Get new data, put in db as db_test_cases, db_test_arrests
    - Kevin
        - instead, use train_test_split method from sklearn
5. LogisticRegression.train(test_data)
6. Analyze results:
    a. Make precision-recall graph

"""

def balance(X_train, y_train):
    """
    Attempt to throw out a ton of data so its an even split.
    Results in high recall but horrible precision (<0.3)
    :param X_train:
    :param y_train:
    :return:
    """
    num_arrests = sum(y_train)
    print(y_train.shape)
    num_non_arrests = len(y_train) - 2 * num_arrests
    print("num_arrests:", num_arrests)

    train = np.hstack((X_train, np.expand_dims(y_train, axis=1)))
    masked_idxs = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
             masked_idxs.append(i)
        if len(masked_idxs) >= num_non_arrests:
            break

    print("masked_idxs:", len(masked_idxs))
    masked_idxs = np.array(masked_idxs)

    m = np.zeros_like(train)
    m[masked_idxs, :] = 1

    masked_train = np.ma.compress_rows(np.ma.masked_array(train, m))

    new_X_train = masked_train[:, :-1]
    new_y_train = masked_train[:, -1]
    print("new x train shape:", new_X_train.shape)

    return new_X_train, new_y_train


def plot_coefs_std(std_coef, feature_list, feature_dict):
    # plot all features
    # sns.barplot(feature_list, std_coef)
    # plt.xticks(rotation=90)
    # plt.show()
    sns.barplot([feature_list[i] for i in range(0, len(feature_list)) if np.abs(std_coef[i]) > 0.1],
                [std_coef[i] for i in range(0, len(feature_list)) if np.abs(std_coef[i]) > 0.1])
    plt.xticks(rotation=90)
    plt.title("Weights across most significant features", fontsize=22)
    plt.xlabel("Feature")
    plt.ylabel("coef * std")
    plt.show()
    # plot all zipcodes
    sns.barplot(feature_list[feature_dict["zc"][0]:feature_dict["zc"][1]],
                std_coef[feature_dict["zc"][0]:feature_dict["zc"][1]])
    plt.xticks(rotation=90)
    plt.title("Weights across zipcode features (one-hot)", fontsize=22)
    plt.xlabel("Zip code")
    plt.ylabel("coef * std")
    plt.show()

    # plot all statute codes
    sns.barplot(feature_list[feature_dict["sc"][0]:feature_dict["sc"][1]],
                std_coef[feature_dict["sc"][0]:feature_dict["sc"][1]])
    plt.xticks(rotation=90)
    plt.title("Weights across statute code features (one-hot)", fontsize=22)
    plt.xlabel("Statute code group")
    plt.ylabel("coef * std")
    plt.show()

    # plot all months
    sns.barplot(feature_list[feature_dict["months"][0]:feature_dict["months"][1]],
                std_coef[feature_dict["months"][0]:feature_dict["months"][1]])
    plt.xticks(rotation=90)
    plt.title("Weights across month features (one-hot)", fontsize=22)
    plt.xlabel("month")
    plt.ylabel("coef * std")
    plt.show()

    # plot all days
    sns.barplot(feature_list[feature_dict["day"][0]:feature_dict["day"][1]],
                std_coef[feature_dict["day"][0]:feature_dict["day"][1]])
    plt.xticks(rotation=90)
    plt.title("Weights across day of the week features (one-hot)", fontsize=22)
    plt.xlabel("Day of the week")
    plt.ylabel("coef * std")
    plt.show()

    # plot all officers
    sns.barplot(feature_list[feature_dict["officers"][0]:feature_dict["officers"][1]],
                std_coef[feature_dict["officers"][0]:feature_dict["officers"][1]])
    plt.xticks(rotation=90)
    plt.title("Weights across officer features (one-hot)", fontsize=22)
    plt.xlabel("Officer")
    plt.ylabel("coef * std")
    plt.xticks([])
    plt.show()

def train_and_test(X, y, feature_list, feature_dict):
    """
    Method to train and test on our data!
    :param X: pandas df of the features
    :param y: some form of a list/nparray/df of the labels, ordered in the same way as X
    :return: the accuracy/score of the model!
    """
    print((X.shape))
    # print(X[0:10])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        train_size=0.9,
        # random_state=0,
        shuffle=True
    )

    # new_X_train, new_y_train = balance(X_train, y_train)
    # X_test, y_test = balance(X_test, y_test)

    class_weights = {0: 0.9, 1: 0.1}
    model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    # print("coefs:", model.coef_)
    # std_coef = (np.std(X_train, 0) * model.coef_)[0]
    # print(len(std_coef))
    # print(len(feature_list))
    # print("coefs * std max:", np.argmax(np.std(X_train, 0) * model.coef_))
    # print("coefs * std min:", np.argmin(np.std(X_train, 0) * model.coef_))

    # plot_coefs_std(std_coef, feature_list, feature_dict)
    # sns.barplot(feature_list, std_coef)
    # plt.xticks(rotation=90)
    # plt.show()
    score = model.score(X_test, y_test)
    print("Score:", score)
    type12errs = type_1_2_errors(model, X_test, y_test)
    precision = type12errs["precision"]
    recall = type12errs["recall"]
    print(type12errs)
    precision_recall(model, X_test, y_test)

    return score, precision, recall

def feature_importance(model, features_dict):
    pass

def precision_recall(model, X_test, y_test):
    y_score = model.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
              average_precision))
    plt.show()

def type_1_2_errors(model, X_test, y_test):
    preds = model.predict(X_test)
    FP = confusion_matrix(y_test, preds)[0][1]
    FN = confusion_matrix(y_test, preds)[1][0]
    TP = confusion_matrix(y_test, preds)[1][1]
    TN = confusion_matrix(y_test, preds)[0][0]

    precision = TP/(TP + FP)
    recall = TP/(TP + FN)

    return {"False positive": FP,
            "False negative": FN,
            "True positive": TP,
            "True Negative": TN,
            "precision": precision,
            "recall": recall}

def main():
    score_list = []
    precision_list = []
    recall_list = []
    df = mongo_to_df()
    data = convert_data(df)
    ml_data, labels, feature_list, feature_dict = clean_data(data)
    score, precision, recall = train_and_test(ml_data, labels, feature_list, feature_dict)
    score_list.append(score)
    precision_list.append(precision)
    recall_list.append(recall)
    avg_score = np.mean(score_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    print("Avg score:", avg_score)
    print("Avg precision:", avg_precision)
    print("Avg recall:", avg_recall)

if __name__ == "__main__":
    main()
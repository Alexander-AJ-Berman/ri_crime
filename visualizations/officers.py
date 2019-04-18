"""
Multiple pie charts where
slice of X = (# of arrests of ethnicity X)/(total # of arrests),
where each officer has their own chart
"""


def pie_one_officer(officer, db_arrests):
    """
    Method that, given an officer and the arrests database
    :param officer: a string representing name of the officer
    :param db_arrests: arrests db
    :return:
    """

    # 1. officer_arrests = Get all arrests from db_arrests by officer
    # 2. Count # arrests of each race/ethnicity in officer_arrests
    # 3. pie chart that

def pie_all_officers():
    # 1. get list of all officer names

    # 2. iterate through
    for officer in officer_list:
        pie_one_officers(officer, db_arrests)
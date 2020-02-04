def make_parser(cls):
    def parse_commas(text):
        return cls(re.sub(r'[^-+\d.]', '', text))

    return parse_commas


def get_column(data_frame, column):
    """From pandas dataframe gets column of data"""
    return data_frame[data_frame.axes[1][column]]


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 60 + int(m) + int(s) / 1000


def get_question(row_num, list_questions):
    returning = 0
    if row_num == 704:
        print("A")
    for i in range(0, len(list_questions)):
        start = list_questions[i][0]
        end = list_questions[i][1]
        if row_num >= start and row_num <= end:
            returning = i + 1
            break
    return returning


def get_range(column_time, time_stamps, question):
    start = -1
    end = -1
    for j in range(0, len(column_time)):
        if column_time[j] <= time_stamps[question - 1]:
            start = j
        elif column_time[j] >= time_stamps[question]:
            end = j
            break
    return start, end


if __name__ == "__main__":
    import re
    import pandas as pd
    import os
    from os import listdir
    from os.path import isfile, join, isdir
    import numpy as np
    import datetime
    from dateutil import parser
    from Data import Region
    from Data import Data
    from Data import DataArray

    path = r"data/questions"
    path_processed = r"data/processed"

    to_int = make_parser(int)

    time_stapms = []
    for i in range(1, 35):
        time_stapms.append("Time " + str(i))
    df = pd.read_csv(r"data/question_time.csv", parse_dates=[time_stapms])

    # print(df.dtypes.__len__())
    print(df)
    axes = df.axes
    # print(axes[1])  # axes[1][1] prvi red
    print(df.dtypes)

    basePath = os.path.dirname(os.path.abspath(__file__))
    filePath = basePath + "\\data\\questions"
    users = [f for f in listdir(filePath) if isdir(join(filePath, f))]  # if isfile(join(filePath, f))]
    print(users)

    arr = list()
    # df.head().itertuples() - More optimized 150x times faster, change if speed needed
    # but be aware call needs to be row.User and has only row in for loop
    for index, row in df.head().iterrows():  # bez head prolazi kroz sve

        folder_name = "user" + str(row["User"])
        file_name = "User " + str(row["User"]) + "_all_gaze.csv"
        print("I=" + str(index), folder_name)  # row[1], row[2], row[3], row[4], row[5])
        times = row.values[0]  # TIME FROM SPLIT TIME CSV
        list_times = times.split(" ")
        sec_times = [get_sec(time_user) for time_user in list_times]  # Got seconds that are same as 1 col in splitTime
        print(sec_times)
        dfu = pd.read_csv(filePath + "\\" + folder_name + "\\" + file_name)
        question_times = get_column(dfu, 0)  # Analysis data per user
        questions = list()
        for question in range(1, 34):
            starting, ending = get_range(question_times, sec_times, question)  # Not optimized
            # print("Q=" + str(question) + " start=" + str(starting) + " ending=" + str(ending))
            if question > 1:  # Fix that questions don't overlap
                if questions[len(questions) - 1][1] > starting:
                    starting = questions[len(questions) - 1][1] + 1
            questions.append([starting, ending])

        # dfu.astype({dfu.axes[1][1]:""})

        row_array = np.arange(len(dfu))  # get row numbers for dataframe 0..row_size

        dfu[dfu.axes[1][6]] = [get_question(user_iter, questions) for user_iter in
                               row_array]

        print(dfu[questions[2][0]:questions[2][1]])  # Example of selected rows for

        # dfu[dfu.axes[1][6]] = 1
        # print(dfu.axes[1])
        # print("----------")
        print(dfu.dtypes)
        # print("----------")
        print(dfu)
        new_headers = [dfu.axes[1][0], dfu.axes[1][1], dfu.axes[1][2], dfu.axes[1][3], dfu.axes[1][4], dfu.axes[1][5],
                       "Question"]
        for i, g in dfu.groupby(dfu.axes[1][6]):
            if i == 0: # All other that are not question from 1 to 33
                continue
            dirName = path_processed + "/" + folder_name
            if not os.path.exists(dirName):
                os.mkdir(dirName)
                print("Directory ", dirName, " Created ")
            g.to_csv(dirName + "\\" + '{}.csv'.format(i), header=new_headers, index_label=False)

        # Foreach in timestamp ( load User folder all_gazes and split)

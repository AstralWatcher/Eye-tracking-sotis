import re
import pandas as pd
import os
from os import listdir
from os.path import isfile, join, isdir


def for_each_user(callback, name_csv):
    """Some advance function if have time to experiment with python"""
    print("Begin")


def merge_regions(data_frame):
    print(data_frame)
    last = -2
    list_indexes_remove = []
    duration = 0
    list_duration = data_frame["FPOGD"].tolist()  # get_column(data_frame, 4)
    # list_duration = [f for f in data_frame["FPOGD"]]

    list_regions = data_frame["Regions"].tolist()  # get_column(data_frame, 7)
    list_questions = data_frame["Question"].tolist()  # get_column(data_frame, 7)
    first_index = 0
    for i in range(0, len(list_duration)):
        if list_questions[i] == 0:
            list_indexes_remove.append(i)
            continue
        if last == list_regions[i]:
            duration = duration + list_duration[i]
            list_indexes_remove.append(i)
        else:

            # Remove data slips that are not 1 to 33 questions
            list_duration[first_index] = duration
            first_index = i
            duration = list_duration[i]
            last = list_regions[i]

    data_frame["FPOGD"] = list_duration
    data_frame = data_frame.drop(list_indexes_remove)
    print(data_frame)
    return data_frame


def clean_regions(data_frame, time=0.1, use_fpogv=False, time_for_other=0.2):
    to_remove = list()
    for row in data_frame.itertuples():
        if row.FPOGD < time or row.FPOGD < time_for_other and row.Regions == -1:
            to_remove.append(row.Index)
        if use_fpogv and not row.FPOGV:
            to_remove.append(row.Index)

    data_frame = data_frame.drop(to_remove)
    return data_frame


if __name__ == "__main__":
    import re
    import pandas as pd
    import os
    from os import listdir
    from os.path import isfile, join, isdir

    PATH = r"data\processed"
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = BASE_PATH + "\\" + PATH
    FILE_NAME = "data.csv"
    list_users = [f for f in listdir(FILE_PATH) if isdir(join(FILE_PATH, f))]
    print(list_users)
    for user in list_users:
        print("Processing " + str(user))
        path_user = PATH + "\\" + user + "\\" + FILE_NAME
        dfr = pd.read_csv(path_user, index_col=False)
        new_df = merge_regions(data_frame=dfr)
        new_df = new_df.drop(columns=["TIME", "TIMETICKS"]) #"FPOGX", "FPOGY"
        print(new_df)

        new_df = clean_regions(new_df, use_fpogv=True)
        print("CLEANING NOISE")
        print(new_df)

        new_df.to_csv(PATH + "\\" + user + "\\" + 'data_cleaned.csv', mode='w', index=False, index_label=False)
        # break

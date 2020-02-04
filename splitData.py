def make_parser(cls):
    def parse_commas(text):
        return cls(re.sub(r'[^-+\d.]', '', text))

    return parse_commas


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
    #df.head().itertuples() - More optimized
    for index, row in df.head().iterrows():  # bez head prolazi kroz sve

        folder_name = "user" + str(row["User"])
        file_name = "User " + str(row["User"]) + "_all_gaze.csv"
        print("I=" + str(index), folder_name)  # row[1], row[2], row[3], row[4], row[5])
        times = row.values[0]
        list_times = times.split(" ")
        print(times)
        dfu = pd.read_csv(filePath + "\\" + folder_name + "\\" + file_name)
        #dfu.astype({dfu.axes[1][1]:""})
        print(dfu.axes[1])
        print("----------")
        print(dfu.dtypes)
        print("----------")
        print(dfu)
        #TODO parse time
        #TODO 2, 2 iters to check which time is needed
        #TODO 3, split in folders
        # Foreach in timestamp ( load User folder all_gazes and split)

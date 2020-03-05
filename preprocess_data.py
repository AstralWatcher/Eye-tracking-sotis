import pandas as pd
def make_parser(cls):
    def parse_commas(text):
        return cls(re.sub(r'[^-+\d.]', '', text))

    return parse_commas


def get_times_by_user(df_times, user):
    cols = get_column(df_times, 1)
    found = cols[cols == user].index[0]
    return get_row(df_times, found)


def get_row(data_frame, row):
    """From pandas dataframe gets row of data, starts with 0 as first row, returns list"""
    return (data_frame.iloc[[row]]).values[0][0]


def get_column(data_frame, column):
    """From pandas dataframe gets column of data, starts with 0 as first column"""
    return data_frame[data_frame.axes[1][column]]


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 60 + int(m) + int(s) / 1000


def get_region(pos_x, pos_y, df_pos):
    for row_in_reg in df_pos.itertuples():
        is_in_x = row_in_reg.X + row_in_reg.Xlen >= pos_x >= row_in_reg.X
        is_in_y = row_in_reg.Y + row_in_reg.Ylen >= pos_y >= row_in_reg.Y
        if is_in_x and is_in_y:
            return row_in_reg.Sem
    return -1


def get_question(row_num, list_questions):
    returning = 0
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


def process_time_data(df_gaze, list_times_got, num_question=34):
    """
    Return row numbers from which question starts and ends from time
    :param df_gaze: data frame from gazepoint
    :param list_times_got: list of times to split per questions em. 3, 00:02:914, 00:11:659, 00:26:816, (m:sec:mili-sec)
    :param num_question: number of questions in list_times_got ( for 33 questions needed 35 in csv)
    :return:
    """
    list_times = list_times_got.split(" ")
    sec_times = [get_sec(time_user) for time_user in list_times]  # Got seconds that are same as 1 col in splitTime
    # print(sec_times)

    question_times = get_column(df_gaze, 0)  # Analysis data per user
    questions = list()
    for question in range(1, num_question):
        starting, ending = get_range(question_times, sec_times, question)  # Not optimized
        # print("Q=" + str(question) + " start=" + str(starting) + " ending=" + str(ending))
        if question > 1:  # Fix that questions don't overlap
            if questions[len(questions) - 1][1] > starting:
                starting = questions[len(questions) - 1][1] + 1
        questions.append([starting, ending])
    return questions


def save_preprocessed(df_got_data, df_got_regions, list_times, directory_path, csv_name='data.csv', res_x=1920,
                      res_y=1080, save=True):
    """
    Save preprocessed csv, which will be assigned Region based on time series
    :param df_got_data: pandas data_frame gazepoint data
    :param df_got_regions: pandas data_Frame for regions X,Y,XLen,Y_len,Region name (so regions will be assigned)
    :param list_times: list of times to split per questions em. 3, 00:02:914, 00:11:659, 00:26:816, (m:sec:mili-sec)
    :param directory_path: path which result will be saved
    :param csv_name: filename of csv result
    :param res_x: X resolution of monitor which was recorded gazepoint
    :param res_y: Y resolution of monitor which was recorded
    :param save: save to file
    :return:
    """
    import numpy as np
    questions = process_time_data(df_got_data, list_times)

    row_array = np.arange(len(df_got_data))  # get row numbers for dataframe 0,1,2..row_size

    df_got_data[df_got_data.axes[1][6]] = [get_question(user_iter, questions) for user_iter in
                                           row_array]  # Adding to 6th col which question it is for

    coordinates_list = [[row_coordinates.FPOGX * res_x, row_coordinates.FPOGY * res_y] for row_coordinates in
                        df_got_data.itertuples()]

    df_got_data["Regions"] = [get_region(cords[0], cords[1], df_got_regions) for cords in coordinates_list]

    new_headers = ["TIME", "TIMETICKS", df_got_data.axes[1][2], df_got_data.axes[1][3], df_got_data.axes[1][4],
                   df_got_data.axes[1][5],
                   "Question", "Regions"]
    df_got_data.columns = new_headers
    if save:
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
            print("Directory ", directory_path, " Created ")
        df_got_data.to_csv(directory_path + "\\" + csv_name, header=new_headers, mode='w', index=False,
                           index_label=False)
    return df_got_data


def read_time_csv(new_path=r"data/question_time.csv", size=35):
    time_stapms = []
    for i in range(1, size):
        time_stapms.append("Time " + str(i))
    df_times = pd.read_csv(new_path, parse_dates=[time_stapms])
    return df_times


def read_regions_csv(new_path=r"data/regions.csv"):
    return pd.read_csv(new_path)


def find_user_in_time(df_time, num):
    try:
        found_col = df_time["User"].tolist().index(num)
        return df_time.iloc[found_col + 1]
    except ValueError:
        return None


if __name__ == "__main__":
    import re
    import os
    from os import listdir
    from os.path import isfile, join, isdir
    import numpy as np

    path = r"data/questions"
    path_processed = r"data/processed"

    # to_int = make_parser(int)

    df_times = read_time_csv()

    df_regions = read_regions_csv()

    # sem_zero = get_region(400, 269, dfr) #check if function get region works
    # print(sem_zero)

    RES_X = 1920
    RES_Y = 1080
    # print(dfr)

    # print(df.dtypes.__len__())
    # print(df)
    axes = df_times.axes
    # print(axes[1])  # axes[1][1] prvi red
    # print(df.dtypes)

    basePath = os.path.dirname(os.path.abspath(__file__))
    filePath = basePath + "\\data\\questions"
    users = [f for f in listdir(filePath) if isdir(join(filePath, f))]  # if isfile(join(filePath, f))]
    print(users)

    arr = list()
    # df.itertuples() - More optimized 150x times faster iterrows
    index = 0
    for row in df_times.itertuples():  # bez head prolazi kroz sve
        index = index + 1
        folder_name = "user" + str(row.User)

        file_name = "User " + str(row.User) + "_all_gaze.csv"
        print("I=" + str(index), folder_name)  # row[1], row[2], row[3], row[4], row[5])
        times = row[1]  # TIME FROM SPLIT TIME CSV

        df_data = pd.read_csv(filePath + "\\" + folder_name + "\\" + file_name)

        dirName = path_processed + "/" + folder_name
        save_preprocessed(df_data, df_regions, times, dirName, "data.csv", RES_X, RES_Y)

        # dfu.astype({dfu.axes[1][1]:""})
        # print(dfu[questions[2][0]:questions[2][1]])  # Example of selected rows for

        # dfu[dfu.axes[1][6]] = 1
        # print(dfu.axes[1])
        # new headers dropped columns
        # dfu.axes[1][0]  #TIME(...
        #  dfu.axes[1][1] #TIMETICKS

        # for i, g in dfu.groupby(dfu.axes[1][6]): #GROUP BY
        #     if i == 0:  # All other that are not question from 1 to 33
        #         continue
        #     dirName = path_processed + "/" + folder_name
        #     if not os.path.exists(dirName):
        #         os.mkdir(dirName)
        #         print("Directory ", dirName, " Created ")
        #     g.to_csv(dirName + "\\" + '{}.csv'.format(i), header=new_headers, mode='w', index=False, index_label=False)

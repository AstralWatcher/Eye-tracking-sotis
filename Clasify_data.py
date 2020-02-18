def get_users(path="data\\processed"):
    import os
    from os import listdir
    from os.path import isfile, join, isdir
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = BASE_PATH + "\\" + path
    list_users = [f for f in listdir(FILE_PATH) if isdir(join(FILE_PATH, f))]
    return FILE_PATH, list_users


def get_label(duration, avrage):
    if duration < avrage:
        lab = 0
    elif avrage * 2 > duration > avrage:
        lab = 1
    else:
        lab = 2
    return lab


if __name__ == "__main__":
    import pandas as pd

    FILE_NAME = "data_cleaned.csv"
    file_path, LIST_USERS = get_users()

    for user in LIST_USERS:
        print("Processing " + str(user))
        path_user = file_path + "\\" + user + "\\" + FILE_NAME
        dfr = pd.read_csv(path_user, index_col=False)
        list_duration = dfr["FPOGD"].tolist()
        avg_duration = sum(list_duration) / len(list_duration)
        print("Suma = " + str(avg_duration))
        print(dfr)

        dfr["FPOGD"] = [get_label(fpog,avg_duration) for fpog in dfr["FPOGD"].tolist()]

        print(dfr)

        #TODO SOME LOGIC FOR CLASSIFICATION?

        # dfr.to_csv(PATH + "\\" + user + "\\" + 'data_cleaned.csv', mode='w', index=False, index_label=False)

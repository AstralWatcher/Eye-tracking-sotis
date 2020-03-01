from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


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


def normalize(got_list):
    lens = [len(nparray) for nparray in got_list]
    max_num = max(lens)
    print("Normalizing to size of vector" + str(max_num))

    new_list = []

    for element in got_list:
        if len(element) < max_num:
            new_element = np.pad(element, (0, max_num - len(element)), 'constant')
            new_list.append(new_element)
    return new_list


def get_data_to_cluster(file_name: str = "data_cleaned.csv", norm: bool = True, func=None):
    file_path, list_users = get_users()
    data_to_send: list = list()
    for user in list_users:
        print("Processing " + str(user))
        path_user = file_path + "\\" + user + "\\" + file_name
        dfr = pd.read_csv(path_user, index_col=False)
        list_duration = dfr["FPOGD"].tolist()
        avg_duration = sum(list_duration) / len(list_duration)
        dfr["FPOGD"] = [get_label(fpog, avg_duration) for fpog in dfr["FPOGD"].tolist()]
        for i, g in dfr.groupby("Question"):  # GROUP BY
            seq = []
            if i == 0:  # All other that are not question from 1 to 33
                continue
            for item in g.itertuples():
                seq.append(item.FPOGX)
                seq.append(item.FPOGY)
                seq.append(item.Regions)
                seq.append(item.FPOGD)
                # [item.FPOGX, item.FPOGY, item.Regions, item.FPOGD])
            seq = np.array(seq)
            data_to_send.append(seq)
    if norm and func is None:
        array_to_return = normalize(data_to_send)
    elif norm:
        array_to_return = func(data_to_send)
    else:
        array_to_return = data_to_send
    return array_to_return


def simple_plot(labels, core_samples_mask):
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = array_to_send[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = array_to_send[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    plt.title('Estimated number of clusters: %d' % n_clusters_ + ' Number of noise data: ' + str(n_noise_))
    plt.show()


def statistics(_labels):
    from math import fabs
    n_clusters_ = len(set(_labels)) - (1 if -1 in _labels else 0)
    n_noise_ = list(_labels).count(-1)
    print("Number of clusters:" + str(n_clusters_))  # Number of clusters in labels, ignoring noise if present.
    print("Number of noise (Unclustered) " + str(n_noise_))
    print("Number of clustered:" + str(fabs(n_noise_ - len(_labels))))
    print("Size of vector:" + str(len(_labels)))


if __name__ == "__main__":
    import pandas as pd

    array_to_send = get_data_to_cluster(func=normalize)
    clustering = DBSCAN(eps=3.2, min_samples=2).fit(array_to_send)  # 8.6
    array_to_send = np.array(array_to_send)

    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_
    print(labels)  # Show all values of labels
    statistics(_labels=labels)
    # TODO Svaku sekvencu sacuvam sa labelom
    # TODO Onda klasifikator (nadgledano ucenje)
    DELIMITER: str = "#"
    NEW_LINE = "\n"
    to_file = ""
    for questions_all_user in range(0, len(array_to_send)):
        list_of_data = list(array_to_send[questions_all_user])
        list_points = ','.join([str(elem) for elem in list_of_data])
        label = labels[questions_all_user]
        to_file = to_file + list_points + DELIMITER + str(label) + NEW_LINE
    f = open(file="clustered_data.txt", mode="w")
    f.write(to_file)
    f.close()

    simple_plot(labels=labels, core_samples_mask=core_samples_mask)
    end_program = input("Press any button to end")

    # dfr.to_csv(PATH + "\\" + user + "\\" + 'data_cleaned.csv', mode='w', index=False, index_label=False)

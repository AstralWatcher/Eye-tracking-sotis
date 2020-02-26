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
    print(max_num)

    new_list = []

    for element in got_list:
        if len(element) < max_num:
            new_element = np.pad(element, (0, max_num - len(element)), 'constant')
            new_list.append(new_element)
    return new_list


if __name__ == "__main__":
    import pandas as pd

    FILE_NAME = "data_cleaned.csv"
    file_path, LIST_USERS = get_users()
    data_to_send: list = list()
    for user in LIST_USERS:
        print("Processing " + str(user))
        path_user = file_path + "\\" + user + "\\" + FILE_NAME
        dfr = pd.read_csv(path_user, index_col=False)
        list_duration = dfr["FPOGD"].tolist()
        avg_duration = sum(list_duration) / len(list_duration)
        #print("Suma = " + str(avg_duration))
        dfr["FPOGD"] = [get_label(fpog, avg_duration) for fpog in dfr["FPOGD"].tolist()]
        #print(dfr)

        for i, g in dfr.groupby("Question"):  # GROUP BY
            seq = []
            if i == 0:  # All other that are not question from 1 to 33
                continue
            #print(g)
            for item in g.itertuples():
                seq.append(item.FPOGX)
                seq.append(item.FPOGY)
                seq.append(item.Regions)
                seq.append(item.FPOGD)
                # [item.FPOGX, item.FPOGY, item.Regions, item.FPOGD])
            # data_to_send.append(np.array(seq))
            seq = np.array(seq)
            data_to_send.append(seq)
            #print("Quesiton=" + str(i))
            # g.to_csv(dirName + "\\" + '{}.csv'.format(i), header=new_headers, mode='w', index=False, index_label=False)
    array_to_send = normalize(data_to_send)
    clustering = DBSCAN(eps=3.2, min_samples=2).fit(array_to_send) #8.6
    print(clustering.labels_)
    array_to_send = np.array(array_to_send)
    # TODO SOME LOGIC FOR CLASSIFICATION?

    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # #############################################################################
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

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    #Svaku sekvencu sacuvam sa labelom
    #Onda klasifikator (nadgledano ucenje)
    print("")

        # dfr.to_csv(PATH + "\\" + user + "\\" + 'data_cleaned.csv', mode='w', index=False, index_label=False)

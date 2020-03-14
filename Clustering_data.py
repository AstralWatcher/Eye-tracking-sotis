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


def get_label(duration, average):
    """
    Return values:
    0 - below average
    1 - above average and less than 2x
    2 - 2x above average
    :param duration: duration of gaze
    :param average: avreage peer user
    :return: int
    """
    if duration < average:
        lab = 0
    elif average * 2 > duration > average:
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


def map_to_dna_2(duration_mapped, region):
    """

    :param duration_mapped: 0,1,2 Mapped by avg
    :param region: region gaze
    :return:
    """
    return [region for _ in range(0, duration_mapped)]


def map_prepare_dna_map_1(regs=None, durs=None, debug=True):
    mapper = ""
    if regs is None:
        regs = [-1, 0, 1, 2, 3, 4, 5]
    if durs is None:
        durs = [0, 1, 2]
    val_dict = dict()
    unique = 0
    mapper = "Duration,Region --> Unique Mapped value"
    for i in range(0, len(regs)):
        for j in range(0, len(durs)):
            unique += 1
            mapper = mapper + "\n" + str(regs[i]) + "," + str(durs[j]) + ": " + str(unique)
            val_dict[regs[i] + (durs[j] + 5) * 10] = unique
    if debug:
        fp = open("map_dna.txt", "w")
        fp.write(mapper)
        fp.close()
        print(val_dict)
    return val_dict


# TODO FINISH MAPPPING


def map_to_dna_1(duration_mapped, region, val_dict):
    """

    :param duration_mapped:
    :param region:
    :param val_dict: value from map_prepare_dna_map_1()
    :return:
    """
    ret = val_dict[region + 10 * (duration_mapped + 5)]
    if ret is not None:
        return ret
    else:
        print("Warn invalid map to dna")
    return None


def clamp(x):
    return max(0, min(x, 255))


def format_to_hexa(r, g, b):
    r = int(round(r * 255))
    g = int(round(g * 255))
    b = int(round(b * 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))


def get_data_to_cluster_from_df(dfr, norm: bool = False, func=None):
    mapper = map_prepare_dna_map_1(debug=False)
    data_to_send = list()
    list_duration = dfr["FPOGD"].tolist()
    avg_duration = sum(list_duration) / len(list_duration)
    dfr["FPOGD"] = [get_label(fpog, avg_duration) for fpog in dfr["FPOGD"].tolist()]
    for i, g in dfr.groupby("Question"):  # GROUP BY
        seq = []
        if i == 0:  # All other that are not question from 1 to 33
            continue
        for item in g.itertuples():
            seq.append(map_to_dna_1(item.FPOGD, item.Regions, mapper))
        seq = np.array(seq)
        data_to_send.append(seq)
    if norm and func is None:
        array_to_return = normalize(data_to_send)
    elif norm:
        array_to_return = func(data_to_send)
    else:
        array_to_return = data_to_send
    return array_to_return


def get_data_to_cluster(file_name: str = "data_cleaned.csv", norm: bool = True, func=None):
    file_path, list_users = get_users()
    data_to_send: list = list()
    mapper = map_prepare_dna_map_1()
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
                # seq.append(item.FPOGX)
                # seq.append(item.FPOGY)
                # seq.append(item.Regions)
                # seq.append(item.FPOGD)
                seq.append(map_to_dna_1(item.FPOGD, item.Regions, mapper))
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


def plot_tsne(array_to_plot, labels, perplexity=100):
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns
    tsne = TSNE(n_components=3, verbose=1, perplexity=perplexity, n_iter=5000)
    tsne_results = tsne.fit_transform(array_to_plot)
    print('t-SNE done!')
    ff = tuple(labels.reshape((750,)))
    map_to_pd = {'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1], 'y': ff}
    df = pd.DataFrame(map_to_pd)
    colour_pelet = sns.color_palette("hls", len(np.unique(map_to_pd['y'])))
    # fig = plt.figure()
    plot3d(labels, tsne_results, "TSNE ")
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=colour_pelet,
    #     data=df,
    #     legend="full",
    #     alpha=0.3
    # )
    # plt.show()


def plot3d(labels, array_to_plot, title=""):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = ax.add_subplot(111, projection='3d')
    if len(array_to_plot[0]) != 3:
        print("Not displaying 3d vector")

    unique_labels = set(labels)  # Black removed and is used for noise instead.
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xyz = array_to_plot[class_member_mask & core_samples_mask]

        plt_x = [el[0] for el in xyz]
        plt_y = [el[1] for el in xyz]
        plt_z = [el[2] for el in xyz]
        # TODO onda napraviti novi ulazi i pozvati preprocess,process,sgt i pca => cluster data
        colour = format_to_hexa(col[0], col[1], col[2])
        ax.scatter(xs=plt_x, ys=plt_y, zs=plt_z, zdir='z', s=20, c=colour, marker="o", depthshade=True)

        xyz = array_to_plot[class_member_mask & ~core_samples_mask]
        plt_x = [el[0] for el in xyz]
        plt_y = [el[1] for el in xyz]
        plt_z = [el[2] for el in xyz]

        ax.scatter(xs=plt_x, ys=plt_y, zs=plt_z, zdir='z', s=14, c=colour, marker="^", depthshade=True)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    plt.title(title + 'Estimated number of clusters: %d' % n_clusters_ + ' Number of noise data: ' + str(n_noise_))
    plt.show()
    print("Finished")
    return 0


# def simple_plot(labels, core_samples_mask, array_to_plot):
#     # Plot result
#     import matplotlib.pyplot as plt
#
#     # Black removed and is used for noise instead.
#     unique_labels = set(labels)
#     colors = [plt.cm.Spectral(each)
#               for each in np.linspace(0, 1, len(unique_labels))]
#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             # Black used for noise.
#             col = [0, 0, 0, 1]
#
#         class_member_mask = (labels == k)
#
#         xy = array_to_send[class_member_mask & core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                  markeredgecolor='k', markersize=14)
#
#         xy = array_to_send[class_member_mask & ~core_samples_mask]
#         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#                  markeredgecolor='k', markersize=6)
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)
#
#     plt.title('Estimated number of clusters: %d' % n_clusters_ + ' Number of noise data: ' + str(n_noise_))
#     plt.show()


def statistics(_labels, debug=True, it=None):
    from math import fabs
    n_clusters_ = len(set(_labels)) - (1 if -1 in _labels else 0)
    n_noise_ = list(_labels).count(-1)
    if it is not None:
        print("Epsilon" + str(it))
    if debug:
        print("Number of clusters:" + str(n_clusters_))  # Number of clusters in labels, ignoring noise if present.
        print("Number of noise (Unclustered) " + str(n_noise_))
        print("Number of clustered:" + str(fabs(n_noise_ - len(_labels))))
        print("Size of vector:" + str(len(_labels)))
    return n_noise_


def get_optimal(embedding):
    list_y = list()
    list_clusters = list()
    _GOTO = 25
    list_x = [item / 100 for item in range(1, _GOTO, 1)]
    for i in range(1, _GOTO, 1):
        clustering = DBSCAN(eps=i / 100, min_samples=2).fit(embedding)  # array_to_send  # 8.6
        list_y.append(statistics(clustering.labels_, it=i / 100))
        list_clusters.append(len(np.unique(np.array(clustering.labels_))))
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.title('Unclustered dependend on Epsilon')
    plt.plot(list_x, list_y)
    plt.subplot(2, 1, 2)
    plt.title('Number of clusters dpending on Epsilon')
    plt.plot(list_x, list_clusters)
    plt.show()


def hirarhical(X, x_den):
    import matplotlib.pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    import scipy.cluster.hierarchy as shc
    plt.title("Eye gaze Dendograms")
    dend = shc.dendrogram(shc.linkage(x_den, method='ward'))
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)
    print(cluster.labels_)
    plot3d(cluster.labels_, X)
    print("DONE hir")
    return cluster


def db_scan(epsilon, X):
    return DBSCAN(eps=epsilon, min_samples=2).fit(X)  # array_to_send


if __name__ == "__main__":
    import pandas as pd

    array_to_send = get_data_to_cluster(func=None, norm=False)
    # array_to_send_2 = get_data_to_cluster(norm=True)

    from sgt import Sgt

    # https://towardsdatascience.com/sequence-embedding-for-clustering-and-classification-f816a66373fb

    sgt = Sgt(kappa=10, lengthsensitive=False)
    embedding = sgt.fit_transform(corpus=array_to_send)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    pca.fit(embedding)
    X = pca.transform(embedding)

    pca_den = PCA(n_components=2)
    pca_den.fit(embedding)
    x_den = pca.transform(embedding)

    get_optimal(X)
    # epsilon = float(input("Unesite Epsilon:"))
    epsilon = 0.1

    array_to_send = np.array(array_to_send)
    clustering = db_scan(epsilon, X)

    clustering_hir = hirarhical(X, x_den)

    labels = clustering_hir.labels_  # TODO Change to clustering.labels_ if want DBSCAN

    plot_tsne(embedding, labels, 35)

    import Clasify

    Clasify.class_distribution([lb for lb in labels])

    if len(X[0]) > 3:
        pca2 = PCA(n_components=3)
        pca2.fit(X)
        X = pca2.transform(X)

    print(labels)  # Show all values of labels
    plot3d(labels=labels, array_to_plot=X)
    statistics(_labels=labels)
    # TODO Svaku sekvencu sacuvam sa labelom
    # TODO Onda klasifikator (nadgledano ucenje)
    DELIMITER: str = "#"
    NEW_LINE = "\n"
    to_file = ""
    for questions_all_user in range(0, len(X)):
        list_of_data = list(array_to_send[questions_all_user])  # array_to_send OR X
        list_points = ','.join([str(elem) for elem in list_of_data])
        label = labels[questions_all_user]
        to_file = to_file + list_points + DELIMITER + str(label) + NEW_LINE
    f = open(file="clustered_data_dna_hir_org.txt", mode="w")
    f.write(to_file)
    f.close()

    # simple_plot(labels=labels, core_samples_mask=core_samples_mask)
    # end_program = input("Press any button to end")

    # dfr.to_csv(PATH + "\\" + user + "\\" + 'data_cleaned.csv', mode='w', index=False, index_label=False)

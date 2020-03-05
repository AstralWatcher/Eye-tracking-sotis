import numpy as np
import preprocess_data as ppd
import process_data as procs
import Clustering_data as cd


def prepare_new_input_dna(new_input, user_id):
    """
    Pre-processing done in clustering, preparing for clasification
    :param new_input: data frame for new input
    :param user_id: user id for getting row of times from questions_time.csv to process splitting
    :return: X that is 3d, processed with PCA and SGT with additional pre-processing and mapping
    """
    from sgt import Sgt
    from sklearn.decomposition import PCA
    df_map_corpus = pd.read_csv("map_dna.txt", delimiter=":", names=["Pre", "Posle"])
    df_map_corpus = df_map_corpus.tail(-1)
    corpus = df_map_corpus["Posle"].values

    df_time = ppd.read_time_csv()
    times = ppd.get_times_by_user(df_time, user_id)

    df_regions = ppd.read_regions_csv()

    df_new = ppd.save_preprocessed(new_input, df_regions, times, "", "data.csv", save=False)
    df_new2 = procs.save_merged_regions(df_new, "", "", use_fpgov_cleaning=True, save=False)
    array_to_send = cd.get_data_to_cluster_from_df(df_new2)
    sgt = Sgt(kappa=10, lengthsensitive=False, alphabets=[el for el in corpus])
    embedding = sgt.fit_transform(corpus=array_to_send)

    pca = PCA(n_components=3)
    pca.fit(embedding)
    X = pca.transform(embedding)
    return X


def prepare_new_input_1(new_input, length):
    """
    New input that passes predict for svm
    :param new_input: new input for svm
    :param length: lenght of [X,Y,Qustion,Region,...] vector
    :return: new_input that passes predict for svm
    """
    if len(new_input) < length:
        new_input = list(np.pad(np.array(new_input), (0, length - len(new_input)), 'constant'))
    else:
        new_input = new_input[0:length]
    return new_input


def svm(data_frame, name):
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    X = data_frame.drop('class', axis=1)
    y = data_frame['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,stratify=None)

    # stratSplit = StratifiedShuffleSplit(test_size=0.2, random_state=42)
    # stratSplit.get_n_splits(X, y)
    # X_train = list()
    # X_test = list()
    # y_train = list()
    # y_test = list()
    #
    # for train_index, test_index in stratSplit.split(X, y):
    #     eX_train, eX_test = X[train_index], X[test_index]
    #     ey_train, ey_test = y[train_index], y[test_index]
    #     X_train.append(eX_train)
    #     X_test.append(eX_test)
    #     y_train.append(y_train)
    #     y_test.append(ey_test)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    import joblib
    # now you can save it to a file
    joblib.dump(svclassifier, name + '.pkl')
    # and later you can load it
    # clf = joblib.load('filename.pkl')


def class_distribution(class_list):
    print("Begin class distribution")
    y_unique = np.unique(class_list)
    vals = dict()
    for i in range(0, len(y_unique)):
        vals[y_unique[i]] = class_list.count(y_unique[i])
    print("\nClass   Support\n")
    for key, value in vals.items():
        print(key, '  ->  ', value)
    print("End class distribution")
    return vals


def make_from_array_new_columns_data_frame(dataframe, name="gaze_"):
    """
    data         class
    1,2,3,4,5    dog
    ..
    -->
    col1,col2,col3,col4,col5,class
    1    ,2    ,3   ,4   ,5   ,dog
    :param dataframe:
    :return:
    """
    X = dataframe.drop('class', axis=1)
    y = dataframe['class']
    x_list = X["data"].tolist()
    y_list = y.tolist()
    data = [csv_item.split(",") for csv_item in x_list]
    j = 0
    new_columns = dict()
    for i in range(0, len(data[0])):
        new_col_vals = [val[i] for val in data]
        new_columns[name + str(i)] = new_col_vals
    new_columns["class"] = y_list
    new_pd = pd.DataFrame(new_columns)
    return new_pd


def train_svm(svm_file_name="svm_save", file_name_clustered_data="clustered_data.txt"):
    csv = pd.read_csv(file_name_clustered_data, delimiter="#", names=["data", "class"])
    print(csv)
    fixed_pdf = make_from_array_new_columns_data_frame(csv)
    y = fixed_pdf['class']
    print("Before learning data set distribution")
    class_distribution(y.tolist())
    svm(fixed_pdf, svm_file_name)


def load_svm(name="svm_save"):
    import joblib
    clf = joblib.load(name + '.pkl')
    return clf


def vote(res):
    import numpy as np
    res = np.array(res)
    to: list = np.unique(res).tolist()
    voting = np.zeros(shape=(len(to),)).tolist()
    for el in range(0, len(to)):
        result = (res == to[el])
        voting[el] = len(res[result])
        print("Voting[" + str(to[el]) + "]=" + str(voting[el]))
    voting[to.index(-1)] /= 2

    print("Voting won=" + str(to[voting.index(max(voting))]))
    return to[voting.index(max(voting))]


if __name__ == "__main__":
    import pandas as pd
    import os

    # vote([-1, 1, -1, 2, 3, 1, 1, 1, -1, -1])

    basePath = os.path.dirname(os.path.abspath(__file__)) + "\\data\\questions"
    new_df = pd.read_csv(basePath + "\\user7\\User 7_all_gaze.csv")
    new_data_prepared = prepare_new_input_dna(new_df, 7)

    train_svm("svm_save", "clustered_data_dna.txt")

    model = load_svm("svm_save")

    rez = model.predict(new_data_prepared)
    vote(rez)
    print("Prcoess End")

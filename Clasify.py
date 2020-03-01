import numpy as np


def prepare_new_input(new_input, length):
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


def svm(data_frame):
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    X = data_frame.drop('class', axis=1)
    y = data_frame['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    import joblib
    # now you can save it to a file
    joblib.dump(svclassifier, 'svm.pkl')
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


if __name__ == "__main__":
    import pandas as pd

    csv = pd.read_csv("clustered_data.txt", delimiter="#", names=["data", "class"])
    print(csv)
    fixed_pdf = make_from_array_new_columns_data_frame(csv)
    y = fixed_pdf['class']
    print("Before learning data set distribution")
    class_distribution(y.tolist())
    svm(fixed_pdf)

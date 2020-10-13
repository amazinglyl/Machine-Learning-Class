import numpy as np
import sklearn.svm
import sklearn.datasets
import sklearn.model_selection
import sklearn.utils


def study_C_fix_split(C_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # prepare the training and testing data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

    # your code here
    score = []
    for i in C_range:
        svc = sklearn.svm.SVC(C=i, kernel='linear', random_state=1)
        svc.fit(X_train, y_train)
        s = svc.score(X_test, y_test)
        score.append(s)
    best_C = C_range[score.index(max(score))]
    return best_C


def study_C_cross_validate(C_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # shuffle the data
    X, y = sklearn.utils.shuffle(X, y)

    # your code here
    score = []
    for i in C_range:
        svc = sklearn.svm.SVC(C=i, kernel='linear', random_state=1)
        s = sklearn.model_selection.cross_val_score(svc, X, y=y)
        score.append(sum(s))
    best_C = C_range[score.index(max(score))]

    return best_C


def study_C_gridCV(C_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # shuffle the data
    X, y = sklearn.utils.shuffle(X, y)

    # your code here
    parameters = {'kernel': ['linear'], 'C': C_range}
    svc = sklearn.svm.SVC()
    clf = sklearn.model_selection.GridSearchCV(svc, parameters)
    clf.fit(X, y)
    print(clf.cv_results_)
    best_C = C_range[clf.best_index_]

    return best_C


def study_C_and_sigma_gridCV(C_range, sigma_range):
    # load the data
    data = sklearn.datasets.load_breast_cancer()
    X, y = data["data"], data["target"]

    # shuffle the data
    X, y = sklearn.utils.shuffle(X, y)

    # your code here
    parameters = {'C': C_range, 'gamma': sigma_range}
    svc = sklearn.svm.SVC()
    clf = sklearn.model_selection.GridSearchCV(svc, parameters)
    clf.fit(X, y)
    #print(clf.best_score_, clf.best_index_, clf.best_params_)
    print(clf.cv_results_)
    best_C = 0

    return best_C


if __name__ == "__main__":
    C_range = np.arange(1, 4, 1)
    gamma = np.arange(5, 8, 1)
    print(C_range)
    # print(study_C_fix_split(C_range))
    # print(study_C_cross_validate(C_range))
    study_C_gridCV(C_range)
    study_C_and_sigma_gridCV(C_range, gamma)

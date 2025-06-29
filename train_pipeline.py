import csv

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.utils import shuffle

from Result import Result
from create_hyperparameter_sets import hyper_parameter_sets


def calc_accuracy(x_set, y_set, _model):
    _acc = _model.score(x_set, y_set)
    # accuracy per class
    _idx_m = y_set == "m"
    _idx_f = y_set == "f"

    _acc_m = _model.score(x_set[_idx_m], y_set[_idx_m])
    _acc_f = _model.score(x_set[_idx_f], y_set[_idx_f])
    return _acc, _acc_m, _acc_f


def set_model(hyper_parameters):
    class_weight = hyper_parameters["class_weight"]
    if hyper_parameters["model"] == "SVC":
        kernel = hyper_parameters["kernel"]
        gamma = hyper_parameters["gamma"]
        model = svm.SVC(class_weight=class_weight, kernel=kernel, gamma=gamma)
    else:
        model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    return model


### load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")  # 6k male (m), 2k female (f)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")  # 2k male (m), 2k female (f)

### Further split training data into train and dev data
X_dev = X_train[5700:6300]
y_dev = y_train[5700:6300]

X_train = np.concatenate((X_train[:5700], X_train[6300:]), axis=0)
y_train = np.concatenate((y_train[:5700], y_train[6300:]), axis=0)

idx_m = y_dev == "m"
idx_f = y_dev == "f"
sum_males = idx_m.sum()
sum_females = idx_f.sum()

print(f"Split development data with {sum_males} males and {sum_females} females")


### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_train_scaler = X_train = scaler.fit_transform(X_train)
X_dev_scaler = X_dev = scaler.transform(X_dev)
X_test_scaler = X_test = scaler.transform(X_test)

# pca = PCA(n_components=75)
# X_train_pca = pca.fit_transform(X_train)
# X_dev_pca = pca.transform(X_dev)
# X_test_pca = pca.transform(X_test)
#
# pca = PCA(n_components=75)
# X_train_scaler_pca = pca.fit_transform(X_train_scaler)
# X_dev_scaler_pca = pca.transform(X_dev_scaler)
# X_test_scaler_pca = pca.transform(X_test_scaler)


results = []

for hyper_parameters in hyper_parameter_sets:
    model = set_model(hyper_parameters)

    ### training
    if hyper_parameters["normalization"] == "scaler":
        _X_train = X_train_scaler
        _X_dev = X_dev_scaler
    # elif hyper_parameters["normalization"] == "pca":
    #     _X_train = X_train_pca
    #     _X_dev = X_dev_pca
    # elif hyper_parameters["normalization"] == "scalar+pca":
    #     _X_train = X_train_scaler_pca
    #     _X_dev = X_dev_scaler_pca
    else:
        _X_train = X_train
        _X_dev = X_dev

    model.fit(_X_train, y_train)

    print(f"Results for hyper-parameter {hyper_parameters}")
    acc, acc_m, acc_f = calc_accuracy(_X_dev, y_dev, model)
    print(f"Acc: {acc:.4f}")

    results.append(Result(hyper_parameters, acc, acc_m, acc_f))

results = sorted(results)
with open('derived data/data.csv', mode="w", newline="") as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(["Model", "weight", "kernel", "gamma", "normalization", "Acc", "M_Acc", "F_Acc"])

for result in results:
    print(result)
    data = list(result.params.values()) + [result.total_acc, result.m_acc, result.f_acc]
    with open('derived data/data.csv', mode="a", newline="") as file:
        formatted_row = [f"{x:.10f}" if isinstance(x, float) else x for x in data]
        writer = csv.writer(file, delimiter=';')
        writer.writerow(formatted_row)

best_model = results[0]
print(f"\nBest model: {best_model}")
model = set_model(best_model.params)
#model = svm.SVC(class_weight='balanced')


### load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy") # 6k male (m), 2k female (f)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")  # 2k male (m), 2k female (f)

idx_m = y_test=="m"
idx_f = y_test=="f"

### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

pca = PCA(n_components=75)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca = PCA(n_components=75)
X_train_scaler_pca = pca.fit_transform(X_train_scaler)
X_test_scaler_pca = pca.transform(X_test_scaler)

if best_model.params["normalization"] == "scaler":
    X_train = X_train_scaler
    X_test = X_test_scaler
elif best_model.params["normalization"] == "pca":
    X_train = X_train_pca
    X_test = X_test_pca
elif best_model.params["normalization"] == "scalar+pca":
    X_train = X_train_scaler_pca
    X_test = X_test_scaler_pca

model.fit(X_train, y_train)

### evaluation
print("\nACCURACY ON TEST SET")
# accuracy (the test data is balanced)
acc, acc_m, acc_f = calc_accuracy(X_test, y_test, model)
print("Acc: {}".format(acc))
print("Acc M: {}".format(acc_m))
print("Acc F: {}".format(acc_f))

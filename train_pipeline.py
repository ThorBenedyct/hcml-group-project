import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.utils import shuffle
from create_hyperparameter_sets import hyper_parameter_sets


def print_accuracy(x_set, y_set, _model):
    _acc = _model.score(x_set, y_set)
    print("Acc: {}".format(_acc))

    # accuracy per class
    _idx_m = y_set == "m"
    _idx_f = y_set == "f"

    _acc_m = _model.score(x_set[_idx_m], y_set[_idx_m])
    _acc_f = _model.score(x_set[_idx_f], y_set[_idx_f])
    print("Acc M: {}".format(_acc_m))
    print("Acc F: {}".format(_acc_f))
    return _acc, _acc_m, _acc_f


def set_modal(hyper_parameters):
    class_weight = hyper_parameters["class_weight"]
    if hyper_parameters["model"] == "SVC":
        kernel = hyper_parameters["kernel"]
        model = svm.SVC(class_weight=class_weight, kernel=kernel)
    else:
        model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    return model


### load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")  # 6k male (m), 2k female (f)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")  # 2k male (m), 2k female (f)

### Further split training data into train and dev data
sum_females = 0
while sum_females < 250:
    X_train, y_train = shuffle(X_train, y_train)

    X_dev = X_train[:1000]
    y_dev = y_train[:1000]

    idx_m = y_dev == "m"
    idx_f = y_dev == "f"
    sum_males = idx_m.sum()
    sum_females = idx_f.sum()

X_train = X_train[1000:]
y_train = y_train[1000:]

print(f"Split development data with {sum_males} males and {sum_females} females")

### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_train = scaler.fit_transform(X_train)
X_dev = scaler.fit_transform(X_dev)
X_test = scaler.transform(X_test)


highest_accuracy = 0
best_model = None

for hyper_parameters in hyper_parameter_sets:
    model = set_modal(hyper_parameters)

    ### training
    model.fit(X_train, y_train)

    print(f"\nResults for hyper-parameter {hyper_parameters}")
    acc, acc_m, acc_f = print_accuracy(X_dev, y_dev, model)

    if acc > highest_accuracy:
        highest_accuracy = acc
        best_model = hyper_parameters


print(f"\nBest hyper-parameters: {best_model}")
model = set_modal(best_model)
model.fit(X_train, y_train)

### evaluation
print("\nACCURACY ON TEST SET")
# accuracy (the test data is balanced)
print_accuracy(X_test, y_test, model)

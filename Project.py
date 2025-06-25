import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm

"""
Task: Develop a methodology that is based on the train data (face embeddings /
templates) and achieves the best performance (accuracy for gender estimation)
on the test data.

You are allow to modify the training proccess including data augmentation, 
the use of different traditional and deep learning models, regularization
techniques and many more.

The training data is highly unbalanced. You will see that just training a 
simple classifier on this data result in a weak and unfair performance.
Your goal is to increase the performance as much as possible, i.e. you
need to develop a fair and accurate methodology.

Keep in mind: Hyperparameter optimization must be done by splitting the 
training set into an additional evaluation set.
"""

### load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy") # 6k male (m), 2k female (f)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy") # 2k male (m), 2k female (f)

X_dev = X_train[:500]
y_dev = y_train[:500]

idx_m = y_test=="m"
idx_f = y_test=="f"


### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


### define model for gender classification
# model = LogisticRegression(max_iter=1000, class_weight="balanced")
model = svm.SVC(class_weight="balanced", kernel="poly")

### training
model.fit(X_train, y_train)

### evaluation

# accuracy (the test data is balanced)
acc = model.score(X_test, y_test)
print("Acc: {}".format(acc))
 
# accuracy per class
y_pred = model.predict(X_test)

acc_m = model.score(X_test[idx_m], y_test[idx_m])
acc_f = model.score(X_test[idx_f], y_test[idx_f])
print("Acc M: {}".format(acc_m))
print("Acc F: {}".format(acc_f))

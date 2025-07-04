import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from colorama import Fore, Back, Style


### load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy") # 6k male (m), 2k female (f)
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy") # 2k male (m), 2k female (f)

idx_m = y_test=="m"
idx_f = y_test=="f"


### feature normalization
scaler = StandardScaler()
# scaler must be fitted on training data only
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


### define model for gender classification
model = svm.SVC(
    class_weight='balanced',
    kernel="poly",
    gamma=0.001
)
### training
model.fit(X_train, y_train)

### evaluation

# accuracy (the test data is balanced)
acc = model.score(X_test, y_test)
print("Acc: {}".format(acc))
 
# accuracy per class
#y_pred = model.predict(X_test)

acc_m = model.score(X_test[idx_m], y_test[idx_m])
acc_f = model.score(X_test[idx_f], y_test[idx_f])
print(f"{Fore.BLUE}Acc M: {acc_m}")
print(f"{Fore.RED}Acc F: {acc_f}")

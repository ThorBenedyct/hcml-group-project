import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

#  Loading Data 
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

#  Feature normalization 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Shape before reduction:")
print("X_train_pca:", X_train.shape)
print("X_test_pca:", X_test.shape)

#  PCA: Use fixed number of components to cut excessive dimensions
#  Reducing from 512D embeddings to 75D to remove noise and simplify the model
pca = PCA(n_components=75) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#  Define SVM classifier 
svm = SVC(kernel='rbf', class_weight='balanced', gamma='scale')

#  Training SVM model 
svm.fit(X_train_pca, y_train)

#  Saveing the Model 
joblib.dump(svm, "svm_pca_model.pkl")

#  Evaluation 
y_pred = svm.predict(X_test_pca)

#  Overall accuracy 
acc = accuracy_score(y_test, y_pred)

# Accuracy per class
acc_m = accuracy_score(y_test[y_test == 'm'], y_pred[y_test == 'm'])
acc_f = accuracy_score(y_test[y_test == 'f'], y_pred[y_test == 'f'])

print(f"Overall Accuracy: {acc:.4f}")
print(f"Male Accuracy:    {acc_m:.4f}")
print(f"Female Accuracy:  {acc_f:.4f}")

#  Loading the saved model 
loaded_model = joblib.load("svm_pca_model.pkl")

# Accuracy from loaded model
acc_loaded = accuracy_score(y_test, loaded_model.predict(X_test_pca))
print(f"Overall Accuracy (Loaded Model): {acc_loaded:.4f}")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


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


print("Compute TSNE")
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(X_test)

# Masking based on labels
idx_m = y_test=="m"
idx_f = y_test=="f"

c = ["red" if g=="m" else "blue" for g in y_test]

print("Plotting")
# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(X_embedded[:,0], X_embedded[:, 1], color=c, alpha=0.5)
plt.show()

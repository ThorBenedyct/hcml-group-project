import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

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

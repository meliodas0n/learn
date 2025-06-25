"""
Implementation of K Nearest Neigbhors Classification from scratch
"""
import pandas as pd, numpy as np
from collections import Counter

class KNN:
  def __init__(self, k):
    self.k = k

  def fit(self, X, y):
    self.X_train = np.array(X.values[1::, :]) if type(X) == pd.DataFrame else np.array(X)
    self.y_train = np.array(y.values[1::, :]) if type(y) == pd.DataFrame else np.array(y)

  def predict(self, X):
    self.X = np.array(X.values[1::, :]) if type(X) == pd.DataFrame else np.array(X)
    return [self._predict_points(x) for x in X]
  
  def _predict_points(self, x):
    distances = np.linalg.norm(self.X_train - x, axis=1)
    k_indices = distances.argsort()[:self.k]
    k_nearest_labels = self.y_train[k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

if __name__ == "__main__":
  print("Contains the implementation of K Nearest Negitbhors")
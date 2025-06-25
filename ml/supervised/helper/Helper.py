"""
Contains custom/helper code for learning ML
"""
def load_boston():
  import pandas as pd
  import numpy as np

  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  df = pd.read_csv(data_url, sep='\s+', skiprows=22, header=None)
  data, target = np.hstack([df.values[::2, :], df.values[1::2, :2]]), df.values[1::2, 2]
  return df, data, target

if __name__ == "__main__":
  print("Machine Learning: Helper")
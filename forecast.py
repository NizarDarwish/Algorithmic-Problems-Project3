import sys
import math
import matplotlib.pyplot as plt
# import keras
import pandas as pd
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping

if len(sys.argv) != 5:
  print("Give all parameters")
  sys.exit(1)

df=pd.read_csv(sys.argv[2], delimiter='\t', header=None)
print('Number of rows and columns:', df.shape)

training_set = df.iloc[:200, 1:].values
test_set = df.iloc[200:, 1:].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, 200):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(X_train)
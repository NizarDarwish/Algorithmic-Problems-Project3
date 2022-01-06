import sys
import math
import matplotlib.pyplot as plt
# import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
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

n = int(sys.argv[4])

training_set = []
test_set = []

for i in range(0, n):
  training_set.append(df.iloc[i, 1:int(0.8*df.shape[1])].values)
  test_set.append(df.iloc[i, int(0.8*df.shape[1]):].values)
  
# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))


training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time-steps and 1 output
timetables = {}
for timetable in range(0, n):
    X_train = []
    y_train = []
    for i in range(10, int(0.8*df.shape[1] - 1)):
        X_train.append(training_set_scaled[timetable, i-10:i])
        y_train.append(training_set_scaled[timetable, i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    timetables[timetable] = [X_train, y_train]

print(timetables[0][0].shape[1])
print(timetables[1][0].shape[1])

model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) )) #(timetables[time_series][0].shape[1], 1)

model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50))

model.add(Dropout(0.2))# Adding the output layer

model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

for time_series in range(0, n):
# Fitting the RNN to the Training set
  model.fit(timetables[time_series][0],timetables[time_series][1], epochs = 2, batch_size = 50)

full_set = []
for i in range(0, n):
  training_set.extend(test_set)
  inputs = training_set[len(training_set) - len(test_set) - 10:].values
  inputs = inputs.reshape(-1,1)

  # #full_set.append(df.iloc[i, 1:].values)
  # inputs = df.iloc[i, 1:]
  # # print(len(inputs))
  # # print(len(test_set))
  # inputs = inputs[len(inputs) - len(test_set[i]) - 10:].values
  # #inputs = inputs.reshape(-1,1)
  # full_set.append(inputs)

# print(len(full_set))
# print(len(full_set[0]))
# print(len(full_set[1]))
test_set_scaled = sc.transform(inputs)

# Creating a data structure with 60 time-steps and 1 output
timetables_test = {}
for timetable in range(0, n):
    X_test = []
    for i in range(10, int(0.2*df.shape[1] - 1)):
        X_test.append(test_set_scaled[timetable, i-10:i])
    X_test= np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    timetables_test[timetable] = X_test
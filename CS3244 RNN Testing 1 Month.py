import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import datetime
import os
import re
import random

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    return rmse

def plot_predictions(real_real,predicted_stock_price,fname):
    plt.clf()
    plt.plot(real_real, color = 'black', label = 'Real Stock Price')
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig('plot/' + fname[:len(fname)-4] + '.png')

os.chdir("C:/Users/Ong Jing Long/Desktop/CS3244 Machine Learning/Project/RNN/1mo_train_val_data_with_quarters")#need to change

# extracting all the company names
fnames = os.listdir()

training_companies = []
for fname in fnames:
    if (re.search('.csv', fname)):
        training_companies.append(fname)

learning_rate = 0.1
epoch = 5
dropout_rate = 0.5
momentum = 0.5
batch_size = 4096
	
RMSE_list = []

regressorRNN = Sequential()
regressorRNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(30,69), activation='tanh'))
regressorRNN.add(Dropout(dropout_rate))
regressorRNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(30,69), activation='tanh'))
regressorRNN.add(Dropout(dropout_rate))
regressorRNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(30,69), activation='tanh'))
regressorRNN.add(Dropout(dropout_rate))
regressorRNN.add(SimpleRNN(units=50, activation='tanh'))
regressorRNN.add(Dropout(dropout_rate))
regressorRNN.add(Dense(units=1))
regressorRNN.compile(optimizer=SGD(lr = learning_rate, decay = 1e-7, momentum = momentum, nesterov=False),
    loss='mean_squared_error')
                                          
train = []
train_indices=[0]

for comp in training_companies:
    training_set = pd.read_csv(comp, index_col='date', parse_dates=['date'])
    if training_set.shape[0] < 30:
        continue
    train.append(training_set)
    train_indices.append(train_indices[-1] + len(training_set))

final_training_set = pd.concat(train)
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled_x = sc.fit_transform(final_training_set.iloc[: , 1:70])
sc2 = MinMaxScaler(feature_range=(0,1))
training_set_scaled_y = sc2.fit_transform(final_training_set.iloc[: , 0].values.reshape(-1,1))
                        
for indices_loop in range (0, len(train_indices)-1):#loop for every companies, note that number of element is number of companies
    X_train = []
    y_train = []
    for i in range(train_indices[indices_loop] + 30, train_indices[indices_loop + 1], 30):
        X_train.append(training_set_scaled_x[i-30:i, 0:69])
        y_train.append(training_set_scaled_y[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    if X_train.size == 0:
        continue
    regressorRNN.fit(X_train, y_train, epochs = epoch, batch_size = batch_size)
    regressorRNN.reset_states()


os.chdir("C:/Users/Ong Jing Long/Desktop/CS3244 Machine Learning/Project/RNN/1mo_test_data_with_quarters")#need to change

fid= open("newlogs.txt",'w+')
# extracting all the company names
filenames = os.listdir()

testing_companies = []
for filename in filenames:
    if (re.search('.csv', filename)):
        testing_companies.append(filename)

for comp in testing_companies:
    testing_set = pd.read_csv(comp, index_col='date', parse_dates=['date'])
    if testing_set.shape[0] < 30:
        continue
    testing_set_scaled_x = sc.transform(testing_set.iloc[: , 1:70])
    testing_set_y = testing_set.iloc[: , 0].values.reshape(-1,1)
    X_test = []
    y_test = []
    for i in range(30, len(testing_set), 30):
        X_test.append(testing_set_scaled_x[i-30:i, 0:69])
        y_test.append(testing_set_y[i])
    X_test, y_test = np.array(X_test), np.array(y_test)
    if X_test.size == 0:
        continue
    RNN_predicted_stock_price = regressorRNN.predict(X_test)
    RNN_predicted_stock_price = sc2.inverse_transform(RNN_predicted_stock_price)
    fid.write("\nRMSE:%f\r\n" %(return_rmse(y_test,RNN_predicted_stock_price)))
    RMSE_list.append(return_rmse(y_test,RNN_predicted_stock_price))
    plot_predictions(y_test, RNN_predicted_stock_price, comp)
    regressorRNN.reset_states()
    
fid.write("\nRMSE Sum:%f\r\n" %(sum(RMSE_list)))
fid.write("\nAverage RMSE:%f\r\n" %(sum(RMSE_list) / len(RMSE_list)))
fid.close()

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
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

def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Actual Stock Price')
    plt.plot(predicted, color='blue',label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

os.chdir("C:/Users/Ong Jing Long/Downloads/TrainValidate_and_Test/TestTrainValidate/1mo_train_val_data")#need to change

# extracting all the company names
fnames = os.listdir()

companies = []
for fname in fnames:
    if (re.search('.csv', fname)):
        companies.append(fname)

random.shuffle(companies)

size = len(companies)

k_fold = 5

try_learning_rates = [0.02, 0.05, 0.07, 0.1. 0.15]
try_epoch = [1, 5, 10]
try_dropout = [0.2, 0.5, 0.8]
try_momentum = [0.2, 0.5, 0.8]
try_batch_size = [2048, 4096]

best_RMSE_sum = math.inf;
best_learning_rate = 0;
best_epoch = 0;
best_dropout = 0;
best_momentum = 0;
best_batch_size = 0;

#write results into file
fid= open("logs.txt",'w+')
for learning_r in try_learning_rates:
    for epoch_try in try_epoch:
        for dropout_try in try_dropout:
            for momentum_try in try_momentum:
                for batch_size_try in try_batch_size:
                    RMSE_sum = 0
                    index = -(size//k_fold)
                    for i in range(k_fold):
                        
                        regressorRNN = Sequential()
                        regressorRNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(30,69), activation='tanh'))
                        regressorRNN.add(Dropout(dropout_try))
                        regressorRNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(30,69), activation='tanh'))
                        regressorRNN.add(Dropout(dropout_try))
                        regressorRNN.add(SimpleRNN(units=50, return_sequences=True, input_shape=(30,69), activation='tanh'))
                        regressorRNN.add(Dropout(dropout_try))
                        regressorRNN.add(SimpleRNN(units=50, activation='tanh'))
                        regressorRNN.add(Dropout(dropout_try))
                        regressorRNN.add(Dense(units=1))
                        regressorRNN.compile(optimizer=SGD(lr = learning_r, decay = 1e-7, momentum = momentum_try, nesterov=False),
                                             loss='mean_squared_error')
                        
                        index += (size//k_fold)
                        
                        validating = []
                        training = companies.copy()

                        #fill the validating and training
                        for temp_loop in range (index, index + (size//k_fold)):
                            validating.append(companies[temp_loop])
                            training.remove(companies[temp_loop])
                        
                        train = []
                        train_indices=[0]

                        for comp in training:
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
                        #training_set_scaled_y = final_training_set.iloc[: , 0].values.reshape(-1,1)
                        
                        for indices_loop in range (0, len(train_indices)-1):#loop for every companies, note that number of element is number of companies
                            X_train = []
                            y_train = []
                            for i in range(train_indices[indices_loop] + 30, train_indices[indices_loop + 1], 30):
                                X_train.append(training_set_scaled_x[i-30:i, 0:69])
                                y_train.append(training_set_scaled_y[i])
                            X_train, y_train = np.array(X_train), np.array(y_train)
                            if X_train.size == 0:
                                continue
                            regressorRNN.fit(X_train, y_train, epochs = epoch_try, batch_size = batch_size_try)
                            regressorRNN.reset_states()
                        
                        for comp in validating:
                            validating_set = pd.read_csv(comp, index_col='date', parse_dates=['date'])
                            if validating_set.shape[0] < 30:
                                continue
                            validating_set_scaled_x = sc.transform(validating_set.iloc[: , 1:70])
                            validating_set_y = final_training_set.iloc[: , 0].values.reshape(-1,1)
                            X_test = []
                            y_test = []
                            for i in range(30, len(validating_set), 30):
                                X_test.append(validating_set_scaled_x[i-30:i, 0:69])
                                y_test.append(validating_set_y[i])
                            X_test, y_test = np.array(X_test), np.array(y_test)
                            if X_test.size == 0:
                                continue
                            RNN_predicted_stock_price = regressorRNN.predict(X_test)
                            RNN_predicted_stock_price = sc2.inverse_transform(RNN_predicted_stock_price)
                            RMSE_sum += return_rmse(y_test,RNN_predicted_stock_price)
                            regressorRNN.reset_states()
                        
                    if RMSE_sum < best_RMSE_sum:
                        best_RMSE_sum = RMSE_sum
                        best_learning_rate = learning_r
                        best_epoch = epoch_try;
                        best_dropout = dropout_try;
                        best_momentum = momentum_try;
                        best_batch_size = batch_size_try;
                    
                    fid.write("HYPERPARAMETERS:\r\n")
                    fid.write("\tlearning_rate:%f\r\n" %(learning_r))
                    fid.write("\tepochs:%f\r\n" %(epoch_try))
                    fid.write("\tdropout_rate:%f\r\n" %(dropout_try))
                    fid.write("\tmomentum_rate:%f\r\n" %(momentum_try))
                    fid.write("\tbatch_size:%f\r\n" %(batch_size_try))
fid.write("\n########################################\nBEST HYPERPARAMETERS:\r\n")
fid.write("\tlearning_rate:%f\r\n" %(best_learning_rate))
fid.write("\tepochs:%f\r\n" %(best_epoch))
fid.write("\tdropout_rate:%f\r\n" %(best_dropout))
fid.write("\tmomentum_rate:%f\r\n" %(best_momentum))
fid.write("\tbatch_size:%f\r\n" %(best_batch_size))                    
#print("The best learning rate is {}.".format(best_learning_rate))

#write results into file
fid.close()

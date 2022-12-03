from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import time
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import read_csv


# Question 1
def read_close_data():
    close_list_2018 = []
    close_list_2019 = []
    months_2018 = []
    months_2019 = []

    with open('data/DAT_XLSX_EURUSD_M1_2018.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            close_list_2018.append(float((line.split(';')[4]).replace(',', '.')))
            months_2018.append(line.split(';')[0])

    with open('data/DAT_XLSX_EURUSD_M1_2019.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            close_list_2019.append(float((line.split(';')[4]).replace(',', '.')))
            months_2019.append(line.split(';')[0])

    close_array_2018 = np.array(close_list_2018)
    return close_list_2018, close_list_2019, months_2018, months_2019, close_array_2018


# Question 1
def draw_graph():
    close_point_2018, close_point_2019, months_2018, months_2019 = read_close_data()[:4]
    x_axis_2018 = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in months_2018]
    x_axis_2019 = [datetime.strptime(d, '%Y-%m-%d %H:%M').date() for d in months_2019]

    fig, ax = plt.subplots()
    ax.plot(x_axis_2018, close_point_2018, label='2018')
    ax.plot(x_axis_2019, close_point_2019, label='2019')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlabel('Months')
    ax.set_ylabel('Close')
    ax.set_title('Close price evolution during 2018 and 2019')
    ax.legend()
    plt.show()


# Question 2
def native_model():
    right_predictions: int = 0
    wrong_predictions: int = 0
    difference_list: list = []
    compared_values: list = []
    data = (read_close_data()[0] + read_close_data()[1])[:-2]

    for i in range(0, len(data) - 1, 5):
        difference_list.append(data[i + 4] - data[i])
        compared_values.append(data[i + 4])

    for i in range(0, len(difference_list) - 1):
        if difference_list[i] >= 0:
            if compared_values[i + 1] >= compared_values[i]:
                right_predictions += 1
            else:
                wrong_predictions += 1
        else:
            if compared_values[i + 1] < compared_values[i]:
                right_predictions += 1
            else:
                wrong_predictions += 1

    prediction_accuracy = (right_predictions / (right_predictions + wrong_predictions)) * 100
    return prediction_accuracy


# Question 3
def cnn_model():
    # create cnn model with keras
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(10, 1)))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model():
    close_point_2018, close_point_2019, months_2018, months_2019 = read_close_data()
    close_point_2018 = close_point_2018[1:]
    close_point_2019 = close_point_2019[1:]
    months_2018 = months_2018[1:]
    months_2019 = months_2019[1:]
    # reshape data to fit the model
    close_point_2018 = close_point_2018[:1000]
    close_point_2018 = close_point_2018.reshape((len(close_point_2018), 1, 1))
    close_point_2019 = close_point_2019[:1000]
    close_point_2019 = close_point_2019.reshape((len(close_point_2019), 1, 1))
    # train the model
    model = cnn_model()
    model.fit(close_point_2018, close_point_2019, epochs=100, verbose=0)
    # save the model
    model.save('model.h5')


def predict():
    close_point_2018, close_point_2019, months_2018, months_2019 = read_close_data()
    close_point_2018 = close_point_2018[1:]
    close_point_2019 = close_point_2019[1:]
    months_2018 = months_2018[1:]
    months_2019 = months_2019[1:]
    # reshape data to fit the model
    close_point_2018 = close_point_2018[:1000]
    close_point_2018 = close_point_2018.reshape((len(close_point_2018), 1, 1))
    close_point_2019 = close_point_2019[:1000]
    close_point_2019 = close_point_2019.reshape((len(close_point_2019), 1, 1))
    # load the model
    model = keras.models.load_model('model.h5')
    # predict the next value
    yhat = model.predict(close_point_2018, verbose=0)
    print(yhat)


def lstm_model():
    # LSTM for international airline passengers problem with regression framing

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # fix random seed for reproducibility
    tf.random.set_seed(7)
    # load the dataset
    # dataframe = read_csv('data/DAT_XLSX_EURUSD_M1_2018.csv', usecols=[5], engine='python')
    # dataset = dataframe.values
    dataset = read_close_data()[4][300000:]
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # testPredictPlot[len(trainPredict) + (look_back):len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), label='real')
    plt.plot(trainPredictPlot, label='train')
    plt.plot(testPredictPlot, label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Question 1
    draw_graph()

    # Question 2
    print(f'Native model accuracy: {native_model():.2f}%')

    # Question 3
    # cnn_model()
    # train_model()
    # predict()

    # Question 4
    # lstm_model()

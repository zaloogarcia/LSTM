from pandas import DataFrame, Series, concat, read_excel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from azureml.core.run import Run
from math import sqrt
import numpy as np
import os, matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')  # For MACOS


def timeseries_to_supervised(df, lag=1):
    df = DataFrame(df)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    # Start training
    for i in range(nb_epoch):
        if (i % 100) == 0:
            print("Epochs is at %d" % i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    model_json = model.to_json()
    with open("model/model.json","w") as json_file:
        json_file.write(model_json)
    model.save_weights("model/model.h5")
    print('Model Saved to disk')
    return model


# Predict from previous data
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# Folders that will be later uploaded
os.makedirs('outputs', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Starts logging the variables and metrics
run = Run.get_context()

# Data pre processing
series = read_excel('train.xlsx')
series.set_index('date', inplace=True)
raw_values = series['sales'].values
diff_values = difference(raw_values, 1)
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
data_length = len(series)
div = int(data_length * 0.3)
train, test = supervised_values[0:-div], supervised_values[-div:]

# Modeling
scaler, train_scaled, test_scaled = scale(train, test)
neurons = 6
batches = 2000
run.log('neurons', neurons), run.log('batches', batches)
lstm_model = fit_lstm(train_scaled, 1, batches, neurons)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# Model validation
predicts = list()
for i in range(len(test_scaled)):
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    print(X)
    yhat = forecast_lstm(lstm_model, 1, X)
    yhat = invert_scale(scaler, X, yhat)
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    run.log('Predicted y', yhat)
    predicts.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print(raw_values[len(train) +i ], expected)
    run.log('True Value', expected)
    print('Day=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# Check the actual error
rmse = sqrt(mean_squared_error(raw_values[-div:], predicts))
run.log('Mean squared error', rmse)
print('Test RMSE: %.3f' % rmse)

# Log the plot
plt.title('LSTM with ' + str(neurons) + 'Neurons')
plt.plot(raw_values[-div:], label='True')
plt.plot(predicts, '-r', label='Prediction')
plt.legend(loc='best')
run.log_image('Prediction', plot=plt)
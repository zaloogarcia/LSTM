import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from azureml.core.model import Model
import os, json


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def inverse_difference(history, yhat):
    return yhat + history


def init():
    global model
    model_path = Model.get_model_path(model_name='model')
    json_file = open(os.path.join(model_path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(os.path.join(model_path, "model.h5"))


def run(raw_data):
    try:
        prediction = []
        scaler = MinMaxScaler(feature_range=(-1, 1))
        raw_data = pd.DataFrame(json.loads(raw_data)['dates'])
        raw_data = pd.to_datetime(raw_data[0])
        days_from_init = pd.to_datetime(pd.datetime(2019, 2, 2))
        data_length = len(raw_data) + abs((days_from_init - raw_data[0]).days)
        last_prediction = [0.17562724]
        last_value = 175.99999

        for i in range(1, data_length):

            npprediction = np.asarray(last_prediction)
            yhat = forecast_lstm(model, 1, npprediction)
            last_prediction = [yhat]
            npprediction = npprediction.reshape(1,-1)
            scaler = scaler.fit(npprediction)
            yhat = invert_scale(scaler, last_prediction, yhat)
            yhat = inverse_difference(last_value, yhat)
            last_value = yhat
            prediction.append(yhat)
        return prediction

    except BaseException as e:
        return str(e.args)


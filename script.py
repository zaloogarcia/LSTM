import keras
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path(model_name='model.json')
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model = model.load_weights('model/model.h5')


def run(raw_data):
    try:
        prediction = []
        raw_data = pd.to_datetime(raw_data['date'])
        days_from_init = pd.datetime(2019,2,2)
        dates_data = raw_data['date']
        i = 0
        for day in dates_data['date']:
            dates_data[i] = abs((day - days_from_init).days)
            date = dates_data[i]
            prediction.append(model.predict(date))
            i += 1
        return prediction

    except Exception as e:
        error = str(e)
        return error
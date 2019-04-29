import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(df, lag=1):
    df = pd.DataFrame(df)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# Group sales per week
data = pd.read_excel('train.xlsx')
data['date'] = pd.to_datetime(data['date']) - pd.to_timedelta(7, unit='d')
data = data.groupby([pd.Grouper(key='date', freq='W-MON')])['sales'].sum().reset_index().sort_values('date')
print(data.head())
# Transform scale
X = data.values
print(X.shape)
X = X.reshape(len(X), 1)
# test, train = X[0:-25], X[-25:]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(data)
scaled_X = scaler.transform(data)
scaled_data = pd.Series(scaled_X[:, 0])
print(scaled_data)

# transform to be stationary
# differenced = difference(X, 1)

# inver transform
# inverted = list()
# for i in range(len(differenced)):
#     value = inverse_difference(X, differenced[i], len(X)-i)
#     inverted.append(value)
# inverted = pd.Series(inverted)
#
# print(inverted)
# plt.plot(data['date'], data['sales'])
# plt.show()

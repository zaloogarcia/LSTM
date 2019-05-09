import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


data = pd.read_excel('train.xlsx')

test_data = data.loc[165:, 'sellings'].values
train_data = data.loc[:165,'sellings'].values

# Normalize the data
scaler = MinMaxScaler()
test_data = test_data.reshape(-1, 1)
train_data = train_data.reshape(-1, 1)

smoothing_window_size = int(len(train_data)/4)
for di in range(0, len(train_data), smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

EMA =0.0
gamma = 0.1

for ti in range(len(train_data)):
    EMA = gamma*train_data[ti]+(1-gamma)*EMA
    train_data[ti] = EMA

all_data = np.concatenate([train_data, test_data], axis=0)

window_size = 10 # days
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pref_idx in range(window_size, N):
    if pref_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date()+dt.timedelta(days=1)
    else:
        date = data.loc[pref_idx, 'date']

    std_avg_predictions.append(np.mean(train_data[pref_idx-window_size:pref_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pref_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

# print(data[0])

print(len(train_data), len(range(165)))
slope, intercept, r_value, p_value, std_error = stats.linregress(range(166), train_data)
print('r2', r_value**2)


# plt.figure(figsize = (18,9))
# plt.plot(range(data.shape[0]+1),all_data,color='b',label='True')
# plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
# plt.xlabel('Date')
# plt.ylabel('sales')
# plt.legend(fontsize=18)
# # plt.show()


window_size = 10
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):

    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

# plt.figure(figsize = (18,9))
# plt.plot(range(data.shape[0]+1),all_data,color='b',label='True')
# plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend(fontsize=18)
# plt.show()

class DataGeneratorSeq(object):

    def __init__(self, sales, batch_size, num_unroll):
        self._sales = sales
        self._sales_length = len(self._sales) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._sales_length // self._batch_size
        self.cursor = [offset * self]

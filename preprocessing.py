import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import models

#car_sensor = pd.read_csv("data/Car_Sensor_Data.csv")
data = pd.read_csv("data/NILU_Dataset_Trondheim_2014-2019.csv")
dataset = data.values
dataset = dataset[1:, :]
d1 = dataset[:, [0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23, 24]]
d2 = dataset[:, [0, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24]]
d3 = dataset[:, [0, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24]]
d4 = dataset[:, [0, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
names = ['Timestamp', 'NO', 'NO2', 'NOx', 'PM10', 'PM2.5', 'humidity', 'pressure', 'rain', 'temperature',  'wind_direction',
         'wind_speed']
df_bakkekirke = pd.DataFrame(d1, columns=names)
df_e6tiller = pd.DataFrame(d2, columns=names)
df_elgseter = pd.DataFrame(d3, columns=names)
df_torvet = pd.DataFrame(d4, columns=['Timestamp', 'NO2', 'PM10', 'PM2.5', 'humidity', 'pressure', 'rain', 'temperature',
                                      'wind_direction', 'wind_speed'])
df_bakkekirke = (df_bakkekirke.dropna()).convert_objects(convert_numeric=True)
df_e6tiller = (df_e6tiller.dropna()).convert_objects(convert_numeric=True)
df_elgseter = (df_elgseter.dropna()).convert_objects(convert_numeric=True)
df_torvet = (df_torvet.dropna()).convert_objects(convert_numeric=True)
"""
plt.subplot(11, 1, 1)
plt.plot(df_bakkekirke['NO'])

plt.subplot(11, 1, 2)
plt.plot(df_bakkekirke['NO2'])

plt.subplot(11, 1, 3)
plt.plot(df_bakkekirke['NOx'])

plt.subplot(11, 1, 4)
plt.plot(df_bakkekirke['PM10'])

plt.subplot(11, 1, 5)
plt.plot(df_bakkekirke['PM2.5'])

plt.subplot(11, 1, 6)
plt.plot(df_bakkekirke['humidity'])

plt.subplot(11, 1, 7)
plt.plot(df_bakkekirke['pressure'])

plt.subplot(11, 1, 8)
plt.plot(df_bakkekirke['rain'])

plt.subplot(11, 1, 9)
plt.plot(df_bakkekirke['temperature'])

plt.subplot(11, 1, 10)
plt.plot(df_bakkekirke['wind_direction'])

plt.subplot(11, 1, 11)
plt.plot(df_bakkekirke['wind_speed'])
plt.show()

g = sns.pairplot(df_bakkekirke)
plt.show()

corr = df_bakkekirke.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
plt.show()
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform((df_bakkekirke.values)[:, 1:])
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[11, 12, 13, 15, 16, 17, 18, 19, 20, 21]], axis=1, inplace=True)
print(reframed.head())


# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
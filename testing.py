"""
first step is to decide on whether I want to make the data (for train/valid/test) conditional
on having persisted for x generations. If I do, it's definitely a less general approach, as one would
need to train the model specifically for (i.e. conditional upon) sequences that have persisted for x (or more) generations.
So ultimately I'd prefer not to have to do this, but in a way it's a simpler problem. The issue with using all trajectories of allele
frequencies is that the vast majority of these will go extinct in a handful of generations (many in the first generation). This means that
my training set will be heavily biased towards samples that 1. have little to no information, and 2. aren't very interesting.
It's hard to imagine being able to train an accurate model for those cases that do persist for a while if they are <1% of the total samples.

For now, just removing simulations that don't survive x generations, but in practice need to alter the C++ program so that I have balanced coverage (right now obviously I have many more examples of the high selection coefficient cases)

It might be cool to train an ensemble from the best model of each type (CNN, LSTM, etc.)
Just a note that perhaps it would be worth creating entirely new datasets for each type of model, as this will increase their independence
(and thus improve the ensemble). Then when training the ensemble, I'd want to use an entirely new dataset again (the joys of being able to generate data at will...)

Couple important notes so far.
1. Tweaks to what data is fed to the model is the dominant effect so far (e.g. I don't think there's any model that will make good predictions if I just feed it the raw junk from every simulation)
2. Balancing representation between parameter values helps a lot (I've only done this in a really crude way so far, but I get a better fit as it gets better balanced (even with a smaller training set so it isn't a size thing)
3. Increasing time series length to 100 (from 50) seems to help quite a bit. Might be partially a CNN thing , as CNNs are effective at long dependencies whereas RNNs tend to have the forgetting problem for very long series, but not really surprising that it helps.

"""


import pandas as pd
import numpy as np
from os import listdir
import re

import tensorflow as tf
from tensorflow import keras
import math

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

def test_train_valid_split(data, train_prop=0.95, valid_prop=0.03, test_prop=0.02):
    dataset_size = len(data.index)
    train_data = data.iloc[:int(train_prop*(dataset_size)), :]
    valid_data = data.iloc[int(train_prop*(dataset_size)):int((train_prop*(dataset_size)) + int(valid_prop*(dataset_size))), :]
    test_data = data.iloc[int(-test_prop*(dataset_size)):, :]
    return train_data, valid_data, test_data

def x_and_y_split(data):
    y = data.iloc[:, 0:3].to_numpy() #.transpose() with Normalizer
    x = data.iloc[:, 3:].to_numpy()
    return x, y

    

def insert_fix_prob(data):
    if data["ps"][0] == 100 and math.isclose(data["sc"][0], 0.0):
        data.insert(0, "fix", 0.0100)
    elif data["ps"][0] == 200 and math.isclose(data["sc"][0], 0.0):
        data.insert(0, "fix", 0.0050)
    elif data["ps"][0] == 100 and math.isclose(data["sc"][0], 0.02):
        data.insert(0, "fix", 0.0402)
    elif data["ps"][0] == 100 and math.isclose(data["sc"][0], 0.04):
        data.insert(0, "fix", 0.0765)
    elif data["ps"][0] == 100 and math.isclose(data["sc"][0], 0.06):
        data.insert(0, "fix", 0.1112)
    elif data["ps"][0] == 100 and math.isclose(data["sc"][0], 0.08):
        data.insert(0, "fix", 0.1438)
    elif data["ps"][0] == 200 and math.isclose(data["sc"][0], 0.02):
        data.insert(0, "fix", 0.0396)
    elif data["ps"][0] == 200 and math.isclose(data["sc"][0], 0.04):
        data.insert(0, "fix", 0.0761)
    elif data["ps"][0] == 200 and math.isclose(data["sc"][0], 0.06):
        data.insert(0, "fix", 0.1118)
    elif data["ps"][0] == 200 and math.isclose(data["sc"][0], 0.08):
        data.insert(0, "fix", 0.1446)
    elif (data["ps"][0] == 100 or data["ps"][0] == 200) and math.isclose(data["sc"][0], 0.1):
        data.insert(0, "fix", 0.176) # not a mistake, the values are the same to 3 decimals
    else:
        print("problem")

data_dir = "/home/joshua/projects/metric/data/raw_allele_data/HSE/"
num_replicates = 1000000
num_files = 4
l = []
# data = pd.concat((pd.read_csv(f) for f in listdir(data_dir)))


for f in listdir(data_dir):
    if math.isclose(float(re.search(r'HSE_[0-9]+_(.*?)_', f).group(1)), 0.0):
        num_replicates=3000000
    data = pd.read_csv(data_dir + f, nrows=num_replicates, header=None, index_col=None)
    data.insert(0,"ps", int(re.search(r'HSE_(.*?)_', f).group(1)))
    data.insert(0,"sc", float(re.search(r'HSE_[0-9]+_(.*?)_', f).group(1)))
    insert_fix_prob(data)
    l.append(data)
data = pd.concat(l, axis=0, ignore_index=True)    
data = data.fillna(value=0.0)

# remove samples that went extinct early
data.drop(data[np.isclose(data.iloc[:, 100], 0.0)].index, inplace=True)


# shuffle data
data = data.sample(frac=1).reset_index(drop=True)
#split into test train valid
train_data, valid_data, test_data = test_train_valid_split(data,)
train_x, train_y = x_and_y_split(train_data)
valid_x, valid_y = x_and_y_split(valid_data)
test_x, test_y = x_and_y_split(test_data)

# need to scale the outputs (probably wouldn't matter so much if I just had one output but the loss function is mse and ps is on a very different scale to sc/fix

# transformer = Normalizer(norm="max")
transformer = StandardScaler()
transformer.fit(train_y) # only fit on train (though wouldn't matter in this specific case)
scaled_train_y = transformer.transform(train_y)
scaled_valid_y = transformer.transform(valid_y)
scaled_test_y = transformer.transform(test_y)


# print(scaled_train_y)

def abs_dist(y_true, y_pred):
    abs_difference = tf.math.abs(y_true[0] - y_pred[0])
    return tf.reduce_mean(abs_difference, axis=-1)


# note that the x features (i.e. allele freqs) are already scaled since they are bounded between 0 and 1
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(300, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dense(300, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dense(200, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dense(200, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dense(50, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dense(3, activation="linear"))

# model = keras.models.Sequential()
# model.add(keras.layers.Dropout(0.1))
# model.add(keras.layers.Dense(500, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(500, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.Dense(500, activation="relu", kernel_initializer="he_uniform"))
# model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(3, activation="linear"))

expand_train_x = np.expand_dims(train_x, axis=2)
expand_valid_x = np.expand_dims(valid_x, axis=2)
expand_test_x = np.expand_dims(test_x, axis=2)

#tsc model
# model = keras.models.Sequential()
# model.add(keras.layers.Input((100,1)))
# model.add(keras.layers.Conv1D(filters=6,kernel_size=7,padding='valid',activation='sigmoid'))
# model.add(keras.layers.AveragePooling1D(pool_size=3))
# model.add(keras.layers.Conv1D(filters=12,kernel_size=7,padding='valid',activation='sigmoid'))
# model.add(keras.layers.AveragePooling1D(pool_size=3))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(3, activation='relu', kernel_initializer='he_normal'))

# model = keras.models.Sequential()
# model.add(keras.layers.Input((100,1)))
# model.add(keras.layers.Conv1D(filters=64,kernel_size=7,padding='same',activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.Conv1D(filters=64,kernel_size=7,padding='same',activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling1D(pool_size=2))
# model.add(keras.layers.Conv1D(filters=128,kernel_size=3,padding='same',activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.Conv1D(filters=128,kernel_size=3,padding='same',activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling1D(pool_size=2))
# model.add(keras.layers.Conv1D(filters=256,kernel_size=3,padding='same',activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.Conv1D(filters=256,kernel_size=3,padding='same',activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.MaxPooling1D(pool_size=2))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'))
# model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(3, activation='linear'))

# model = keras.models.Sequential()
# model.add(keras.layers.Input((100,1)))
# model.add(keras.layers.Conv1D(filters=64,kernel_size=7,padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('elu'))
# model.add(keras.layers.Conv1D(filters=64,kernel_size=7,padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('elu'))
# model.add(keras.layers.MaxPooling1D(pool_size=2))
# model.add(keras.layers.Conv1D(filters=128,kernel_size=3,padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('elu'))
# model.add(keras.layers.Conv1D(filters=128,kernel_size=3,padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('elu'))
# model.add(keras.layers.MaxPooling1D(pool_size=2))
# model.add(keras.layers.Conv1D(filters=256,kernel_size=3,padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('elu'))
# model.add(keras.layers.Conv1D(filters=256,kernel_size=3,padding='same', kernel_initializer='he_normal'))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation('elu'))
# model.add(keras.layers.MaxPooling1D(pool_size=2))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'))
# # model.add(keras.layers.Dropout(0.3))
# model.add(keras.layers.Dense(3, activation='linear'))

model = keras.models.Sequential()
model.add(keras.layers.Input((100,1)))
model.add(keras.layers.LSTM(50, activation='tanh', return_sequences=True))
model.add(keras.layers.LSTM(50, activation='tanh', return_sequences=True))
model.add(keras.layers.LSTM(100, activation='tanh'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(3, activation='linear'))

def scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

optimizer = tf.keras.optimizers.Adam(0.002)
model.compile(loss="mse", optimizer=optimizer, metrics=[abs_dist])
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(expand_train_x, scaled_train_y, epochs=25, validation_data=(expand_valid_x, scaled_valid_y), callbacks=[callback])
# history = model.fit(train_x, scaled_train_y, epochs=20, validation_data=(valid_x, scaled_valid_y))

# y_reg = model.predict(test_x[0:30, :])
y_reg = model.predict(expand_test_x[0:30, :])

print(transformer.inverse_transform(y_reg))
print(transformer.inverse_transform(scaled_test_y[0:30,:]))
# model.evaluate(test_x, scaled_test_y)
model.evaluate(expand_test_x, scaled_test_y)

# random note that I might need to consider some more standard regularisation techniques as well
# though I think I should only add it if it's actually overfitting. Rationale being that even with the
# "causal regularisation", there's still the risk that the statistical mapping to the causal model's parameters
# is itself overfitted. Adding the causal regularisation should fix issues with overfitting wrt fixation prob
# but not necessarily with predicting the parameters associated with the causal regularisation. Will have to
# see how this bears out in reality though



# model.summary()

# train_prop = 0.6
# valid_prop = 0.2
# test_prop = 0.2
# dataset_size = num_replicates * num_files

# train_data = data.iloc[:int(train_prop*(dataset_size)), :]
# valid_data = data.iloc[int(train_prop*(dataset_size)):int((train_prop*(dataset_size)) + int(valid_prop*(dataset_size))), :]
# test_data = data.iloc[int(-test_prop*(dataset_size)):, :]

# print(data)
# print(train_data)
# print(valid_data)
# print(test_data)


# print(data.iloc[0,2:50])

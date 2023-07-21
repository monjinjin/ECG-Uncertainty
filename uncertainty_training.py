# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 01:33:44 2022

@author: jinli
"""
#import h5py
#print(h5py.__version__)

import wfdb  # waveform database
import pandas as pd
import numpy as np
import os
from os import mkdir
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.preprocessing import LabelEncoder
import warnings


# code is brittle, requires Python version 3.8.8 and numpy version 1.23.1
# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
warnings.simplefilter(action='ignore', category=FutureWarning)

# https://github.com/anishapant21/ECG-Arrhythmia/blob/main/CNN.ipynb - source

# weight imbalance / balance the data
# data augmentation
# precision and recall *
# sensitivity specificity *
# bootstrap / bagging
# pytorch & convolution, padding
# loading


dir_in = r"C:\Users\jinli\Downloads\Monica\Python Projects\Duke Research\mitbih.dataset"
dir_out = r"C:\Users\jinli\Downloads\Monica\Python Projects\Duke Research\mitbih.dataset\.atr"

# valid and invalid beats


realbeats = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
beatscount = [0] * 19

N = ['N', 'L', 'R', 'e', 'j']
S = ['A', 'a', 'J', 'S']
V = ['V', 'E']
F = ['F']
Q = ['/', 'f', 'Q']


classes = ['N', 'S', 'V', 'F', 'Q']

if not os.path.exists(dir_out):
    mkdir(dir_out)

# preparing data


# records = [f for f in listdir(dir_in) if os.path.isfile(os.path.join(dir_in, f)) if (f.find('.dat') != -1)]
records = np.loadtxt(os.path.join(dir_in, "RECORDS"), dtype=str)
print(records)
print(len(records))


def read_data():
    data_set = []
    for record in records:
        temp = wfdb.rdrecord(dir_in + '/' + record)
        data_set.append(temp.p_signal)
    return data_set


dataset = read_data()


# beat segmentation


def classify(symbol):
    if symbol in realbeats:
        return 1
    else:
        return 0


def segment(signal_MLII, beat_loc):
    window = 180
    x = beat_loc - window
    y = beat_loc + window
    sample = signal_MLII[x:y]
    return sample


all_signals = []
all_labels = []
for r in records:
    temp = wfdb.rdrecord(dir_in + '/' + r)
    annotation = wfdb.rdann(dir_in + '/' + r, 'atr')
    symbol = annotation.symbol
    location = annotation.sample
    #print(annotation.__dict__)
    loc_index = [i for i, x in enumerate(symbol) if x in realbeats]
    loc = [location[i] for i in loc_index]
    label_i = []
    signal = temp.p_signal
    signal_MLII = signal[:, 0]

    for i, x in enumerate(location):
        label_dec = classify(symbol[i])
        segmentation = segment(signal_MLII, x)
        if label_dec == 1 and len(segmentation) == 360:
            all_signals.append(segmentation)
            all_labels.append(symbol[i])

    for i, x in enumerate(symbol):
        if x in realbeats:
            beatscount[realbeats.index(x)] = beatscount[realbeats.index(x)] + 1

# stack arrays in sequence vertically


all_beats = np.vstack(all_signals)
#labels = pd.Series(realbeats)
print(all_beats.shape)
labels = pd.Series(all_labels)


# check distribution


for i in range(len(realbeats)):
    print(realbeats[i] + ': ', beatscount[i])


# resampling


print(len(all_beats))
# labels_array = np.array(realbeats)
labels_array = np.array(all_labels)

'''
for beats in all_beats:
    if beats in N:
        labels = 'N'
    elif beats in S:
        labels = 'S'
    elif beats in V:
        labels = 'V'
    elif beats in F:
        labels = 'F'
    elif beats in Q:
        labels = 'Q'
'''
for beats in all_beats:
    if beats in N:
        labels[beats] = 'N'
    elif beats in S:
        labels[beats] = 'S'
    elif beats in V:
        labels[beats] = 'V'
    elif beats in F:
        labels[beats] = 'F'
    elif beats in Q:
        labels[beats] = 'Q'

df0 = all_beats[labels == 'N']
df1 = all_beats[labels == 'S']
df2 = all_beats[labels == 'V']
df3 = all_beats[labels == 'F']
df4 = all_beats[labels == 'Q']
#print(len(df0))

df0_sampled = resample(df0, replace=True, n_samples=20000, random_state=42)
df1_sampled = resample(df1, replace=True, n_samples=20000, random_state=42)
df2_sampled = resample(df2, replace=True, n_samples=20000, random_state=42)
df3_sampled = resample(df3, replace=True, n_samples=20000, random_state=42)
df4_sampled = resample(df4, replace=True, n_samples=20000, random_state=42)

y0 = ['N'] * len(df0_sampled)
y1 = ['S'] * len(df1_sampled)
y2 = ['V'] * len(df2_sampled)
y3 = ['F'] * len(df3_sampled)
y4 = ['Q'] * len(df3_sampled)

X_final = []
X_final.extend(df0_sampled)
X_final.extend(df1_sampled)
X_final.extend(df2_sampled)
X_final.extend(df3_sampled)
X_final.extend(df4_sampled)

y_final = []
y_final.extend(y0)
y_final.extend(y1)
y_final.extend(y2)
y_final.extend(y3)
y_final.extend(y4)

# standardization of data


scalar = StandardScaler()
scaled = scalar.fit_transform(X_final)


def check(y):
    temp = pd.DataFrame(y, columns=["Labels"])
    print("Value distribution:\n")
    count = temp["Labels"].value_counts()
    percent = temp["Labels"].value_counts(normalize=True).mul(100).round(2)
    print(pd.concat([count, percent], axis=1, keys=["Counts", "Percent"]))


# splitting data into train & test sets


strad = StratifiedShuffleSplit()
assin_strad = strad.split(scaled, y_final)

train_index, test_index = next(assin_strad)
train_data_scaled = scaled[train_index]

lab = LabelEncoder()
labels_final = lab.fit_transform(y_final)

train_label = labels_final[train_index]
assin_val = strad.split(train_data_scaled, train_label)

train_index_fin, val_index = next(assin_val)
x_val = train_data_scaled[val_index]
train_data_scaled_fin = train_data_scaled[train_index_fin]
train_labels_fin = train_label[train_index_fin]

y_val = train_label[val_index]
train_data_scaled = scaled[test_index]
test_labels = labels_final[test_index]

check(test_labels)
check(train_label)
check(y_val)

# convolutional neural network


MODEL_PATH = os.path.join(dir_out, "saved_models")
# os.makedirs("saved_models", exist=True)
os.makedirs("saved_models", exist_ok=True)

import pickle


# a function to save trained models in pickle object
def save_model(name, model, extension=".pickle"):
    path = os.path.join(MODEL_PATH, name + extension)
    print("Saving Model : ", name)
    file = open(path, "wb")
    pickle.dump(model, file)
    file.close()


from tensorflow import keras

CNN_X_train = train_data_scaled_fin.reshape(len(train_data_scaled_fin), len(train_data_scaled_fin[0]), 1)
CNN_val = x_val.reshape(len(x_val), len(x_val[0]), 1)

CNN_model = keras.Sequential()
CNN_model.add(keras.layers.Conv1D(64, kernel_size=3, input_shape=(360, 1), activation="relu"))
CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
CNN_model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
CNN_model.add(keras.layers.MaxPool1D(pool_size=2))
CNN_model.add(keras.layers.Flatten())
CNN_model.add(keras.layers.Dense(100, activation="relu"))
CNN_model.add(keras.layers.Dense(7, activation="softmax"))
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
opt = keras.optimizers.Adam(lr=0.0001)
CNN_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
CNN_model.summary()

#history = CNN_model.fit(CNN_X_train, train_labels_fin, epochs=25, validation_data=(CNN_val, y_val))
history = CNN_model.fit(CNN_X_train, train_labels_fin, epochs=20, validation_data=(CNN_val, y_val))
# max_val_acc=max(history.history['accuracy'])

# return CNN_model.evaluate(X_test, y_test)[1]

# cnn_prediction = CNN_model.predict_classes(CNN_X_train)
cnn_prediction = np.argmax(CNN_model.predict(CNN_X_train), axis=-1)
# CNN_X_train[1]
print(cnn_prediction[0:10])
print(cnn_prediction)

'''
print(y_val.shape)
print(test_labels.shape)
print(train_data_scaled.shape)
print(test_index.shape)
print(cnn_prediction.shape)
print(CNN_X_train.shape)
print(train_data_scaled_fin.shape)
print(train_labels_fin.shape)
print(x_val.shape)
'''

precision = precision_score(train_labels_fin, cnn_prediction, average = "weighted")
recall = recall_score(train_labels_fin, cnn_prediction, average = "weighted")
print("precision:", precision)
print("recall:", recall)



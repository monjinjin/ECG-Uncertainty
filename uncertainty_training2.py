# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 16:54:44 2022

@author: jinli
"""

#google code lab

import wfdb  # waveform database
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as Fc
import pandas as pd
import numpy as np
import os
from os import mkdir
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
import re
# code is brittle, requires Python version 3.8.8 and numpy version 1.23.1
# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
warnings.simplefilter(action='ignore', category=FutureWarning)

# https://github.com/anishapant21/ECG-Arrhythmia/blob/main/CNN.ipynb - source

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

with open(r'C:/Users/jinli/Downloads/Monica/Python Projects/Duke Research/signals.txt', 'w') as fp:
    for item in dataset:
        fp.write('%s\n' % item)

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
labels_array = np.array(all_labels)

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
#print(labels)

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

#y0 = ['N'] * len(df0_sampled)
y0 = [0]*len(df0_sampled)
#y1 = ['S'] * len(df1_sampled)
y1 = [1]*len(df1_sampled)
#y2 = ['V'] * len(df2_sampled)
y2 = [2]*len(df2_sampled)
#y3 = ['F'] * len(df3_sampled)
y3 = [3]*len(df3_sampled)
#y4 = ['Q'] * len(df3_sampled)
y4 = [4]*len(df3_sampled)

X_final = []
X_final.extend(df0_sampled)
X_final.extend(df1_sampled)
X_final.extend(df2_sampled)
X_final.extend(df3_sampled)
X_final.extend(df4_sampled)
#print(df0_sampled.size)

y_final = []
y_final.extend(y0)
y_final.extend(y1)
y_final.extend(y2)
y_final.extend(y3)
y_final.extend(y4)
#print(len(y0))

# standardization of data


scalar = StandardScaler()
scaled = scalar.fit_transform(X_final)


def check(y):
    temp = pd.DataFrame(y, columns=["Labels"])
    print("Value distribution:\n")
    count = temp["Labels"].value_counts()
    percent = temp["Labels"].value_counts(normalize=True).mul(100).round(2)
    print(pd.concat([count, percent], axis=1, keys=["Counts", "Percent"]))



# pytorch

#transform = transforms.Compose(
    #[transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# change elements in list from numpy to tensor using for loop


# split into train and test 

#print(len(X_final))
#print(len(y_final))

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X_final, y_final, test_size = 0.2)

X_train = torch.FloatTensor(xtrain)
X_test = torch.FloatTensor(xtest)
Y_train = torch.FloatTensor(ytrain)
Y_test = torch.FloatTensor(ytest)

#trainset = torchvision.datasets.CIFAR10(root=dir_in, train=True, download=True, transform=transform)

from torch.utils.data import TensorDataset
# data tensor

trainset = TensorDataset(X_train, Y_train)
print(X_train)
print(Y_train)
#X - beats, Y - labels
#transpose
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

#testset = torchvision.datasets.CIFAR10(root=dir_in , train=False, download=True, transform=transform)
testset = TensorDataset(X_test, Y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# convolutional neural network

# CHANGE HERE
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding = 1) # needs input channel, output channel, kernel size, stride, padding (changes output shape)
        #reshape the input tensor to batch size to 1 as channel number and 360 as signal length
        #(1,64,3)
        #add padding to conv1d - kernel 3, padding 1; kernel 5, padding 2
        self.pool = nn.MaxPool1d(2) 
        self.conv2 = nn.Conv1d(64, 32, 3, padding = 1)
        #(64,32,3) -> signal length = 90
        self.fc1 = nn.Linear(90*32, 100) #input channel, output channel
        #weight_dimension = (90*32, 100) 
        self.fc2 = nn.Linear(100, 7)
        #weight_dimension(100, 7)
        
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(Fc.relu(self.conv1(x)))
        x = self.pool(Fc.relu(self.conv2(x)))
        #(batch_size, channel_number, signal_length) ->
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #(batch_size, (signal_length*channel_number))
        x = Fc.relu(self.fc1(x))
        x = Fc.relu(self.fc2(x))
        return x


net = Net()


# loss function & optimizer

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
# -> change to keras
#SGD change to Adam

# training


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #print(inputs.shape)
        inputs = torch.reshape(inputs, (4, 1, 360))
        outputs = net(inputs)
        #outputs = outputs.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# an issue here
PATH = r'C:/Users/jinli/Downloads/Monica/Python Projects/Duke Research/signals.txt'
torch.save(net.state_dict(), PATH)


# testing


net = Net()
net.load_state_dict(torch.load(PATH))
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(2)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        beats, labels = data
        # calculate outputs by running beats through the network
        beats = torch.reshape(beats, (4, 1, 360))
        outputs = net(beats)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test beats: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


predictionlist = []
labellist = []


# again no gradients needed
with torch.no_grad():
    for data in testloader:
        beats, labels = data
        beats = torch.reshape(beats, (4, 1, 360))
        outputs = net(beats)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            label = label.type(torch.IntTensor)
            prediction = prediction.type(torch.IntTensor)
            predictionlist.append(prediction)
            labellist.append(label)
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
            


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    

#print(len(predictionlist))
#print(len(labellist))

cm = confusion_matrix(labellist, predictionlist)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP

print("precision: ", TP/(TP+FP))
print("recall: ", TP/(TP+FN))
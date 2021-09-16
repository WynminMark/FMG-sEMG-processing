# -*- coding: gbk -*-
import numpy as np
import csv
import sys
import scipy.io as sio
sys.path.append('..')
import os
from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from keras.optimizers import SGD
import math
import matplotlib.pyplot as plt

num_epoch = 150

def data_load(a):
    ra = 'ra'
    rd = 'rd'
    sa = 'sa'
    sd = 'sd'
    sit = 'sit'
    stand = 'stand'
    walk = 'walk'

    data_ra = sio.loadmat(a+ra)[ra]
    data_rd = sio.loadmat(a+rd)[rd]
    data_sa = sio.loadmat(a+sa)[sa]
    data_sd = sio.loadmat(a+sd)[sd]
    data_sit = sio.loadmat(a+sit)[sit]
    data_stand = sio.loadmat(a+stand)[stand]
    data_walk = sio.loadmat(a+walk)[walk]
    return data_ra, data_rd, data_sa, data_sd, data_sit, data_stand, data_walk

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
sgd = SGD(lr=0.0, momentum=0.99, decay=0.0, nesterov=False)


subject = ['chx_', 'hoy_', 'kl_', 'mm_', 'wyx_', 'yhl_', 'yjl_', 'yyj_', 'zjs_', 'zkj_', 'zs_']
train_subject = ['chx_', 'hoy_', 'kl_', 'mm_', 'wyx_', 'yhl_', 'yjl_', 'yyj_', 'zjs_', 'zkj_']
data_ra = []
data_rd = []
data_sa = []
data_sd = []
data_sit = []
data_stand = []
data_walk = []

for i in train_subject:
    data = data_load(i)

    data_ra.extend(data[0])
    data_rd.extend(data[1])
    data_sa.extend(data[2])
    data_sd.extend(data[3])
    data_sit.extend(data[4])
    data_stand.extend(data[5])
    data_walk.extend(data[6])
    
data_ra = np.array(data_ra)
data_rd = np.array(data_rd)
data_sa = np.array(data_sa)
data_sd = np.array(data_sd)
data_sit = np.array(data_sit)
data_stand = np.array(data_stand)
data_walk = np.array(data_walk)

train_ra = data_ra[0:len(data_ra)//50*50, 0:5].reshape(-1, 50, 5)
train_rd = data_rd[0:len(data_rd)//50*50, 0:5].reshape(-1, 50, 5)
train_sa = data_sa[0:len(data_sa)//50*50, 0:5].reshape(-1, 50, 5)
train_sd = data_sd[0:len(data_sd)//50*50, 0:5].reshape(-1, 50, 5)
train_sit = data_sit[0:len(data_sit)//50*50, 0:5].reshape(-1, 50, 5)
train_stand = data_stand[0:len(data_stand)//50*50, 0:5].reshape(-1, 50, 5)
train_walk = data_walk[0:len(data_walk)//50*50, 0:5].reshape(-1, 50, 5)

data_test = data_load('zs_')
test_ra = data_test[0][0:len(data_test[0])//50*50, 0:5].reshape(-1, 50, 5)
test_rd = data_test[1][0:len(data_test[1])//50*50, 0:5].reshape(-1, 50, 5)
test_sa = data_test[2][0:len(data_test[2])//50*50, 0:5].reshape(-1, 50, 5)
test_sd = data_test[3][0:len(data_test[3])//50*50, 0:5].reshape(-1, 50, 5)
test_sit = data_test[4][0:len(data_test[4])//50*50, 0:5].reshape(-1, 50, 5)
test_stand = data_test[5][0:len(data_test[5])//50*50, 0:5].reshape(-1, 50, 5)
test_walk = data_test[6][0:len(data_test[6])//50*50, 0:5].reshape(-1, 50, 5)

train = []
train_label = []
test = []
test_label = []

sit_train_l = np.zeros((len(train_sit), 7))
stand_train_l = np.zeros((len(train_stand), 7))
walk_train_l = np.zeros((len(train_walk), 7))
stairA_train_l = np.zeros((len(train_sa), 7))
stairD_train_l = np.zeros((len(train_sd), 7))
rampA_train_l = np.zeros((len(train_ra), 7))
rampD_train_l = np.zeros((len(train_rd), 7))

sit_test_l = np.zeros((len(test_sit), 7))
stand_test_l = np.zeros((len(test_stand), 7))
walk_test_l = np.zeros((len(test_walk), 7))
stairA_test_l = np.zeros((len(test_sa), 7))
stairD_test_l = np.zeros((len(test_sd), 7))
rampA_test_l = np.zeros((len(test_ra), 7))
rampD_test_l = np.zeros((len(test_rd), 7))

for i in range(len(sit_train_l)):
    sit_train_l[i, 0] = 1

for i in range(len(stand_train_l)):
    stand_train_l[i, 1] = 1

for i in range(len(walk_train_l)):
    walk_train_l[i, 2] = 1

for i in range(len(stairA_train_l)):
    stairA_train_l[i, 3] = 1

for i in range(len(stairD_train_l)):
    stairD_train_l[i, 4] = 1

for i in range(len(rampA_train_l)):
    rampA_train_l[i, 5] = 1

for i in range(len(rampD_train_l)):
    rampD_train_l[i, 6] = 1


for i in range(len(sit_test_l)):
    sit_test_l[i, 0] = 1

for i in range(len(stand_test_l)):
    stand_test_l[i, 1] = 1

for i in range(len(walk_test_l)):
    walk_test_l[i, 2] = 1

for i in range(len(stairA_test_l)):
    stairA_test_l[i, 3] = 1

for i in range(len(stairD_test_l)):
    stairD_test_l[i, 4] = 1

for i in range(len(rampA_test_l)):
    rampA_test_l[i, 5] = 1

for i in range(len(rampD_test_l)):
    rampD_test_l[i, 6] = 1

train = np.concatenate((train_sit, train_stand, train_walk, train_sa, train_sd, train_ra, train_rd), axis = 0)
train_label = np.concatenate((sit_train_l, stand_train_l, walk_train_l, stairA_train_l, stairD_train_l, rampA_train_l, rampD_train_l), axis = 0)
test = np.concatenate((test_sit, test_stand, test_walk, test_sa, test_sd, test_ra, test_rd), axis = 0)
test_label = np.concatenate((sit_test_l, stand_test_l, walk_test_l, stairA_test_l, stairD_test_l, rampA_test_l, rampD_test_l), axis = 0)


print(np.shape(train))
print(np.shape(train_label))
print(np.shape(test))
print(np.shape(test_label))

print('feature is ready')

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, Masking, Permute, Input, Conv1D,BatchNormalization
from keras.layers import GlobalAveragePooling1D, CuDNNLSTM, concatenate, Activation, GRU, SimpleRNN
import keras
#ES：monitor-监测的值，patience-没有进步的运行轮数（运行十轮后仍无提升，则停止）
#MC：在每个epoch之后保存模型至filepath（save_best_only:只保存最好模型）
callbacks_list = [
  keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,),
  e.g.keras.callbacks.ModelCheckpoint(filepath='mfcn.h5', monitor='val_loss', save_best_only=True)
]

NUM_CELLS = 8
sequenceLength = 50
ip = Input(shape=(sequenceLength,9))
#ip = Masking(mask_value= 0)(ip)

x = LSTM(NUM_CELLS)(ip)
x = Dropout(0.8)(x)

y = Permute((2, 1))(ip)#置换输入维度
y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)

y = Activation('relu')(y)
y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)

y = Activation('relu')(y)
y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)


y = Activation('relu')(y)
y = GlobalAveragePooling1D()(y)

x = concatenate([x, y])
out = Dense(1, activation='softmax')(x)
model = Model(ip, out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(fe_tra,la_tra, validation_split=0.25, epochs=30, batch_size=10, verbose=1)
model.save('class_model.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

preds = model.evaluate(fe_te,la_te,batch_size = 10)
print(preds)




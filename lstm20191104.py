import numpy as np
import scipy.io as sio
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten


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

def main():
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

    # round numbers and reshape to N 50*5 data segment
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


    model = Sequential()
    model.add(LSTM(32, dropout = 0.3, recurrent_dropout = 0.15, input_shape = (50, 5), activation = 'tanh'))
    model.add(Dense(7, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    model.fit(train, train_label, epochs = 200, batch_size = 10, shuffle = True)
    score = model.evaluate(test, test_label, batch_size = 15)

    model.save('m1106.h5')
    print(score)

    return
    
main()






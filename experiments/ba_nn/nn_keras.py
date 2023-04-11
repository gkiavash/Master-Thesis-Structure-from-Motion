import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Layer

from keras import backend as K
import tensorflow as tf

from utils import get_initial_matches_and_matrices


class MLPBlock(Dense):
    def __init__(self, activation=None, **kwargs):
        super(MLPBlock, self).__init__(units=2, activation=activation, **kwargs)

        self.linear_1 = Dense(2, input_dim=2, kernel_initializer='normal', activation='linear')
        self.linear_2 = Dense(10, activation='linear')

    def call(self, inputs):
        x = self.linear_1(inputs)
        x_output = self.linear_2(x)
        # print(tf.print(x_output))
        return x_output
        # print(x_output.get_shape())
        # zz = tf.constant(x_output)
        #
        # R = rotation_matrix(*x_output[:2])
        # t = x_output[3:6]
        # t_x = np.array([
        #     [0, -t[2], t[1]],
        #     [t[2], 0, -t[0]],
        #     [-t[1], t[0], 0]
        # ])
        # fx = x_output[6]
        # fy = x_output[7]
        # cx = x_output[8]
        # cy = x_output[9]
        # K_camera = np.array([
        #     [fx, 0, cx],
        #     [0, fy, cy],
        #     [0, 0, 1]
        # ])
        # E = t_x @ R
        # F = np.transpose(np.linalg.inv(K_camera)) @ E @ np.linalg.inv(K_camera)
        # print(F)
        #
        # return x_output


def run():
    K, R, t, F, E, pts_x, pts_y = get_initial_matches_and_matrices()

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_x.fit(pts_x)
    scaler_y.fit(pts_y)

    xscale=scaler_x.transform(pts_x)
    yscale=scaler_y.transform(pts_y)
    print("scale", xscale.shape, yscale.shape)

    X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
    print("split", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    def model_1():
        model = Sequential()
        model.add(Dense(10, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, activation='relu'))  # Rx, Ry, Rz, Tx, Ty, Tz, fx, fy, cx, cy  (10)
        model.add(Dense(y_train.shape[1], activation='linear'))  # right image's x and y
        return model

    def model_2():
        model = Sequential()
        model.add(MLPBlock())
        model.add(Dense(2, activation='linear'))
        return model

    model = model_2()
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=64,  verbose=1, validation_split=0.2)
    model.summary()
    print(history.history.keys())

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

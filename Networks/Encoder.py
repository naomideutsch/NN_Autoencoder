from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose

from tensorflow.python.keras.activations import sigmoid, relu

import tensorflow as tf


class Encoder(NN):
    def __init__(self):
        super(Encoder, self).__init__()

        # encoder
        self.conv1 = Conv2D(32, 3, padding="same", activation='relu', strides=2)
        self.conv2 = Conv2D(64, 3, padding="same", activation='relu', strides=2)
        self.flatten = Flatten()
        self.fully_connected1 = Dense(512, activation='relu')
        self.fully_connected2 = Dense(10)


        print("Encoder network created")



    def call(self, x, sigmoid_activation=True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        if sigmoid_activation:
            return sigmoid(x)
        return relu(x)

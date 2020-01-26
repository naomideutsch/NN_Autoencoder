from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose

from tensorflow.python.keras.activations import sigmoid, relu

import tensorflow as tf
class Decoder(NN):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fully_connected3 = Dense(512, activation='relu')
        self.fully_connected4 = Dense(7*7*64, activation='relu')
        self.reshape = Reshape((7, 7, 64))
        self.conv_transpose1 = Conv2DTranspose(32, 3, padding="same", activation='relu', strides=2)
        self.conv_transpose2 = Conv2DTranspose(1, 3, padding="same", activation='sigmoid',
                                               strides=2)

        print("Decoder network created")



    def call(self, x):
        x = self.fully_connected3(x)
        x = self.fully_connected4(x)
        x = self.reshape(x)
        x = self.conv_transpose1(x)
        x = self.conv_transpose2(x)
        return x

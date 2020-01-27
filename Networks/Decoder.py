from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose, Embedding, InputLayer, BatchNormalization, ReLU, LeakyReLU

from tensorflow.python.keras.activations import sigmoid, relu

import tensorflow as tf
class Decoder(NN):
    def __init__(self, raanan_architecture=False):
        super(Decoder, self).__init__()

        self.input_layer = InputLayer()


        self.fully_connected3 = Dense(512)
        self.fully_connected4 = Dense(7*7*64)
        self.reshape = Reshape((7, 7, 64))
        self.conv_transpose1 = Conv2DTranspose(32, 3, padding="same", strides=2)
        self.conv_transpose2 = Conv2DTranspose(1, 3, padding="same", activation='sigmoid',
                                               strides=2)

        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()

        self.bn1 = None
        self.bn2 = None
        self.bn3 = None

        if raanan_architecture:
            # self.fully_connected3 = Dense(1024)
            # self.fully_connected4 = Dense(7 * 7 * 128)
            # self.reshape = Reshape((7, 7, 128))
            # self.conv_transpose1 = Conv2DTranspose(64, 4, padding="same", strides=2)
            # self.conv_transpose2 = Conv2DTranspose(1, 4, padding="same", activation='tanh',
            #                                        strides=2)
            # self.bn1 = BatchNormalization()
            # self.bn2 = BatchNormalization()
            # self.bn3 = BatchNormalization()

            self.relu1 = LeakyReLU()
            self.relu2 = LeakyReLU()
            self.relu3 = LeakyReLU()

            self.fully_connected3 = Dense(512)
            self.fully_connected4 = Dense(7 * 7 * 64)
            self.reshape = Reshape((7, 7, 64))
            self.conv_transpose1 = Conv2DTranspose(32, 3, padding="same", strides=2)
            self.conv_transpose2 = Conv2DTranspose(1, 3, padding="same", activation='sigmoid',
                                                   strides=2)

        print("Decoder network created with raanan architecture={}".format(raanan_architecture))



    def call(self, x):
        x = self.input_layer(x)


        x = self.fully_connected3(x)
        if self.bn1 != None:
            x = self.bn1(x)
        x = self.relu1(x)

        x = self.fully_connected4(x)
        if self.bn2 != None:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.reshape(x)
        x = self.conv_transpose1(x)
        if self.bn3 != None:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv_transpose2(x)
        return x

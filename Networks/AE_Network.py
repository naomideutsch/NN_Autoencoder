from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose


class AE_Network(NN):
    def __init__(self):
        super(AE_Network, self).__init__()

        # encoder
        self.conv1 = Conv2D(32, 2, activation='relu', strides=2)
        self.conv2 = Conv2D(64, 2, activation='relu', strides=2)
        self.flatten = Flatten()
        self.fully_connected1 = Dense(512, activation='relu')
        self.fully_connected2 = Dense(10, activation='relu')

        # decoder
        self.fully_connected3 = Dense(512, activation='relu')
        self.fully_connected4 = Dense(7*7*64, activation='relu')
        self.reshape = Reshape((7, 7, 64))
        self.conv_transpose1 = Conv2DTranspose(32, 2, activation='relu', strides=2)
        self.conv_transpose2 = Conv2DTranspose(1, 2, activation='sigmoid', strides=2)

        print("AE_Network network created")




    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.flatten(x)

        x = self.fully_connected1(x)

        x = self.fully_connected2(x)

        return x

    def decode(self, x):
        x = self.fully_connected3(x)

        x = self.fully_connected4(x)

        x = self.reshape(x)

        x = self.conv_transpose1(x)

        x = self.conv_transpose2(x)

        return x



    def call(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

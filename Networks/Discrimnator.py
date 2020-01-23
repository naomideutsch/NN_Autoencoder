from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose, BatchNormalization, LeakyReLU, InputLayer

from tensorflow.python.keras.activations import sigmoid, relu
import tensorflow as tf

class Discrimnator(NN):
    def __init__(self):
        super(Discrimnator, self).__init__()

        self.input_layer = InputLayer(dtype=tf.float32)

        self.conv1 = Conv2D(64, 4, padding="same", strides=2, dtype=tf.float32)
        self.lrelu1 = LeakyReLU(dtype=tf.float32)
        self.conv2 = Conv2D(128, 4, padding="same", strides=2, dtype=tf.float32)
        self.bn1 = BatchNormalization(dtype=tf.float32)
        self.lrelu2 = LeakyReLU(dtype=tf.float32)
        self.flatten = Flatten(dtype=tf.float32)
        self.fully_connected1 = Dense(1024, dtype=tf.float32)
        self.bn2 = BatchNormalization(dtype=tf.float32)
        self.lrelu3 = LeakyReLU(dtype=tf.float32)

        self.fully_connected2 = Dense(1, dtype=tf.float32)

        print("Discrimnator network created")


    def call(self, x):
        x = self.input_layer(x)
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu2(x)

        x = self.flatten(x)

        x = self.fully_connected1(x)
        x = self.bn2(x)
        x = self.lrelu3(x)

        x = self.fully_connected2(x)
        return x


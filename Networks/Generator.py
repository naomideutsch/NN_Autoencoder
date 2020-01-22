from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose, BatchNormalization
from tensorflow.python.keras.activations import sigmoid, relu, tanh
import tensorflow as tf

from tensorflow.keras.backend import set_floatx

set_floatx("float32")


class Generator(NN):
    def __init__(self, dtype=tf.float32):

        super(Generator, self).__init__()

        self.fully_connected3 = Dense(1024, activation='relu', dtype=dtype)
        self.bn1 = BatchNormalization(dtype=dtype)
        self.fully_connected4 = Dense(7*7*128, activation='relu', dtype=dtype)
        self.bn2 = BatchNormalization(dtype=dtype)

        self.reshape = Reshape((7, 7, 128), dtype=dtype)
        self.conv_transpose1 = Conv2DTranspose(64, 4, padding="same", activation='relu', strides=2, dtype=dtype)
        self.bn3 = BatchNormalization(dtype=dtype)

        self.conv_transpose2 = Conv2DTranspose(1, 4, padding="same", strides=2)

        print("Generator network created")

        self.sigmoid_activation = False

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.fully_connected3(x)

        x = self.bn1(x)

        x = self.fully_connected4(x)

        x = self.bn2(x)

        x = self.reshape(x)

        x = self.conv_transpose1(x)

        x = self.bn3(x)

        x = self.conv_transpose2(x)

        return tanh(x)

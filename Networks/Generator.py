from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Reshape, \
    Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, InputLayer
from tensorflow.python.keras.activations import sigmoid, relu, tanh
import tensorflow as tf





class Generator(NN):
    def __init__(self):

        super(Generator, self).__init__()


        self.input_layer = InputLayer(dtype=tf.float32)
        self.fully_connected1 = Dense(1024, dtype=tf.float32)
        self.bn1 = BatchNormalization(dtype=tf.float32)
        self.relu1 = ReLU(dtype=tf.float32)
        self.fully_connected2 = Dense(7*7*128, dtype=tf.float32)
        self.bn2 = BatchNormalization(dtype=tf.float32)
        self.relu2 = ReLU(dtype=tf.float32)
        self.reshape = Reshape((7, 7, 128), dtype=tf.float32)
        self.conv_transpose1 = Conv2DTranspose(64, 4, padding="same", strides=2, dtype=tf.float32)
        self.bn3 = BatchNormalization(dtype=tf.float32)
        self.relu3 = ReLU(dtype=tf.float32)
        self.conv_transpose2 = Conv2DTranspose(1, 4, padding="same", strides=2, activation='tanh', dtype=tf.float32)
        # self.relu4 = ReLU(dtype=tf.float32)

        # self.input_layer = InputLayer(dtype=tf.float32)
        # self.fully_connected1 = Dense(7 * 7 * 256, use_bias=False)
        # self.bn1 = BatchNormalization()
        # self.relu1 = LeakyReLU()
        #
        # self.reshape = Reshape((7, 7, 256))
        #
        # self.conv_transpose1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        # self.bn3 = BatchNormalization()
        # self.relu3 = LeakyReLU()
        #
        # self.conv_transpose2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        # self.bn4 = BatchNormalization()
        # self.relu4 = layers.LeakyReLU()
        #
        # model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        # assert model.output_shape == (None, 28, 28, 1)


        print("Generator network created")


    def call(self, x):
        x = self.input_layer(x)
        x = self.fully_connected1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.fully_connected2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.reshape(x)

        x = self.conv_transpose1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv_transpose2(x)
        # x = self.relu4(x)
        return x

from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN

class Gan(NN):
    def __init__(self, generator, discriminator):
        super(Gan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        print("Gan network created")


    def set_model_trainable_status(self, model, status):
        for layer in model.layers:
            layer.trainable = status


    def call(self, x):

        # self.generator.sigmoid_activation = False
        # self.set_model_trainable_status(self.discriminator, False)

        x = self.generator(x)
        # x = self.discriminator(x)
        #
        # self.set_model_trainable_status(self.discriminator, True)
        # self.generator.sigmoid_activation = True
        return x

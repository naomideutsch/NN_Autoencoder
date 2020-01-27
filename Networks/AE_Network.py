from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NN import NN
from Networks.Decoder import Decoder
from Networks.Encoder import Encoder

class AE_Network(NN):
    def __init__(self):
        super(AE_Network, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        print("AE_Network network created")


    def call(self, x):
        x = self.encoder(x, False)
        x = self.decoder(x)
        return x

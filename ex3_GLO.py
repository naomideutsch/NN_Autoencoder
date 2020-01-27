from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.python.keras.losses import MSE, BinaryCrossentropy

from utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import argparse
import tensorflow as tf
import numpy as np
import os
import itertools

from Networks.Decoder import Decoder

from train_test import GloTrainer

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--opt_z_iters', '-ziters', type=int, default=10, help='Number of iteration to optimize z before the model optimization')

    parser.add_argument('--epochs', '-ep', type=int, default=1, help='number of epochs')
    parser.add_argument('--latent_vec_size', '-z', type=int, default=10, choices=[64, 128], help='The size of z of '
                                                                               'the generator')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--model_learning_rate', '-mlr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--z_learning_rate', '-zlr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--sigmoid_norm', action="store_true" ,help='use sigmoid last activation')


    parser.add_argument('--plot_freq', '-pf', type=int, default=500,
                        help='iteration check point to the plot')
    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')
    return parser.parse_args()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Creates Training Objects By Names ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_network(network_type):
    import Networks
    package = Networks
    return get_object(network_type, package)


def get_optimizer(optimizer_type, lr=1e-4):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam(lr)
    return None


def get_loss(loss_type):
    loss = None
    if loss_type == "MSE":
        loss = MSE
    return loss


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data Loaders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_dataset(normalization_factor):
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)

    train_images = normalize_real_image(train_images, normalization_factor).astype(np.float32)

    return train_images

def normalize_real_image(real_data, normalize_with_sigmoid=True):
    """
    Normalize images according to the last activation of the network (if sigmoid the values will be between [0, 1]
    and if tanh [-1,1]
    :param normalize_with_sigmoid: Determines the type of the last activation
    """
    if normalize_with_sigmoid:
        return (real_data / 255.0)
    else:
        return (real_data / 127.5 - 1.).astype(np.float32)

def denormalize_generate_image(fake_data, normalize_with_sigmoid=True):
    """
    Denormalize images according to the last activation of the network (if sigmoid the values will be between [0, 1]
    and if tanh [-1,1]
    :param normalize_with_sigmoid: Determines the type of the last activation
    """
    if normalize_with_sigmoid:
        return fake_data * 255.0  # Denormalization
    else:
        return tf.clip_by_value((fake_data + 1) * 127.5, 0, 255)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def generate_and_save_images(model, seed, output_path, title):
    """
    Insert every input in the seed into the model inorder
    to generate image, all the images will be displayed in the same image.
    assumes that there are 16 inputs in the seed.
    """

    predictions = model(tf.Variable(seed, trainable=False))

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(denormalize_generate_image(predictions[i, :, :, 0]), cmap='gray')
      plt.axis('off')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plt.savefig(os.path.join(output_path, '{}.png'.format(title)))
    plt.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def train_main(args, real_ds, plot_freq, output_path, model):
    """
    The train procedure.
    """
    model_optimizer = get_optimizer(args.optimizer, args.model_learning_rate)
    z_space_optimizer = get_optimizer(args.optimizer, args.z_learning_rate)

    loss = get_loss("MSE")
    trainer = GloTrainer(model, model_optimizer, z_space_optimizer, loss, real_ds.shape[0], args.latent_vec_size)
    plotter = Plotter(['model loss', 'z space Loss'], "GLO", os.path.join(output_path, "Loss"))
    z_space_vecs = np.random.normal(size=(real_ds.shape[0], args.latent_vec_size), scale=np.sqrt(1.0/args.latent_vec_size))
    indices = np.arange(real_ds.shape[0])

    train_counter = 0
    model_step = trainer.get_model_step()
    relevant_z_vecs = None

    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        for i in range(int(real_ds.shape[0]/args.batches)):
            start = i * args.batches
            relvant_indices = indices[start: start + args.batches]
            if relevant_z_vecs is None:
                relevant_z_vecs = tf.Variable(z_space_vecs[relvant_indices], trainable=True)
            else:
                relevant_z_vecs.assign(z_space_vecs[relvant_indices])
            relevant_images = real_ds[relvant_indices]
            model_step(relevant_images, relevant_z_vecs)
            normalize_result = relevant_z_vecs.numpy() / np.maximum(np.linalg.norm(relevant_z_vecs.numpy(), axis=0, keepdims=True), 1)
            z_space_vecs[relvant_indices] = normalize_result


            if train_counter % plot_freq == 0:

                template = 'Epochs {}, model Loss: {}, z space Loss: {}'
                print(template.format(epoch + 1,
                                      trainer.model_loss_mean.result(), trainer.z_space_loss_mean.result()))

                plotter.add("model loss", train_counter,
                                          tf.cast(trainer.model_loss_mean.result(), tf.float32).numpy())
                plotter.add("z space Loss", train_counter,
                                          tf.cast(trainer.z_space_loss_mean.result(), tf.float32).numpy())

            train_counter += 1

        trainer.model_loss_mean.reset_states()
        trainer.z_space_loss_mean.reset_states()
        generate_and_save_images(model, z_space_vecs[:16],
                                 args.output_path, "glo_epoch_{}_output".format(epoch)) # create output for every epoch
    plotter.plot()
    return z_space_vecs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

if __name__ == '__main__':
    """
    GLO part
    """
    args = get_args()
    tf.keras.backend.set_floatx('float32')

    train_ds = get_dataset(args.sigmoid_norm)
    model = Decoder(True, args.sigmoid_norm)
    z_space_vecs = train_main(args, train_ds, args.plot_freq,
               args.output_path, model)
    cov = np.cov(z_space_vecs.T)
    mean = np.mean(z_space_vecs, axis=0)
    seed = np.random.multivariate_normal(size=(16), mean=mean, cov=cov)

    generate_and_save_images(model, seed , args.output_path, "glo_output")









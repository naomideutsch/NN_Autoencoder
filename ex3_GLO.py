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
    parser.add_argument('--latent_vec_size', '-z', type=int, default=128, choices=[64, 128], help='The size of z of '
                                                                               'the generator')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--model_learning_rate', '-mlr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--z_learning_rate', '-zlr', type=float, default=0.01, help='learning rate')



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
def get_dataset(batch_size, latent_vec_size):
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)

    train_images = (train_images / 127.5 - 1.).astype(np.float32)  # Normalization


    train_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(x_train.shape[0]).batch(batch_size)
    return train_images, y_train, x_train.shape[0]

def denormalize_generate_image(fake_data):
    return tf.clip_by_value((fake_data + 1) * 127.5, 0, 255)  # Denormalization

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




def generate_sample(model, latent_vec_size, output_dir):
    seed = tf.random.normal([16, args.latent_vec_size])

    z_space_vec = np.random.normal(size=(1, latent_vec_size),
                                                scale=np.sqrt(1.0/latent_vec_size))
    output = model(tf.Variable(z_space_vec, trainable=False))

    plt.figure()
    plt.imshow(denormalize_generate_image(output[0, :, :, 0]), cmap='gray')

    title = "GLO_output"

    plt.title(title)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, title + ".png"))

def generate_and_save_images(model, latent_vec_size, output_path):
    seed = tf.random.normal([16, latent_vec_size])

    # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = denormalize_generate_image(model(tf.Variable(seed, trainable=False)))

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    plt.savefig(os.path.join(output_path, 'GLO_output.png'))
    plt.show()


def visualize_latent(latent_vecs, label, title, output_path, max_examples, embed_tech):
    categorical_plotter = CategoricalPlotter(np.unique(label), title, output_path)

    if embed_tech == "lda":
        lda = LinearDiscriminantAnalysis(n_components=2)
        result = lda.fit_transform(latent_vecs, label[:min(max_examples, label.shape[0])])
    else:
        tsne = TSNE(n_components=2)
        result = tsne.fit_transform(latent_vecs)

    for i in range(result.shape[0]):
        categorical_plotter.add(label[i], result[i, 0], result[i, 1])

    categorical_plotter.plot()
    print("visulaization of z_space is done")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def train_main(args, real_ds, ds_size, plot_freq, output_path, model):

    model_optimizer = get_optimizer(args.optimizer, args.model_learning_rate)
    z_space_optimizer = get_optimizer(args.optimizer, args.z_learning_rate)

    loss = get_loss("MSE")
    trainer = GloTrainer(model, model_optimizer, z_space_optimizer, loss, ds_size, args.latent_vec_size)

    plotter = Plotter(['model loss', 'z space Loss'], "GLO", os.path.join(output_path, "Loss"))

    z_space_vecs = np.random.normal(size=(ds_size, args.latent_vec_size))

    indices = np.arange(real_ds.shape[0])


    # try:
    train_counter = 0
    model_step = trainer.get_model_step()
    # z_space_step = trainer.get_z_space_step()

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

            # z_space_step(relevant_images, relevant_z_vecs)

            model_step(relevant_images, relevant_z_vecs)

            normalize_result = relevant_z_vecs.numpy() / np.linalg.norm(relevant_z_vecs.numpy(), axis=0, keepdims=True)

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


        batch_idx = 0

    # Reset the metrics for the next epoch
    plotter.plot()

    # except Exception as e:
    #     raise (e)
    # finally:
    #     print("train is done")
    #     return z_space_vecs


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


if __name__ == '__main__':
    args = get_args()
    tf.keras.backend.set_floatx('float32')

    train_ds, label, dataset_size = get_dataset(args.batches, args.latent_vec_size)
    model = Decoder(True)


    z_space_vecs = train_main(args, train_ds, dataset_size, args.plot_freq,
               args.output_path, model)
    generate_and_save_images(model, args.latent_vec_size, args.output_path)
    # generate_sample(model, args.latent_vec_size , args.output_path)
    # visualize_latent(z_space_vecs, label, "z_space_with_tsne", args.output_path, 1000, "tsne")










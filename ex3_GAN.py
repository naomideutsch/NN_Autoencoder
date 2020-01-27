from __future__ import absolute_import, division, print_function, unicode_literals
from utils import *

import argparse
import tensorflow as tf
import numpy as np
import os


from Networks.Generator import Generator
from Networks.Discrimnator import Discrimnator


from train_test import GanTrainer

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=1, help='number of epochs')
    parser.add_argument('--latent_vec_size', '-z', type=int, default=100, help='The size of z of '
                                                                              'the generator')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
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


def get_optimizer(optimizer_type):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam(1e-4)
    return None


def get_loss(loss_type):
    loss = None
    if loss_type == "cross_entropy":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data Loaders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_dataset(batch_size, *args):
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(x_train.shape[0]).batch(batch_size)
    return train_ds

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def generate_zspace_interpolation(generator, latent_vec_size, output_path, interplate_images):
    """
    Generate the interpulation of the z space asked in the Targil.
    @:param interplate_images: The number of interpolate images in the output
    """
    fake_vec1 = np.random.normal(0, 1, (1, latent_vec_size))
    fake_vec2 = np.random.normal(0, 1, (1, latent_vec_size))

    outputs = []
    titles = []

    alphas = np.linspace(0, 1, num=interplate_images)
    for a in alphas:
        outputs.append(generator(a*fake_vec1 + (1-a)*fake_vec2, training=False))
        titles.append("a={}".format(a))

    generate_image_from_list(outputs, titles, "z_space_interpolation", output_path)



def generate_image_from_list(images, images_titles, title, output_path):
    """
    Generate an image from a list of images
    """

    fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
    for i in range(len(images)):
        axs[i].imshow(images[i][0, :, :, 0] * 127.5 + 127.5, cmap='gray')
        axs[i].set_title(images_titles[i])
    fig.suptitle(title)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, title + ".png"))
    plt.close()



def generate_and_save_images(model, epoch, test_input, output_path):

  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  if not os.path.exists(output_path):
    os.mkdir(output_path)

  plt.savefig(os.path.join(output_path, 'image_at_epoch_{}.png'.format(epoch)))
  plt.close()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def train_main(args, train_ds, plot_freq, output_path, generator,
               discriminator):
    """
    The train procedure.
    """

    gen_optimizer = get_optimizer(args.optimizer)
    gen_loss = get_loss("cross_entropy")
    disc_optimizer = get_optimizer(args.optimizer)
    disc_loss = get_loss("cross_entropy")
    trainer = GanTrainer(generator, discriminator, gen_optimizer, disc_optimizer, gen_loss, disc_loss)

    generator_plotter = Plotter(['train'], "generator", os.path.join(output_path, "Loss"))
    discriminator_plotter = Plotter(['train'], "discriminator", os.path.join(output_path, "Loss"))

    seed = tf.random.normal([16, args.latent_vec_size])

    try:
        train_counter = 0
        train_step = trainer.get_step()


        for epoch in range(args.epochs):
            for real_images in train_ds:
                fake_vecs = np.random.normal(0, 1, (args.batches, args.latent_vec_size))

                train_step(real_images, fake_vecs)

                if train_counter % plot_freq == 0:

                    template = 'Epochs {}, Discriminator Loss: {}, Generator Loss: {}'
                    print(template.format(epoch + 1,
                                          trainer.disc_loss_mean.result(),
                                          trainer.gen_loss_mean.result()))

                    discriminator_plotter.add("train", train_counter,
                                     tf.cast(trainer.disc_loss_mean.result(), tf.float32).numpy())
                    generator_plotter.add("train", train_counter,
                                     tf.cast(trainer.gen_loss_mean.result(), tf.float32).numpy())

                train_counter += 1

            trainer.disc_loss_mean.reset_states()
            trainer.gen_loss_mean.reset_states()

            generate_and_save_images(generator, epoch, seed, args.output_path) # create output for every epoch

        # Reset the metrics for the next epoch
        discriminator_plotter.plot()
        generator_plotter.plot()

    finally:
        print("train is done")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float32')

    args = get_args()


    train_ds = get_dataset(args.batches)

    generator = Generator()
    discriminator = Discrimnator()

    train_main(args, train_ds, args.plot_freq,
               args.output_path, generator,discriminator)

    generate_zspace_interpolation(generator, args.latent_vec_size, args.output_path, 7)










from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.python.keras.losses import MSE
from utils import get_object, Plotter, CategoricalPlotter, get_num_dataset, BinaryCrossEntropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import argparse
import tensorflow as tf
import numpy as np
import os
import itertools

from train_test import Trainer, Validator

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nntype', default="AE_Network", help='The type of the network')
    parser.add_argument('--dstype', default="num", help='The type of the dataset')

    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=20, help='number of epochs')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--ts', type=int, default=None, help='train size')

    parser.add_argument('--loss', default="cross_entropy", help='the loss function type')

    parser.add_argument('--plot_freq', '-pf', type=int, default=1875,
                        help='iteration check point to the plot')

    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')

    return parser.parse_args()


def get_network(network_type):
    import Networks
    package = Networks
    return get_object(network_type, package)


def get_optimizer(optimizer_type):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam()
    return None


def get_loss(loss_type, sump_num):
    if loss_type == "cross_entropy":
        return BinaryCrossEntropy(sump_num)

    if loss_type == "mse":
        return MSE

    return None


def get_dataset(batches_num, train_size=None, val_size=None, dataset_name="num"):
    (x_train, y_train), (x_test, y_test) = get_num_dataset()

    if train_size is not None:
        idx = np.random.choice(x_train.shape[0], train_size)
        x_train = x_train[idx]
        y_train = y_train[idx]

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Add a channels dimension
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batches_num)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batches_num)
    return train_ds, test_ds


def train_main(epochs, train_ds, test_ds, trainer, validator, plot_freq, network_type, output_path):

    loss_plotter = Plotter(['train'], network_type, os.path.join(output_path, "Loss"))
    try:
        train_counter = 0
        train_step = trainer.get_step()
        test_step = validator.get_step()
        for epoch in range(epochs):
            for images, labels in train_ds:
                train_step(images, labels)
                train_counter += 1
                if train_counter % plot_freq == 0:
                    for test_images, test_labels in test_ds:
                        test_step(test_images, test_labels)
                    template = 'Epoch {}, Loss: {}, Test Loss: {}'
                    print(template.format(epoch + 1,
                                          trainer.train_loss.result(),
                                          validator.test_loss.result()))

                    loss_plotter.add("train", train_counter,
                                     tf.cast(trainer.train_loss.result(), tf.float64).numpy())

                    validator.test_loss.reset_states()

            # Reset the metrics for the next epoch
            trainer.train_loss.reset_states()

    finally:
        loss_plotter.plot()


def visualize_latent(ae, data_to_visualize, labels, title, output_path):
    categoricalPlotter = CategoricalPlotter(np.unique(labels), title, output_path)

    lda = LinearDiscriminantAnalysis(n_components=2)
    results_x = []
    results_y = []
    for i in range(len(data_to_visualize)):
        latent_vec = ae.encode(data_to_visualize[i])

        prediction = lda.fit(latent_vec, labels[i])
        results_x.append(prediction[0])
        results_y.append(prediction[1])
        categoricalPlotter.add(labels[i], prediction[0], prediction[1])

    categoricalPlotter.plot()





if __name__ == '__main__':
    args = get_args()
    tf.keras.backend.set_floatx('float64')
    batches = args.batches
    epochs = args.epochs
    optimizer = get_optimizer(args.optimizer)
    loss = get_loss(args.loss, args.batches)
    train_ds, test_ds = get_dataset(batches, train_size=args.ts, dataset_name=args.dstype)

    network = get_network(args.nntype)

    trainer = Trainer(network, optimizer, loss)
    validator = Validator(network, loss)

    train_main(epochs, train_ds, test_ds, trainer, validator, args.plot_freq, args.nntype,
               args.output_path)
    network.summary()

    (x_train, y_train), (x_test, y_test) = get_num_dataset()
    x_train = x_train[..., tf.newaxis, tf.newaxis]
    x_test = x_test[..., tf.newaxis, tf.newaxis]

    visualize_latent(network, x_train, y_train, "title", args.output_path)
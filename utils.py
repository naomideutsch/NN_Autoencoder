from Networks.NN import NN
import pkgutil
import inspect
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.python.keras.losses import Loss, SparseCategoricalCrossentropy, MSE, KLDivergence
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

from tensorflow.python import keras

from tensorflow.python.keras.layers import Flatten



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~     Losses      ~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class BinaryCrossEntropy(Loss):

    def __init__(self, n):
        super(BinaryCrossEntropy, self).__init__()
        self.n = n

    def call(self, x, pred):
        cross_entropy = -1. * x * tf.math.log(pred) - (1. - x) * tf.math.log(1. - pred)
        loss = tf.reduce_mean(cross_entropy)
        return loss

def kl_divergence(p, p_hat):
    # tf.print(p_hat)
    # return 1
    return p * tf.math.log(p) - p * tf.math.log(p_hat) + (1 - p) * tf.math.log(1 - p) - (1 - p) * tf.math.log(1 - p_hat)

def add_density_regularization(loss, alpha, b):

    def foo(dest, pred, latent_vec):

        return tf.cast(loss(dest, pred), dtype=tf.dtypes.float32) + \
               alpha * kl_divergence(0.1,
                                     tf.reduce_mean(tf.math.abs(tf.cast(latent_vec,
                                                                        dtype=tf.dtypes.float32))))

    return foo


    # def foo(dest, pred, latent_vec, tape):
    #     tf.print(dest[:,:,:,0].shape)
    #     tf.print(latent_vec.shape)
    #
    #     g = tape.gradient(latent_vec, dest[:,:,:,0])
    #     tf.print(g)
    #     # return loss(dest, pred) + keras.regularizers.l2(alpha)(g)
    #     return loss(dest, pred)
    #
    # return foo



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~     Plotting      ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class Plotter:
    def __init__(self, plots_names, title, output_path):
        self.x = dict()
        self.y = dict()
        self.title = title
        self.output_path = output_path
        for name in plots_names:
            self.x[name] = []
            self.y[name] = []

    def add(self, name, x, y):
        self.x[name].append(x)
        self.y[name].append(y)

    def plot(self):
        plt.figure()
        for name in self.x.keys():
            plt.plot(self.x[name], self.y[name], label=name)
        plt.legend()
        plt.xlabel("itarations")
        plt.ylabel("value")
        plt.title(self.title)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        plt.savefig(os.path.join(self.output_path, self.title + ".png"))


class CategoricalPlotter:
    def __init__(self, categories_names, title, output_path):
        self.categories_names = categories_names
        self.output_path = output_path
        self.title = title
        self.x = dict()
        self.y = dict()
        for i in range(len(self.categories_names)):
            self.x[i] = []
            self.y[i] = []

    def add(self, label_idx, x, y):
        self.x[label_idx].append(x)
        self.y[label_idx].append(y)


    def plot(self):
        plt.figure()

        colors = cm.rainbow(np.linspace(0, 1, len(self.categories_names)))
        for i in range(len(self.categories_names)):
            plt.scatter(self.x[i], self.y[i], label=self.categories_names[i], color=colors[i])
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(self.title)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        plt.savefig(os.path.join(self.output_path, self.title + ".png"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~   File Managing   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_module_classes(module, classes, max_depth=2):
    if max_depth > 0:
        for name, obj in inspect.getmembers(module):
            if inspect.ismodule(obj):
                classes.union(classes, get_module_classes(obj, classes, max_depth-1))

            if inspect.isclass(obj) and issubclass(obj, NN):
                classes.add(obj)
    return classes


def get_object(object_type, package, *args):
    classes = set()
    prefix = package.__name__ + "."
    for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix):
        # print("Found submodule %s (is a package: %s)" % (modname, ispkg))
        module = __import__(modname)
        result = get_module_classes(module, set())
        classes = classes.union(result)

    for class_obj in classes:
        if object_type == class_obj.__name__:
            return class_obj(*args)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~   Datasets loaders   ~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def get_denoising_dataset(x_array, p):

    total_arrays = []

    for array in x_array:
        x_noise_array = []
        for image in array:
            x_noise_array.append(add_noise_to_image(image, p))
        total_arrays.append(np.array(x_noise_array))

    return total_arrays


def get_num_dataset():
    from tensorflow.python import keras
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def add_noise_to_image(image, p):
    flatten_image = np.reshape(image, -1)
    indices = np.random.choice(len(flatten_image), int(np.floor(len(flatten_image)*p)))
    new_values = np.random.choice([0,1], int(np.floor(len(flatten_image)*p)))
    flatten_image[indices] = new_values
    new_image = np.reshape(flatten_image, image.shape)
    return new_image














def preprocess_image(im, crop_size=224):
    im = im.resize([crop_size, crop_size])
    I = np.asarray(im).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    I = np.reshape(I, (1,) + I.shape).astype(np.float32)
    return I

def create_random_image(size=(224, 224, 3)):
    return Image.fromarray(np.uint8(np.clip(np.random.normal(loc=128, scale=128, size=size), a_min=0, a_max=255)))

def add_random_noise(image):
    noise = np.random.normal(image.shape())
    result = image + noise
    return result

def unnormalize_image(image):
    image += [[[104.00698793, 116.66876762, 122.67891434]]]
    return np.uint8( tf.clip_by_value(np.flip(image, 2), clip_value_min=0.0, clip_value_max=255.0))

#
# def clip_0_1(image):
#     return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor, scale_factor=1):
    tensor = tensor*scale_factor
    tensor = unnormalize_image(np.array(tensor, dtype=np.float32)[0])
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def tensor_to_numpy(tensor):
    return np.array(tensor)


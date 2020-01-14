import tensorflow as tf


class Trainer:
    def __init__(self, model, optimizer, loss):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def get_step(self):
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images)
                loss = self.loss(images, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)

        return train_step


class Validator:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def get_step(self):
        @tf.function
        def test_step(images, labels):
            predictions = self.model(images)
            t_loss = self.loss(images, predictions)
            self.test_loss(t_loss)

        return test_step





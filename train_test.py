import tensorflow as tf


class Trainer:
    def __init__(self, model, optimizer, loss, loss_with_latent=False):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss_with_latent = loss_with_latent

    def get_step(self):
        @tf.function
        def train_step(dest, to_predict):
            with tf.GradientTape() as tape:
                predictions = self.model(to_predict)
                if self.loss_with_latent:
                    latent_vec = self.model.encode(to_predict)
                    loss = self.loss(dest, predictions, latent_vec)
                else:
                    loss = self.loss(dest, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.train_loss(loss)

        return train_step


class Validator:
    def __init__(self, model, loss, loss_with_latent=False):
        self.model = model
        self.loss = loss
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.loss_with_latent = loss_with_latent

    def get_step(self):
        @tf.function
        def test_step(dest, to_predict):
            predictions = self.model(to_predict)
            if self.loss_with_latent:
                latent_vec = self.model.encode(to_predict)
                t_loss = self.loss(dest, predictions, latent_vec)
            else:
                t_loss = self.loss(dest, predictions)
            self.test_loss(t_loss)

        return test_step





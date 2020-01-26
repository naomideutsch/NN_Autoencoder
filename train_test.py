import tensorflow as tf
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, loss, loss_with_latent=False):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss_with_latent = loss_with_latent
        self.last_gradient = None


    def get_step(self):
        @tf.function
        def train_step(dest, to_predict):
            with tf.GradientTape(persistent=True) as tape:
                predictions = self.model(to_predict)
                if self.loss_with_latent:
                    latent_vec = self.model.encode(to_predict)
                    loss = self.loss(dest, predictions, latent_vec)

                else:
                    loss = self.loss(tf.cast(dest, predictions.dtype), predictions)

            self.last_gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(self.last_gradient, self.model.trainable_variables))
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



class GanTrainer:
    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer, gen_loss, disc_loss):
        self.generator = generator
        self.discriminator = discriminator

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.gen_loss = gen_loss
        self.disc_loss = disc_loss

        self.gen_loss_mean = tf.keras.metrics.Mean(name='train_loss')
        self.disc_loss_mean = tf.keras.metrics.Mean(name='train_loss')



    def get_step(self):
        @tf.function
        def train_step(images, noise):

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = self.generator(noise, training=True)

                real_output = self.discriminator(images, training=True)
                fake_output = self.discriminator(generated_images, training=True)

                gen_loss = self.gen_loss(tf.ones_like(fake_output), fake_output)
                real_loss = self.disc_loss(tf.ones_like(real_output), real_output)
                fake_loss = self.disc_loss(tf.zeros_like(fake_output), fake_output)
                disc_loss = real_loss + fake_loss

                self.gen_loss_mean(gen_loss)
                self.disc_loss_mean(disc_loss)

            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return train_step


class GloTrainer:
    def __init__(self, decoder, model_optimizer, zspace_optimizer,  loss, ds_size, latent_vec_size):
        self.decoder = decoder
        self.model_optimizer = model_optimizer
        self.zspace_optimizer = zspace_optimizer

        self.loss = loss
        self.model_loss_mean = tf.keras.metrics.Mean(name='train_loss')
        self.z_space_loss_mean = tf.keras.metrics.Mean(name='train_loss')
        self.last_gradients = None

    def get_step(self):
        @tf.function
        def train_step(images, relevant_z_vecs):

            with tf.GradientTape() as dec_tape:
                generated_images = self.decoder(relevant_z_vecs, training=True)
                dec_loss = self.loss(generated_images, images)
                self.model_loss_mean(dec_loss)
            model_gradient = dec_tape.gradient(dec_loss, self.decoder.trainable_variables)
            self.model_optimizer.apply_gradients(zip(model_gradient, self.decoder.trainable_variables))

            with tf.GradientTape() as zspace_tape:
                zspace_tape.watch(relevant_z_vecs)

                generated_images = self.decoder(relevant_z_vecs, training=True)
                z_space_loss = self.loss(generated_images, images)
                self.z_space_loss_mean(z_space_loss)

            zspace_gradients = tf.convert_to_tensor(zspace_tape.gradient(z_space_loss, relevant_z_vecs))
            self.zspace_optimizer.apply_gradients(zip([zspace_gradients], [relevant_z_vecs]))

            return self.last_gradients
        return train_step




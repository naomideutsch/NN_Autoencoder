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
            with tf.GradientTape(persistent=True) as tape:
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
    def __init__(self, generator, generator_optimizer, gen_loss):
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.gen_loss = gen_loss
        self.gen_loss_mean = tf.keras.metrics.Mean(name='train_loss')

    def get_step(self):
        @tf.function
        def train_step(images, noise):

            with tf.GradientTape() as gen_tape:
                generated_images = self.generator(noise, training=True)
                gen_loss = self.gen_loss(generated_images, images)
                self.gen_loss_mean(gen_loss)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return train_step




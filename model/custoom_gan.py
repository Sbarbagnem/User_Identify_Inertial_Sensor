import tensorflow as tf
import numpy as np

from model.CGAN.cgan import Generator, Discriminator


class CGAN():
    # TODO
    def __init__(self, latent_dim, cond_dim):
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

    # define generator
    def new_discriminator(self):
        self.discriminator = Discriminator()

    # define discriminator
    def new_generator(self):
        self.generator = Generator(self.latent_dim, self.cond_dim)

    # define function to sample Z from latent space
    def sample_z(self):
        pass

    # define function to sample conditional informations
    def sample_c(self):
        pass

    # define loss and optimizer object
    def loss_opt_metric(self):
        pass

    # define train step
    @ tf.function
    def train_cgan_step(self):
        pass

    # define train loop
    def train_cgan(self):
        pass

    # define function to plot syntethic data

    '''
    @ tf.function
    def train_cgan_step(self, batch):
        batch = tf.cast(batch, dtype=tf.float32)
        batch_size = batch.shape[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim, 1), dtype=tf.float64)
        generated_signal = self.generator(random_latent_vectors)
        combined_signal = tf.concat([generated_signal, batch], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_signal)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim, 1), dtype=tf.float64)

        # add real label to all fake image to get performance discriminator and update generator weights
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return d_loss, g_loss, generated_signal

    def train_gan(self):
        for epoch in range(1, self.epochs_gan + 1):

            for batch, _, _ in self.train_data:
                # Train the discriminator & generator on one batch of real images.
                d_loss, g_loss, _ = self.train_cgan_step(batch)
                self.mean_loss_discriminator.update_state(values=d_loss)
                self.mean_loss_generator.update_state(values=g_loss)

            print("discriminator loss at epoch %d: %.2f" %
                  (epoch, self.mean_loss_discriminator.result().numpy()))
            print("generator loss at epoch %d: %.2f" %
                  (epoch, self.mean_loss_generator.result().numpy()))

            self.mean_loss_discriminator.reset_states()
            self.mean_loss_generator.reset_states()

    '''

    '''
    def loss_opt_metric_gan(self):
        # Instantiate one optimizer for the discriminator and another for the generator.
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)

        # Instantiate a loss function.
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mean_loss_generator = tf.keras.metrics.Mean()
        self.mean_loss_discriminator = tf.keras.metrics.Mean()
    '''

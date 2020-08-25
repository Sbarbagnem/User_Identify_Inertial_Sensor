import tensorflow as tf
import numpy as np
from model.resNet182D.resnet18_2D import make_basic_block_layer


class Generator(tf.keras.Model):

    def __init__(self, latent_dim, cond_dim, number_axis=4, seq_len=100):
        super(Generator, self).__init__()

        self.number_axis = number_axis
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.input_layer = tf.keras.layers.Input(shape=(100, 1))

        lstm_layers = []

        for _ in range(3):
            lstm_layers.append(tf.keras.layers.LSTMCell(
                units=200))

        self.lstm = tf.keras.layers.RNN(
            cell=lstm_layers, return_sequences=False)

        #self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.fc = tf.keras.layers.Dense(
            self.seq_len*self.number_axis, activation='tanh')

        self.out = self.call(self.input_layer)

        super(Generator, self).__init__(
            inputs=self.input_layer,
            outputs=self.out)

    def call(self, inputs, training=None):
        x = self.lstm(inputs, training=training)
        #x = self.avg_pool(x)
        x = self.fc(x, training=training)
        x = tf.reshape(x, [-1, self.seq_len, self.number_axis])

        return x


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_layer = tf.keras.layers.Input(shape=(100, 4))
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(5, 1),
                                            strides=(1, 1),
                                            padding="valid",
                                            kernel_regularizer=tf.keras.regularizers.l2)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 1),
                                               strides=(2, 1),
                                               padding="valid")
        self.res_block_1 = make_basic_block_layer(filter_num=32,
                                             blocks=2,
                                             name='residual_block_1',
                                             kernel=(3, 3))
        self.res_block_2 = make_basic_block_layer(filter_num=64,
                                             blocks=2,
                                             name='residual_block_2',
                                             stride=1,
                                             kernel=(3, 3))
        self.avgpool_2d = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

        self.out = self.call(self.input_layer)

        super(Discriminator, self).__init__(
            inputs=self.input_layer,
            outputs=self.out)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.res_block_1(x, training=training)
        x = self.res_block_2(x, training=training)
        x = self.avgpool_2d(x)
        x = self.fc(x, training=training)

        return x

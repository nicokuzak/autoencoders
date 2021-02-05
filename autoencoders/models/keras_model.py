import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class KerasModel(Model):
    def __init__(self, input_dim, filter_sizes):
        super(KerasModel, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(filter_sizes[0], activation="tanh"),
            layers.Dense(filter_sizes[1], activation="tanh"),
            layers.Dense(filter_sizes[2], activation="tanh"),
            layers.Dense(filter_sizes[3], activation="tanh")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(filter_sizes[2], activation="tanh"),
            layers.Dense(filter_sizes[1], activation="tanh"),
            layers.Dense(filter_sizes[0], activation="tanh"),
            layers.Dense(input_dim) #what is this size?
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# autoencoder.compile(optimizer='adam', loss='mae')

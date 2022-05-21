from tabnanny import verbose
import flwr as fl
import tensorflow as tf
import json
import numpy as np
import tensorflowjs as tfjs
from tensorflow import keras


# Load and compile Keras model
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

#model.save("./model_1")

#model = keras.models.load_model('./model_1')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        w = model.get_weights()
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val), steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        w = model.get_weights()
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client

c = CifarClient()

fl.client.start_numpy_client(server_address="[::]:8081", client=c)





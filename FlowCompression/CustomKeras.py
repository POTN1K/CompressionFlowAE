""" Custom Keras Operations

In this file the custom layer and losses are created, for its use in further developing the autoencoder model
"""

# Libraries
import tensorflow as tf
from keras.layers import Layer


# Custom layer
class Filter(Layer):
    """
    Filter, subclass Layer. The object created a custom Layer for a Neural Network.
    When initialized the filter is specified. It receives 4 latent space components and
    sets to zero the unused for the current training.

    It is a manual layer, every change in the filter attributes requires a recompiling of the model.
    Used for a hierarchical autoencoder. For specifics about its use, refer to
    hierarchicalGenerator.py file.

    Created based on other customizable layers, the specific methods get_config and build should not be touched.
    """

    def __init__(self, n=4, **kwargs):
        """
        Initialize Filter layer
        :param n: number of components filtered out of the model (1-4)
        :param kwargs: Other inputs for Layer class
        """
        super(Filter, self).__init__(**kwargs)
        self.n = n

    def get_config(self):
        """ Update config attributes """
        config = super().get_config()
        config.update({
            "n": self.n
        })
        return config

    def build(self, input_shape):
        """ Compile layer """
        super(Filter, self).build(input_shape)

    def call(self, x1, x2, x3, x4):
        """
        Sets last n latent spaces to zero. All latent spaces must have the same shape
        :param x1, x2, x3, x4: tf.Tensor, latent space
        :return: Combined latent space components into larger latent space
        """
        temp = tf.shape(x1)
        if temp is None:
            temp = 1
        if self.n < 4:
            x4 = tf.zeros(temp)
        if self.n < 3:
            x3 = tf.zeros(temp)
        if self.n < 2:
            x2 = tf.zeros(temp)
        x = tf.concat([x1, x2, x3, x4], axis=3)
        return x


# --------------------------------------------------------------------
# Custom loss functions
# These numerical differentiations are defined to comply with the tensorflow functions of Keras
# They take use an assumed grid size of 1
def central_difference(before, after):
    return tf.math.scalar_mul(0.5, tf.math.subtract(after, before))


def forward_difference(current, after):
    return tf.math.subtract(after, current)


def backward_difference(before, current):
    return tf.math.subtract(current, before)


def one_gradient(kxnxn, i, j, axis):
    # computes the spacial gradient of 1 point with respect to a specified axis
    if axis == 0:
        if i == 0:
            return forward_difference(kxnxn[:, i + 1, j], kxnxn[:, i, j])
        elif i == 23:
            return backward_difference(kxnxn[:, i, j], kxnxn[:, i - 1, j])
        else:
            return central_difference(kxnxn[:, i + 1, j], kxnxn[:, i - 1, j])
    elif axis == 1:
        if j == 0:
            return forward_difference(kxnxn[:, i, j + 1], kxnxn[:, i, j])
        elif j == 23:
            return backward_difference(kxnxn[:, i, j], kxnxn[:, i, j - 1])
        else:
            return central_difference(kxnxn[:, i, j + 1], kxnxn[:, i, j - 1])
    else:
        raise NotImplementedError


@tf.function
def custom_gradient(kxnxn, axis):
    # computes the gradient of a batch of the points in the images along a specified axis
    n = 24
    return tf.transpose([[one_gradient(kxnxn, i, j, axis=axis) for i in range(n)] for j in range(n)], (2, 0, 1))


def custom_loss_function(y_true, y_pred):
    """
    :param y_true: The original flow that the AE receives in a batch
    :param y_pred: The reconstructed flow in a batch
    :return: returns the tuple of loss terms (1 value per image) that are added later on in the keras architecture
    """
    # obtaining the velocity components
    u_true = y_true[:, :, :, 0]
    v_true = y_true[:, :, :, 1]
    u_pred = y_pred[:, :, :, 0]
    v_pred = y_pred[:, :, :, 1]

    # the energy is calculated by computing the mass specific kinetic energy point wise.
    energy_true = tf.math.add(tf.multiply(u_true, u_true), (tf.multiply(v_true, v_true)))
    energy_pred = tf.math.add(tf.multiply(u_pred, u_pred), (tf.multiply(v_pred, v_pred)))
    # then we take the difference point wise and average it so we get one value per image
    energy_difference = tf.math.reduce_mean(tf.math.squared_difference(energy_true, energy_pred), axis=[1, 2])
    # curl = nabla x velocity is computed with 2 dimensions so it returns a single value per point
    curl_true = tf.math.subtract(custom_gradient(u_true, axis=1), custom_gradient(v_true, axis=0))
    curl_pred = tf.math.subtract(custom_gradient(u_pred, axis=1), custom_gradient(v_pred, axis=0))
    # the difference is again taken point wise and averaged later
    curl_difference = tf.math.reduce_mean(tf.math.squared_difference(curl_true, curl_pred), axis=[1, 2])
    # the divergence computes the divergence in every point, takes the average of the values in an image
    # and takes the absolute value of that as we want the divergence to be as close to zero as possible
    divergence = tf.math.abs(
        tf.math.reduce_mean(tf.math.add(custom_gradient(u_pred, axis=0), custom_gradient(v_pred, axis=1)), axis=[1, 2]))
    # these are the regular MSE terms
    u_mse = tf.math.reduce_mean(tf.math.squared_difference(u_true, u_pred), axis=[1, 2])
    v_mse = tf.math.reduce_mean(tf.math.squared_difference(v_true, v_pred), axis=[1, 2])

    return energy_difference, curl_difference, u_mse, v_mse, divergence

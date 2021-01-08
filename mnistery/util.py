import numpy as np


def sigmoid_double(x):
    """ Represents the sigmoid function applied to a single real-valued argument x. """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(z):
    """ Applies the sigmoid function to a vector of values z, and returns a vector of results. """
    return np.vectorize(sigmoid_double)(z)


def sigmoid_deriv_double(x):
    """ Represents the derivative of the sigmoid function applied to a single real-valued argument x. """
    return sigmoid_double(x) * (1 - sigmoid_double(x))


def sigmoid_deriv(z):
    """ Applies the derivative of the sigmoid function to a vector of values z, and returns a vector of results. """
    return np.vectorize(sigmoid_deriv_double)(z)


class MSE:
    """ Mean square error calculator """
    @staticmethod
    def loss_function(predictions, labels):
        diff = predictions - labels
        return 0.5 * sum(diff * diff)[0]

    @staticmethod
    def loss_derivative(predictions, labels):
        return predictions - labels

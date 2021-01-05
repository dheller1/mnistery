import os
import numpy as np


def encode_onehot_digit(digit):
    """ Encodes a digit between 0 and 9 into a vector of length 10, which contains all zeroes except for the
    element at position 'digit', which is 1. """
    arr = np.zeros((10, 1))
    arr[digit] = 1.0
    return arr


def load_data():
    f = np.load(os.path.join('..', 'mnist.npz'))
    return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])


def reshape_data(data):
    """
    :param data: a 2-tuple which contains an (n, 28, 28) array for the features and an (n,) array for the labels
    :return: a list of 2-tuples with n entries. Each entry comprises a 784-long array for the features and
    a 10-long array representing a onehot-encoded label.
    """
    features = [np.reshape(x, (28*28, 1)) for x in data[0]]
    labels = [encode_onehot_digit(y) for y in data[1]]
    return list(zip(features, labels))

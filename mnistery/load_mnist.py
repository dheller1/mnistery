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


def _normalize_data(data):
    """ Normalizes pixel values from 8-bit number range (0-255) to a float between 0 and 1. """
    return data / 255.0


def reshape_and_normalize_data(data):
    """
    :param data: a 2-tuple which contains an (n, 28, 28) array for the features with values between 0 and 255,
    and an (n,) array for the labels
    :return: a list of 2-tuples with n entries. Each entry comprises a 784-long array for the features, with values
    between 0 and 1, and a 10-long array representing a onehot-encoded label.
    """
    features = [_normalize_data(np.reshape(x, (28*28, 1))) for x in data[0]]
    labels = [encode_onehot_digit(y) for y in data[1]]
    return list(zip(features, labels))

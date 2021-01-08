import os
import numpy as np
from matplotlib import pyplot as plt

from mnistery.load_mnist import load_data, reshape_and_normalize_data
from mnistery.neuralnet import SequentialNetwork, DenseLayer, ActivationLayer
from mnistery.util import sigmoid_double


def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)


def show_image(array):
    reshaped = np.reshape(array, (28, 28))
    plt.imshow(reshaped)
    plt.show()


def predict(features, weights, bias):
    # return np.dot(weights, features)
    return sigmoid_double(np.dot(weights, features) + bias)


def evaluate(data, digit, threshold, weights, bias):
    total_samples = 1.0 * len(data)
    correct_predictions = 0
    for x in data:
        prediction = predict(x[0], weights, bias)
        if prediction > threshold and np.argmax(x[1]) == digit:
            correct_predictions += 1
        if prediction <= threshold and np.argmax(x[1]) != digit:
            correct_predictions += 1
    return correct_predictions / total_samples


def main():
    train, test = load_data()
    train_data = reshape_and_normalize_data(train)  # a list of 2-tuples with normalized features
    test_data = reshape_and_normalize_data(test)

    avg = average_digit(train_data, 8)
    # print(avg)
    # show_image(avg)
    x3 = train_data[2][0]  # it's a 4
    x18 = train_data[17][0]  # it's an 8

    weights = np.transpose(avg)
    bias = -45

    print(evaluate(data=train_data, digit=8, threshold=0.5, weights=weights, bias=bias))
    print(evaluate(data=test_data, digit=8, threshold=0.5, weights=weights, bias=bias))

    weight0 = weight2 = weight4 = None
    bias0 = bias2 = bias4 = None

    loadfile = 'net_export.npz'
    if os.path.isfile(loadfile):
        f = np.load(loadfile)
        weight0 = f['weight0']
        weight2 = f['weight2']
        weight4 = f['weight4']
        bias0 = f['bias0']
        bias2 = f['bias2']
        bias4 = f['bias4']
        print(f'Restored network parameters from {loadfile}.')

    net = SequentialNetwork()
    net.add(DenseLayer(28*28, 14 * 28, weight0, bias0))
    net.add(ActivationLayer(14 * 28))
    net.add(DenseLayer(14 * 28, 7 * 28, weight2, bias2))
    net.add(ActivationLayer(7 * 28))
    net.add(DenseLayer(196, 10, weight4, bias4))
    net.add(ActivationLayer(10))

    epochs = 10
    for epoch in range(epochs):
        net.train(train_data, mini_batch_size=10, learning_rate=3.0)
        net.export(f'net_export{epoch}.npz')
        if test_data:
            n_test = len(test_data)
            print('Epoch {0}: {1} / {2} predictions correct'.format(epoch, net.evaluate(test_data), n_test))
        else:
            print('Epoch {0} complete'.format(epoch))


if __name__ == '__main__':
    main()

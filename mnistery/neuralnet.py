import random
import numpy as np
from mnistery.util import sigmoid, sigmoid_deriv, MSE


class Layer:
    """ A single layer in a sequential neural network.
    It supports forward-propagation of data and backpropagation of deltas. """
    def __init__(self):
        self.params = []

        self.previous = None  # parent layer
        self.next = None  # successor layer

        self.input_data = None
        self.output_data = None

        self.input_delta = None
        self.output_delta = None

    def connect_to(self, parent_layer):
        self.previous = parent_layer
        parent_layer.next = self

    def forward(self):
        """ Performs a forward step to the next layer, calculating output_data.
        To be implemented by concrete subclasses. """
        raise NotImplementedError

    def get_forward_input(self):
        # get input data from the previous step, except we are the first step overall.
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        """ Performs a backward step to the last layer, calculating output_delta.
        To be implemented by concrete subclasses. """
        raise NotImplementedError

    def get_backward_input(self):
        # get input delta from the output of the next step, except we are the last step.
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(self):
        pass

    def update_params(self, learning_rate):
        pass

    def describe(self):
        raise NotImplementedError


class ActivationLayer(Layer):
    """ A layer which uses the sigmoid function to activate neurons. """
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        # backward pass is the element-wise multiplication of the error term delta with the sigmoid derivative,
        # evaluated at the input to this layer.
        self.output_delta = delta * sigmoid_deriv(data)

    def describe(self):
        print('|-- ' + self.__class__.__name__)
        print('  |-- dimensions: ({},{})'.format(self.input_dim, self.output_dim))


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, weight=None, bias=None):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if weight is None:
            self.weight = np.random.randn(output_dim, input_dim)  # randomly init weight matrix..
        else:
            assert weight.shape == (output_dim, input_dim)
            self.weight = weight
        if bias is None:
            self.bias = np.random.randn(output_dim, 1)  # ..and bias
        else:
            assert bias.shape == (output_dim, 1)
            self.bias = bias
        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)  # deltas for weights
        self.delta_b = np.zeros(self.bias.shape)  # deltas for biases

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.delta_b += delta  # add current delta to bias delta
        self.delta_w += np.dot(delta, data.transpose())
        self.output_delta = np.dot(self.weight.transpose(), delta)

    def update_params(self, learning_rate):
        self.weight -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        print('|-- ' + self.__class__.__name__)
        print('  |-- dimensions: ({},{})'.format(self.input_dim, self.output_dim))


class SequentialNetwork:
    def __init__(self, loss=None):
        print('Initialize Network...')
        self.layers = []
        if loss is None:
            self.loss = MSE()
        else:
            self.loss = loss

    def add(self, layer):
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect_to(self.layers[-2])

    def export(self, filename):
        content = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                content[f'weight{i}'] = layer.weight
            if hasattr(layer, 'bias'):
                content[f'bias{i}'] = layer.bias
        np.savez(filename, **content)

    def train(self, training_data, mini_batch_size, learning_rate):
        n = len(training_data)
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.train_batch(mini_batch, learning_rate)

    def train_batch(self, mini_batch, learning_rate):
        self.forward_backward(mini_batch)
        self.update(mini_batch, learning_rate)

    def update(self, mini_batch, learning_rate):
        learning_rate = learning_rate / len(mini_batch)
        for layer in self.layers:
            layer.update_params(learning_rate)
        for layer in self.layers:
            layer.clear_deltas()

    def forward_backward(self, mini_batch):
        """ Performs full forward-backward passes for each sample in mini_batch. """
        for x, y in mini_batch:
            self.layers[0].input_data = x
            for layer in self.layers:  # feed single sample forward through each layer
                layer.forward()
            # calculate loss gradient based on labels y and predictions in the last layer, store it as the last
            # layer's input delta.
            self.layers[-1].input_delta = self.loss.loss_derivative(self.layers[-1].output_data, y)
            for layer in reversed(self.layers):
                layer.backward()  # back-propagate deltas through each layer

    def single_forward(self, x):
        """ Performs a single forward pass for sample x and returns the result (i.e. predictions). """
        self.layers[0].input_data = x
        for layer in self.layers:
            layer.forward()
        return self.layers[-1].output_data

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.single_forward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

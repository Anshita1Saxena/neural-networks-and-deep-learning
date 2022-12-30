import numpy as np
import random
import time

"""
network_matrix.py
A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation. It is not optimized and omits many desirable features 
as this is a baby neural network implementation.
This implementation is fully matrix based approach, we are processing here the examples batch-wise.
Python 3 Implementation
Reference to understand Weight and Biases Dimensions: 
https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html#dynamic-resizing
"""


class NetworkMatrix(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.sizes = sizes
        self.num_layers = len(sizes)
        # This np.random.randn() will generate the gaussian distribution with mean = 0 and variance = 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Return the output of the network if 'a' is input"""
        # This function is utilized at the time of evaluation of test examples.
        # We are evaluating the test set example by example though.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic gradient descent. The "training_data" is
        a list of tuples "(x, y)" representing the training inputs and the desired outputs. The other
        non-optional parameters are self-explanatory. If "test-data" is provided then the network will
        evaluated against the test data after each epoch, and partial progress printed out. This is useful
        for tracking progress, but slows things down substantially."""
        start = time.time_ns()
        n_test = 0
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        # From python 3, xrange doesn't exist, instead use range
        # In python 2, it existed and faster than range because it is a sequence object and evaluates lazily.
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batch_num = int(len(training_data)/mini_batch_size)
            X, Y = [], []
            mini_batch_matrix = []
            # TODO: Improvement on replacing for loop with some numpy or pandas hacks for faster processing
            # Loop implementation: Arranging the data into a matrix to pass it to the network
            k = 0
            for mini_batch in range(mini_batch_num):
                inter_x = np.zeros((mini_batch_size, 784))
                inter_y = np.zeros((mini_batch_size, 10))
                for x, y in training_data[k:k+mini_batch_size]:
                    for i in range(mini_batch_size):
                        inter_x[i] = x[:, 0]
                        inter_y[i] = y[:, 0]
                X.append(inter_x)
                Y.append(inter_y)
                mini_batch_matrix.append((inter_x, inter_y))
                k += mini_batch_size
            # After the creation of final matrix, send the batches one by one
            for mini_batch in mini_batch_matrix:
                self.update_mini_batch(mini_batch, eta)
            # If the test data is provided, program will evaluate the network after each epoch of
            # training data, and print out the partial progress. This is useful for tracking progress,
            # but slows things down substantially.
            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            # Calculate the time at each epoch to compare the efficiency in comparison to simple network code.
            end = time.time_ns()
            print(end - start / 1e9, " seconds")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation
        to a single mini batch. The "mini-batch" is a list of tuples "(x,y)", and "eta" is the learning
        rate."""
        # Call the backpropagation function and directly update the weights for the 10 samples as it is now a matrix.
        delta_nabla_b, delta_nabla_w = self.backprop(mini_batch[0], mini_batch[1])
        self.weights = [w - (eta / len(mini_batch[0])) * nw for w, nw in zip(self.weights, delta_nabla_w)]
        self.biases = [b - (eta / len(mini_batch[0])) * np.sum(nb, axis=0).reshape(b.shape) for b, nb in
                       zip(self.biases, delta_nabla_b)]

    def backprop(self, x, y):
        """Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
        nabla_b and nabla_w are layer by layer list of numpy arrays, similar to self.biases and
        self.weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        # List to store all the activations layer by layer
        activations = [x]
        # list to store all the z vectors layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w.T) + np.sum(b, axis=1)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_primes(zs[-1])
        nabla_b[-1] = delta  # Equation 3 at output layer
        nabla_w[-1] = np.dot(delta.T, activations[-2])  # Equation 4 at output layer

        for l in range(2, self.num_layers):
            # Since the numbering starts from 2, we place -ve sign before the number
            # this way the layers will be reversed
            z = zs[-l]
            sp = sigmoid_primes(z)
            delta = np.dot(delta, self.weights[-l+1]) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta.transpose(), activations[-l-1])

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # argmax for highest activation
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        # sum of the perfectly matching values
        return sum(int(x == y) for x, y in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        # According to the quadratic cost derivative
        return (output_activations - y)


# Miscellaneous functions
def sigmoid(z):
    """Return the sigmoid function result"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_primes(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

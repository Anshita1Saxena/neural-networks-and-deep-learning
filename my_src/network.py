import numpy as np
import random
import time


"""
network.py
A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network.
Gradients are calculated using backpropagation. It is not optimized and omits many desirable features 
as this is a baby neural network implementation.  
Python 3 Implementation
"""


class Network(object):

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
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # (2,1)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # (2, 784)

    def feed_forward(self, a):
        """Return the output of the network if 'a' is input"""
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
        # From python 3, xrange doesn't exist.
        # In python 2, it existed and faster than range because it is a sequence object and evaluates lazily.
        for j in range(epochs):
            random.shuffle(training_data)
            # Divide the training data into batches of 10 examples.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # If the test data is provided, program will evaluate the network after each epoch of
            # training data, and print out the partial progress. This is useful for tracking progress,
            # but slows things down substantially.
            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
        end = time.time_ns()
        print(end - start / 1e9, " seconds")

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation
        to a single mini batch. The "mini-batch" is a list of tuples "(x,y)", and "eta" is the learning
        rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # The below loop is to process the training example one by one
        for x, y in mini_batch:
            # Below is the function of backpropagation algorithm is the fast
            # way of computing the gradient of cost function.
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [dnw + nw for dnw, nw in zip(delta_nabla_w, nabla_w)]
            nabla_b = [dnb + nb for dnb, nb in zip(delta_nabla_b, nabla_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

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
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        # For computing the cost and outer layer weight and biases
        delta = self.cost_derivative(activations[-1], y) * sigmoid_primes(zs[-1])
        nabla_b[-1] = delta     # Equation 3 at output layer
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())    # Equation 4 at output layer
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # For computing the entire network traveling from back layer by layer
        for l in range(2, self.num_layers):
            # Since the numbering starts from 2, we place -ve sign before the number
            # this way the layers will be reversed
            z = zs[-l]
            sp = sigmoid_primes(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

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

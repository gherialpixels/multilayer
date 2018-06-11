
import numpy as np


def sigmoid(num, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-num))


def threshold(num):
    if num >= 0.5:
        return 1.0
    else:
        return 0.0


def loss(output, prediction):
    return output - prediction


def cost(errs):
    return sum(errs) / (len(errs) + len(errs[0]))


def train(inputs, output, weights, lr=0.03):
    layer0 = inputs
    for b in xrange(100):
        layer1 = predict(layer0, weights[0])
        layer2 = predict(layer1, weights[1])

        l2_error = loss(output, layer2)
        # part of the sigmoid function, but did not work
        l2_delta = l2_error * (layer2 - layer2 * layer2)

        l1_error = l2_delta.dot(weights[1].T)
        l1_delta = l1_error * (layer1 - layer1 * layer1)

        weights[1] += layer1.T.dot(l2_delta)
        weights[0] += layer0.T.dot(l1_delta)

        if b % 5 == 0:
            error = l2_error
            print "average error: ", np.mean(np.abs(error))


def predict(inputs, weights, testing=False):
    if not testing:
        # -1 to act as a bias
        return sigmoid(np.dot(inputs, weights) - 1)
    else:
        return threshold(sigmoid(np.dot(inputs, weights)))


def test(inputs, output, weights):
    accuracy = []

    layer0 = inputs
    layer1 = predict(layer0, weights[0])
    layer2 = predict(layer1, weights[1])

    guess = threshold(layer2.all())
    print guess, "\n", layer2


np.random.seed(120)

input_size = 100
n_weights = 1

# input
x = np.array([[np.random.randint(-200, 200), np.random.randint(-200, 200)] \
            for a in xrange(input_size)])

# output
y = np.array([[1 if x[i][0] ** 2 + x[i][1] ** 2 > 16 else 0] for i in xrange(input_size)])

# weights / synapses
syn0 = 2 * np.random.random((2, input_size)) - 1
syn1 = 2 * np.random.random((input_size, 1)) - 1


train(x, y, [syn0, syn1])

"""
xtest = np.array([[np.random.randint(-200, 200), np.random.randint(-200, 200)] \
                for a in range(50)])
ytest = np.array([[1 if (x[i][0] ** 2)* 3 + 20 > x[i][1] else 0] \
                for i in range(50)])

test(xtest, ytest, [syn0, syn1])
"""


import numpy as np
import random


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def threshold(num):
    if num >= 0.5:
        return 1
    else:
        return 0


def loss(output, prediction):
    return output - prediction


def cost(errs):
    return sum(errs) / (len(errs) + len(errs[0]))


def train(inputs, output, weights, lr=0.03):
    layers = [inputs]
    weight_length = len(weights)
    for b in xrange(600):
        for i in range(weight_length):
            if not len(layers) > weight_length:
                pass
            else:
                layers.append(predict(layers[-1], syn))

        error = loss(layers[-1], output)

        weights[0] += layer0.T.dot(error) * lr
        weights[1] += layer1.T.dot(error) * lr

        if b % 100 == 0:
            print "average error: ", cost(error)


def predict(inputs, weights, testing=False):
    if not testing:
        return sigmoid(np.dot(inputs, weights))
    else:
        return threshold(sigmoid(np.dot(inputs, weights)))


def test(inputs, output, weights):
    accuracy = []

    layer0 = inputs
    for b in xrange(10):
        layer1 = predict(layer0, weights[0])
        layer2 = predict(layer1, weights[1])

        accuracy.append(cost(loss(output, threshold(layer2))))

    print "accuracy: ", sum(accuracy) / len(accuracy)

random.seed(1)
np.random.seed(1)

input_size = 100
n_weights = 1

# input
x = np.array([[random.randint(-200, 200), random.randint(-200, 200)] for a in xrange(input_size)])

# output
y = np.array([[1 if x[i][0] ** 3 > x[i][1] else 0] for i in xrange(input_size)])

# weights / synapses
synapses = [2 * np.random.random((2, input_size)) - 1]  # this -1 is the bias
for i in range(n_weights):
    synapses.append(2 * np.random.random((input_size, input_size)) - 1)

synapses.append(2 * np.random.random((input_size, 1)) - 1)


train(x, y, synapses)

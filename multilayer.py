
import numpy as np
import random

random.seed(4)


def return_node(n_inputs, layer, bias):
    node = {"weights": [random.uniform(-1, 1) for k in xrange(n_inputs)], "bias": bias, "layer": layer, "lr": 0.03}
    return node


def return_point(x, y, b=1.0, c=0.0):
    if b * x * x + c > y:
        return {"pos": (x, y), "target": 1}
    else:
        return {"pos": (x, y), "target": 0}


def order_nodes(nodes, max_layer, new_list):
    if max_layer >= 0:
        for nd in nodes:
            if nd["layer"] == max_layer:
                new_list.append(nd)

        max_layer -= 1
        return order_nodes(nodes, max_layer, new_list)
    else:
        return new_list[::-1]


def calculate_error(errs):
    return sum(errs) / len(errs)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convert_probability(x):
    if x >= 0.5:
        return 1
    else:
        return 0


def guessing(node, inputs):
    total = 0
    weights = node["weights"]
    weight_length = len(weights)
    for m in xrange(weight_length):
        total += weights[m] * inputs[m]

    return sigmoid(total + node["bias"])


def train(nodes, point, errors):
    # IMPORTANT: Here is where the max_layer is set

    # order the nodes according to layer (because why not)
    nodes_ordered = order_nodes(nodes, 1, [])

    # compute the output of the first layer of hidden nodes
    # these are then parsed to the second hidden layer of nodes
    guess00 = guessing(nodes_ordered[0], point["pos"])
    guess01 = guessing(nodes_ordered[1], point["pos"])

    # second layer of hidden nodes (only one node though)
    guess10 = guessing(nodes_ordered[2], [guess00, guess01])

    # find the error, append to a list to get a percentage error
    error = point["target"] - guess10
    errors.append(error)
    if len(errors) > 20:
        errors = errors[1:]

    # adjusting weights using the error
    for a in xrange(len(nodes_ordered)):
        weights = nodes_ordered[a]["weights"]

        # first and second layer distinguished differently with for-loop, will generalise next time
        if a < 2:
            weights[a] += point["pos"][a] * error * nodes_ordered[a]["lr"]
        else:
            weights[0] += guess00 * error * nodes_ordered[a]["lr"]
            weights[1] += guess01 * error * nodes_ordered[a]["lr"]

    return errors


def display_training(nodes):
    # points on a 2D grid
    points = [return_point(random.randint(-200, 200), random.randint(-200, 200)) for i in xrange(100)]

    error_list = []

    print "output: \n"
    length = len(points)
    for b in xrange(1200):
        print "<-- training started/restarted -->"
        for k in xrange(length):
            error_list = train(nodes, points[k], error_list)
            if k % 50 == 0:
                print calculate_error(error_list)

    print "end of training..."


def test(nodes):
    accuracy = []

    testing_points = [return_point(random.randint(-200, 200), random.randint(-200, 200), 0.5, 50) for i in xrange(50)]

    nodes_ordered = order_nodes(nodes, 1, [])

    print "\ntesting network\naccuracy"

    for point in testing_points:
        guess00 = guessing(nodes_ordered[0], point["pos"])
        guess01 = guessing(nodes_ordered[1], point["pos"])

        guess10 = guessing(nodes_ordered[2], [guess00, guess01])

        if convert_probability(guess10) == point["target"]:
            accuracy.append(1.0)
        else:
            accuracy.append(0.0)

        average_accuracy = calculate_error(accuracy)
        print average_accuracy * 100


nds = [return_node(2, 0, -1), return_node(2, 0, -1), return_node(2, 1, -1)]

display_training(nds)
test(nds)

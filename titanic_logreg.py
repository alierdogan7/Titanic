import pprint

from math import log10

import math
import numpy as np


def predict_and_get_accuracy():
    dataset = read_csv("s")
    x = dataset['training']['x']
    y = dataset['training']['y']
    m = len(x)
    alpha = 0.01
    max_iters = 3000
    theta0, theta_set = gradient_descent(x, y, alpha, max_iters)

    correct_count = 0
    for i in range(m):
        result = h_theta(theta0, theta_set, x[i])
        if result >= 0.5:
            predict = 1
            if predict == y[i]:
                correct_count += 1
        else:
            predict = 0
            if predict == y[i]:
                correct_count += 1

    accuracy = correct_count / float(m) * 100
    print("Accuracy with theta0:%f, theta_set:%s, alpha:%f, max_iters:%d: %f percent." %
                            (theta0, str(theta_set), alpha, max_iters, accuracy))


def gradient_descent(x, y, alpha=0.01, max_iterations=1000, ep=0.0001 ):
    converged = False
    m = x.shape[0]  # number of rows
    iterations = 0

    # initial values of thetas
    theta0 = np.random.rand()
    theta_set = np.random.random(x.shape[1])

    # error with initial thetas
    J = cost_function(x, y, theta0, theta_set)

    while not converged:
        grad_theta0 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) for i in range(m)])
        grad_theta1 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][0] for i in range(m)])
        grad_theta2 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][1] for i in range(m)])
        grad_theta3 = 1.0/m * sum([ (h_theta(theta0, theta_set, x[i]) - y[i]) * x[i][2] for i in range(m)])

        #update thetas
        theta0 -= alpha * grad_theta0
        theta_set[0] -= alpha * grad_theta1 #update theta1
        theta_set[1] -= alpha * grad_theta2 #update theta2
        theta_set[2] -= alpha * grad_theta3 #update theta3

        #compute the error again
        e = cost_function(x, y, theta0, theta_set)

        if abs(J-e) <= ep:
            converged = True
            print("Converged with %d iterations." % iterations)

        J = e #updating the cost with new thetas
        iterations += 1

        if iterations >= max_iterations:
            converged = True
            print("Reached max. iterations: %d ." % iterations)

    return theta0, theta_set


def cost_function(x, y, theta0, theta_set):
    # x is a mx3 matrix for all examples' features
    # y is a mx1 matrix for all examples' classes
    m = len(x)
    sum = 0

    for i in range(m):
        h_theta_x_i = h_theta(theta0, theta_set, x[i])
        sum += y[i] * log10(h_theta_x_i) + (1 - y[i]) * log10(1 - h_theta_x_i)

    return -sum / m


def h_theta(theta0, theta_set, x):
    # theta_set is a 3x1 matrix
    # x is a 1x3 matrix for a single example's features
    # returns theta0 + theta1 * x1 + theta2 * x2 + theta3 * x3
    return sigmoid(theta0 + x.dot(theta_set).sum())


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# X is a data set matrix with parameters: x1 Pclass, x2 Sex, x3 Age
def read_csv(filename):
    splitted_data = {'training': {'x': [], 'y': []},
                     'validation': {'x': [], 'y': []},
                     'test': {'x': [], 'y': []}}

    dataset = np.loadtxt(open("titanicdata.csv", "rb"), delimiter=",", skiprows=1)
    x = dataset[:, [1, 2, 3]] # separate features and classes
    y = dataset[:, [0]]

    normalize_features(x)

    splitted_data['training']['x'] = x[:400]
    splitted_data['training']['y'] = y[:400]
    splitted_data['validation']['x'] = x[400:700]
    splitted_data['validation']['y'] = y[400:700]
    splitted_data['test']['x'] = x[700:]
    splitted_data['test']['y'] = y[700:]

    return splitted_data


def normalize_features(x):
    age_set = [data[2] for data in x]  # index 2 corresponds to age column
    min_age, max_age = min(age_set), max(age_set)

    for index in range(len(x)):
        x[index][2] = (x[index][2] - min_age) / (max_age - min_age)


predict_and_get_accuracy()
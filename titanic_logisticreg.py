import math
import matplotlib.pyplot as plt
import numpy as np

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    # total error, J(theta)
    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)])
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] )

        if abs(J-e) <= ep:
            print 'Converged, iterations: ', iter, '!!!'
            converged = True

        J = e   # update error
        iter += 1  # update iter

        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True

    return t0,t1



def read_csv(filename):
    dataset = []
    splitted_data = {'training': [], 'validation': [], 'test':[]}

    with open(filename) as csvfile:
        read_array = lambda line: line.strip('\r\n').split(',')
        attributes = read_array(csvfile.readline())

        for row in csvfile:
            dataset.append({attr: float(value) if attr == 'Age' else int(value) for attr, value in zip(attributes, read_array(row))})

        normalize_features(dataset)

        splitted_data['training'] = dataset[:401]
        splitted_data['validation'] = dataset[401:701]
        splitted_data['test'] = dataset[701:]

        return splitted_data


def normalize_features(dataset):
    age_set = [data['Age'] for data in dataset]
    min_age, max_age = min(age_set), max(age_set)

    for index in range(len(dataset)):
        dataset[index]['Age'] = (dataset[index]['Age'] - min_age) / (max_age - min_age)



# def sigmoid(z):
#     return 1 / (1 + math.exp(-z))

# assumıng that m = number of traınıng data, lınes ın the tıtanıc dataset.
# ıts gonna be m by 3, plus m by 1 bıas term
# ın terms of matrıx multıplıacatıon
# x0 + X*W = Y
# h_theta(y).
# W ıs a 3 by 1 vector
# yıeldıng m by 1 output predıctıon.

def h_theta(**kwargs):
    sum = kwargs['theta0']
    for theta, x in zip(kwargs['theta_set'], kwargs['param_set']):
        sum += theta * x
    return sum


# def cost_function(theta_set, data_set):
#     m = len(data_set)
#     sum = 0
#     for example in data_set:
#         y_i = example['Survived']
#         param_set = [ example['Age'], example['Pclass'], example['Sex']]
#         h_theta = hypothesis(theta_set=theta_set, param_set=param_set)
#
#         sum += ( (-y_i * math.log10(h_theta)) - ((1 - y_i) * math.log10(1 - h_theta)) )
#
#     return sum / m
#
#
# '''
# theta_j_param_name: derivative of J(theta) is going to be taken according to that parameter
# theta_set: current theta values
# dataset: example rows
# '''
# def gradient_function(theta_j_param_name, theta_set, data_set):
#     m = len(data_set)
#     sum = 0
#     for example in data_set:
#         y_i = example['Survived']
#         param_set = [ example['Age'], example['Pclass'], example['Sex']]
#         h_theta = hypothesis(theta_set=theta_set, param_set=param_set) #calculate h_theta of x
#
#         sum += ( h_theta - y_i) * example[theta_j_param_name]
#
#     return sum / m
#
#
# def converge():
#     theta_set =

def test():
    # x = np.arange(-20, 20, 0.1)
    # y = list(map(sigmoid, x))
    # plt.plot(x, y)
    # plt.show()

    normalized_data = read_csv('titanicdata.csv')




test()


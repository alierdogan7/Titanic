import math
import operator
import pprint
import matplotlib.pyplot as plt
import numpy as np

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

    # pprint.pprint(dataset)


def distance_measure(instance1, instance2):
    #Pclass, sex, age
    instance1 = [instance1[attr] for attr in ('Age', 'Sex', 'Pclass',)]
    instance2 = [instance2[attr] for attr in ('Age', 'Sex', 'Pclass',)]

    distance = 0
    for feature in zip(instance1, instance2):
        distance += (feature[0] - feature[1])**2 #euclidian distance
        # distance += abs(feature[0] - feature[1]) #manhattan distance

    distance = math.sqrt(distance)
    return distance


def get_k_neighbors(instance, dataset, k):
    distances = []
    for data in dataset:
        distances.append((distance_measure(instance, data), data))

    distances.sort(key=operator.itemgetter(0))
    return distances[:k]


def get_response(neighbors): #takes the list of tuples, (distance, data), returned from get_k_neighbors() method
    survived, died = 0, 0
    for distance, data in neighbors:
        # if distance == 0: # fix for distance=0 issue. if it is, predict with the same class
        #     return data['Survived']
        #
        # vote_weight = 1 / (distance**2) # for precise predictions, do a weighted voting for predicting the class

        if data['Survived']:
            # survived += vote_weight
            survived += 1
        else:
            # died += vote_weight
            died += 1

    return 1 if survived >= died else 0

def scatter_plot(dataset):
    # np.random.seed(5)
    # x = np.arange(1, 101)
    # y = 20 + 3 * x + np.random.normal(0, 60, 100)
    # plt.plot(x, y, "o")

    plt.show()

# print(distance_measure(1,1))
#
# x=get_k_neighbors({'Age': 10, 'Pclass': 3, 'Sex': 1},
#                 [{'Age': 10, 'Pclass': 5, 'Sex': 1}, {'Age': 10, 'Pclass': 7, 'Sex': 1},
#                 {'Age': 10, 'Pclass': 2, 'Sex': 1}], 1)

normalized_data = read_csv('titanicdata.csv')

# scatter_plot(normalized_data['training'])

def print_cv_accuracy():
    #cross validation
    accuracies = []
    for k in [1] + list(range(2, 17, 2)):
    # for k in range(2, 13):
        validation_set_size = len(normalized_data['validation'])
        accurate_count = 0
    
        for data in normalized_data['validation']:
            neighbors = get_k_neighbors(data, normalized_data['training'], k)
            if data['Survived'] == get_response(neighbors):
                accurate_count += 1
    
        accuracies.append((k, (float(accurate_count) / validation_set_size)))

    pprint.pprint(accuracies)
    for k, acc in accuracies:
        print("Accuracy of kNN with parameter k=%d is %f" % (k, acc*100))


def print_testing_accuracy(k=1):
    #testing
    testing_set_size = len(normalized_data['test'])
    accurate_count = 0
    
    for data in normalized_data['test']:
        neighbors = get_k_neighbors(data, normalized_data['training'] + normalized_data['validation'], k)
        if data['Survived'] == get_response(neighbors):
            accurate_count += 1
    
    print('Accuracy of k=%d:  %f' % (k, 100*(float(accurate_count) / testing_set_size)))

def read_csv_to_matrix(filename):
    splitted_data = {'training': {'x': [], 'y': []},
                     'validation': {'x': [], 'y': []},
                     'test': {'x': [], 'y': []}}

    dataset = np.loadtxt(open("titanicdata.csv", "rb"), delimiter=",", skiprows=1)
    x = dataset[:, [1, 2, 3]] # separating features and classes
    y = dataset[:, [0]]

    age_set = [data[2] for data in x]  # index 2 corresponds to age column
    min_age, max_age = min(age_set), max(age_set)

    for index in range(len(x)):
        x[index][2] = (x[index][2] - min_age) / (max_age - min_age)


    splitted_data['training']['x'] = x[:400]
    splitted_data['training']['y'] = y[:400]
    splitted_data['validation']['x'] = x[400:700]
    splitted_data['validation']['y'] = y[400:700]
    splitted_data['test']['x'] = x[700:]
    splitted_data['test']['y'] = y[700:]

    return splitted_data

#
# def sklearn_accuracy():
#     # X = [[0], [1], [2], [3]]
#     dataset = read_csv_to_matrix('')
#     training = dataset['training']
#     validation = dataset['validation']
#     test = dataset['test']
#
#     trainingX = training['x']
#     y = training['y']
#     trainingY = np.array(y.T)[0]
#
#     from sklearn.neighbors import KNeighborsClassifier
#
#
#     #cross-validation data
#     accuracies = []
#     for k in [1] + list(range(2, 17, 2)):
#     # for k in range(2, 13):
#         neigh = KNeighborsClassifier(n_neighbors=k)
#         neigh.fit(trainingX, trainingY)
#         neigh.predict_proba(validation['x'])
#         score = neigh.score(validation['x'], validation['y'])
#         print("Score of scikitlearn on this with K=%d ==> %f" % (k, score))
#
#     pprint.pprint(accuracies)
#
    # # testing data
    # k = 4
    # neigh = KNeighborsClassifier(n_neighbors=k)
    # neigh.fit(trainingX, trainingY)
    # neigh.predict_proba(test['x'])
    # score = neigh.score(test['x'], test['y'])
    # print("Score of scikitlearn on this with K=%d ==> %f" % (k, score))

# sklearn_accuracy()

# print_cv_accuracy()
#
# for k in [1] + list(range(2, 17, 2)):
#     print_testing_accuracy(k)
#!/usr/bin/env python
# encoding: utf-8

"""Uses the SOMs in some test cases. Some visualizations are shown when
available and some statistics are printed. The data used isn't included
however.
"""

from itertools import islice
from sklearn import metrics
from som.basic_som import BasicSOM
from som.gridless_som import GridlessSOM
from som import normalize
import csv
import random


def test_colors():
    """Quickly tests the SOMS for uniformly distributed random data in three
    dimensions. They are represented as colors.
    """
    data = [(random.random(), random.random(), random.random())
            for x in range(1000)]
    test_data = [(random.random(), random.random(), random.random())
                 for x in range(1000)]
    # You can try it with some assymetric data:
    # for i in range(100):
    #     _, g, b = data[i]
    #     data[i] = 1, g, b

    # Test with a normal SOM
    som = BasicSOM(data, 40, 40, init_variance=10)
    som.train(iterations=1000)
    som.color_plot()
    som.hit_map()
    som.distance_map()
    print('Basic SOM (40x40, var=10)')
    print_som_measures(som, test_data)

    # Test with an assymetric SOM
    topology_kwargs = {
        'toroidal': False,
        'push': .1,
        'inhibition': 10,
    }
    som = GridlessSOM(data, 40, 40, init_learning_rate=.5,
                      topology_kwargs=topology_kwargs)
    som.train(iterations=1000)
    som.color_plot()
    print('ASOM (40x40, lr=.5, %s)' % repr(topology_kwargs))
    print_som_measures(som, test_data)


def read_iris_data():
    """Reads the iris dataset and returns a tuple of data and labels."""

    with open('data/iris.csv') as file_:
        both = (([float(y) for y in x[:4]], x[4]) for x in csv.reader(file_))
        return tuple(zip(*both))


def test_iris(data, labels):
    """Quickly test the SOMs for the iris dataset and shows some statistics."""

    # Normalize the data
    data = list(normalize(data))

    # Split data set in train and test set (Iris data set contains 150 entries)
    train_data = data[:100]
    train_labels = labels[:100]
    test_data = data[100:]
    test_labels = labels[100:]

    # Test with a regular SOM
    som = BasicSOM(train_data, 8, 8, init_variance=4, init_learning_rate=.1,
                   labels=labels)
    som.train(iterations=200)
    som.label_plot()
    som.polar_plots()
    som.hit_map()
    som.distance_map()
    print('Basic SOM (8x8, var=4, lr=.1)')
    print_som_measures(som, test_data)
    print_cluster_measures(som, test_data, test_labels)

    topology_kwargs = {
        'toroidal': False,
        'push': .1,
        'inhibition': 20,
    }
    som = GridlessSOM(train_data, 8, 8, init_learning_rate=.5,
                      labels=train_labels, topology_kwargs=topology_kwargs)
    som.train(iterations=200)
    som.label_plot()
    print('ASOM (8x8, lr=.5, %s)' % repr(topology_kwargs))
    print_som_measures(som, test_data)
    print_cluster_measures(som, test_data, test_labels)


def read_bone_marrow_data(only_first_n=None):
    """Reads a certain csv file of cytometric data and returns both the data
    and the class labels.
    """

    def partial_csv(file_):
        """Given an CSV file, generates its 2d array without header and only the
        first n lines. Reads all if only_first_n is None.
        """
        return islice(csv.reader(file_), 1, only_first_n)

    # Read the files
    with open('data/BoneMarrow_Basal1.csv') as file_:
        data = [list(map(float, x)) for x in partial_csv(file_)]
    with open('data/manualGating.csv') as file_:
        labels = (x[1] for x in partial_csv(file_))
        labels = [None if x == 'Unknown' else x for x in labels]

    # Make sure there is no bias by shuffling the data
    both = list(zip(data, labels))
    random.shuffle(both)

    # Return a tuple of data and labels
    return tuple(zip(*both))


def test_bone_marrow(data, labels):
    """Quickly test the SOMs for the bone marrow dataset and shows some
    statistics.
    """
    # Normalize the data
    data = list(normalize(data))

    # Split into train and test set
    train_data = data[:4000]
    train_labels = labels[:4000]
    test_data = data[4000:5000]
    test_labels = labels[4000:5000]

    som = BasicSOM(data, 40, 40, init_variance=14, init_learning_rate=.1,
                   labels=labels)
    som.train(iterations=2000)
    som.label_plot()
    som.polar_plots()
    som.distance_map()
    som.hit_map()
    print('Basic SOM (40x40, var=14, lr=.5)')
    print_som_measures(som, test_data)
    print_cluster_measures(som, test_data, test_labels)

    topology_kwargs = {
        'toroidal': False,
        'push': .12,
        'inhibition': 40,
    }
    som = GridlessSOM(train_data, 40, 40, init_learning_rate=.5,
                      labels=train_labels, topology_kwargs=topology_kwargs)
    som.train(iterations=2000)
    som.label_plot()
    print('ASOM (40x40, lr=.5, %s)' % repr(topology_kwargs))
    print_som_measures(som, test_data)
    print_cluster_measures(som, test_data, test_labels)


def print_som_measures(som, test_data):
    """Prints the quantization error and topological error."""
    print('  Quantization error:', som.quantization_error(test_data))
    print('  Topological error:', som.topology_error(test_data))


def print_cluster_measures(som, test_data, test_labels):
    """Prints some clustering statistics."""

    pred_labels = som.predict(test_data)
    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    adj_rand = metrics.adjusted_rand_score(test_labels, pred_labels)
    adj_mi = metrics.adjusted_mutual_info_score(test_labels, pred_labels)
    norm_mi = metrics.normalized_mutual_info_score(test_labels, pred_labels)
    print('  Accuracy:', accuracy)
    print('  Adjusted rand score:', adj_rand)
    print('  Adjusted mutual info:', adj_mi)
    print('  Normalized mutual info:', norm_mi)


if __name__ == '__main__':
    # Test both SOMs (basic an asymmetric) with random uniformly distributed
    # three-dimensional data
    test_colors()

    # Test with Iris dataset
    data, labels = read_iris_data()
    test_iris(data, labels)

    # Test with cytometric data
    data, labels = read_bone_marrow_data()
    test_bone_marrow(data, labels)

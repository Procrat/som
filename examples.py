#!/usr/bin/env python
# encoding: utf-8

from itertools import islice
from sklearn import metrics
from som.basic_som import BasicSOM
from som.gridless_som import GridlessSOM
from som import normalize
import csv
import random


#random.seed(3)


def read_bone_marrow_data(only_first_n=None):
    def partial_csv(file_):
        """Given an CSV file, generates a 2d array without header and only the
        first n lines. Reads all if only_first_n is None."""
        return islice(csv.reader(file_), 1, only_first_n)

    with open('data/BoneMarrow_Basal1.csv') as file_:
        data = [list(map(float, x)) for x in partial_csv(file_)]
    with open('data/manualGating.csv') as file_:
        labels = (x[1] for x in partial_csv(file_))
        labels = [None if x == 'Unknown' else x for x in labels]

    # Make sure there is no bias by shuffling the data
    both = list(zip(data, labels))
    random.shuffle(both)
    return tuple(zip(*both))


def test_bone_marrow(data, labels):
    data = list(normalize(data))

    train_data = data[:4000]
    train_labels = labels[:4000]
    #test_data = data[4000:5000]
    #test_labels = labels[4000:5000]
    test_data = data[4000:4300]
    test_labels = labels[4000:4300]

    #som = BasicSOM(data, 40, 40, init_variance=14, init_learning_rate=.1,
                   #labels=labels)
    #som.train(iterations=2000)
    #som.label_plot()
    #som.polar_plots()
    #som.distance_map()
    #som.hit_map()
    #print(cluster_scores(som, test_data, test_labels))
    #pred_labels = som.predict(test_data)
    #accuracy = sum(x == y for x, y in zip(pred_labels, test_labels)) / len(test_labels)
    #print('accuracy:', accuracy)
    #print('Measures (basic, 40, 40, var=7, lr=.5)')
    #print('QE', som.quantization_error(test_data))
    #print('TE', som.topology_error(test_data))
    #return

    #data = list(normalize(data))
    topology_kwargs = {
        'toroidal': True,
        'push': 0,
        #'push': 0,
        'inhibition': 50,
        #'inhibition': 200,
    }
    som = GridlessSOM(train_data, 40, 40, init_learning_rate=.5,
                      labels=train_labels, topology_kwargs=topology_kwargs)
    #som.train(iterations=1000)
    som.train(iterations=1000)
    #som.train(iterations=10000)
    #import pprint; pprint.pprint(som)
    #som.voronoi_plot()
    som.label_plot()
    #som.polar_plots()
    #print(cluster_scores(som, test_data, test_labels))
    pred_labels = som.predict(test_data)
    accuracy = sum(x == y for x, y in zip(pred_labels, test_labels)) / len(test_labels)
    print('accuracy:', accuracy)
    print('ASOM (40x40, lr=.5, %s)' % repr(topology_kwargs))
    print('QE', som.quantization_error(test_data))
    print('TE', som.topology_error(test_data))


def cluster_scores(som, test_data, test_labels):
    pred_labels = som.predict(test_data)
    pred_labels = ['xxx' if label is None else label for label in pred_labels]
    test_labels = ['yyy' if label is None else label for label in test_labels]
    adj_rand = metrics.adjusted_rand_score(test_labels, pred_labels)
    adj_mi = metrics.adjusted_mutual_info_score(test_labels, pred_labels)
    norm_mi = metrics.normalized_mutual_info_score(test_labels, pred_labels)
    acc = metrics.accuracy_score(test_labels, pred_labels)
    f1 = metrics.f1_score(test_labels, pred_labels)
    confusion = metrics.confusion_matrix(test_labels, pred_labels)
    return confusion
    return (acc, f1, adj_rand, adj_mi, norm_mi)


def find_best_config():
    # dim, init_learning_rate, toroidal, normalize,
    #params = [(15, 40), (.05, .75), False, False]
    #try:
        #pass
    #except (KeyError, QHullError, OverflowError):
        #pass
    pass


def test_colors():
    data = [(random.random(), random.random(), random.random())
            for x in range(1000)]
    test_data = [(random.random(), random.random(), random.random())
                 for x in range(1000)]
    # You can try it with some assymetric data:
    # for i in range(100):
    #     _, g, b = data[i]
    #     data[i] = 1, g, b

    # Try with a normal SOM
    som = BasicSOM(data, 40, 40, init_variance=10)
    som.train(iterations=1000)
    #som.color_plot()
    #som.hit_map()
    #som.distance_map()
    print('Measures (basic, 40, 40)')
    print('QE', som.quantization_error(test_data))
    print('TE', som.topology_error(test_data))

    # Test with an assymetric SOM
    topology_kwargs = {
        'toroidal': False,
        'push': .12,
        #'push': 0,
        'inhibition': 15,
        #'inhibition': 200,
    }
    som = GridlessSOM(data, 40, 40, init_learning_rate=.5,
                      topology_kwargs=topology_kwargs)
    som.train(iterations=1000)
    #som.train(iterations=1)
    som.color_plot()
    print('ASOM (40x40, lr=.5, %s)' % repr(topology_kwargs))
    print('QE', som.quantization_error(test_data))
    print('TE', som.topology_error(test_data))


if __name__ == '__main__':
    #test_colors()
    data, labels = read_bone_marrow_data(5000)
    test_bone_marrow(data, labels)

#!/usr/bin/env python
# encoding: utf-8
import random
import csv
from collections import Counter
from itertools import chain, islice
from math import exp

import matplotlib.pyplot as plt


class SOM():
    def __init__(self, data, width=10, height=10, init_variance=None,
                 init_learning_rate=0.1, labels=None):
        self.width, self.height = width, height
        self.data = data
        self.codebook = [[Node(i, j, len(data[0]))
                          for j in range(width)]
                         for i in range(height)]
        if init_variance is None:
            init_variance = ((width ** 2 + height ** 2) / 2) ** 0.5
        self.init_variance = init_variance
        self.init_learning_rate = init_learning_rate
        self.labels = labels

    def train(self, iterations=100):
        for iteration in range(iterations):
            (input_index, input_vector) = self._random_vector()
            bmu = self.bmu(input_index, input_vector)
            self.update_neighbours(iteration, iterations, input_vector, bmu)

    def bmu(self, input_index, input_vector):
        # Find codebook vector with smallest distance
        winner = min(chain(*self.codebook),
                     key=lambda x: x.distance_sq(input_vector))
        # Count as a hit for the chosen vector (optionally with a label)
        if self.labels is not None:
            winner.hit(self.labels[input_index])
        else:
            winner.hit()

        return winner

    def update_neighbours(self, iteration, iterations, input_vector, bmu):
        t = iteration / iterations

        # Linear
        #learning_rate = self.init_learning_rate * (1 - t)
        learning_rate = self.init_learning_rate / (1 + t)
        #learning_rate = self.init_learning_rate * exp(-t)
        # Exponential
        #learning_rate = self.init_learning_rate * \
        #    (.005 / self.init_learning_rate) ** t

        #variance = self.init_variance * (1 - t)
        variance = self.init_variance / (1 + t)
        #variance = self.init_variance ** (-t + 1)
        #variance = self.init_variance * (.001 / self.init_variance) ** t

        for node in chain(*self.codebook):
            neighbourhood = gaussian(node, bmu, variance)
            node.update(learning_rate * neighbourhood, input_vector)

    def predict(self, test_vector):
        # TODO
        pass

    def __repr__(self):
        return repr(self.codebook)

    def color_plot(self):
        """Shows a representation of the SOM where every codebook vector is
        represented as a color.  Of course, this only works for 3- or
        4-dimensional data."""
        assert 3 <= len(self.codebook[0][0].vector) <= 4

        values = [[x.vector for x in row] for row in self.codebook]
        plt.imshow(values, interpolation='none')
        plt.title('Color plot')
        plt.show()

    def label_plot(self):
        # TODO
        assert self.labels is not None

        #plt.imshow(values, interpolation='none')
        #plt.text(2, 5, 'lolcatz')
        #plt.title('Label plot')
        #plt.show()

    def hit_map(self):
        """Shows a heatmap of the codebook where the heat represents how many
        times the nodes were chosen as a BMU."""

        hits = [[node.get_hits() for node in row] for row in self.codebook]
        plt.imshow(hits, interpolation='none')
        plt.title('Hit map')
        plt.show()

    def distance_map(self):
        """Shows a plot of how far the vector of a node is from its
        neighbours. A darker color means it's further away."""

        distances = []
        for row, nodes in enumerate(self.codebook):
            for col, node in enumerate(nodes):
                node_distances = []
                for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (row + drow < 0 or row + drow > len(self.codebook) or
                            col + dcol < 0 or col + dcol > len(nodes)):
                        continue
                    neighbour = self.codebook[row + drow][col + dcol].vector
                    node_distances.append(node.distance(neighbour))
                distances[row][col] = sum(node_distances) / len(node_distances)

        plt.imshow(distances, interpolation='none')
        plt.title('Distance map / U-matrix')
        plt.show()

    def _random_vector(self):
        index = random.randrange(len(self.data))
        return (index, self.data[index])


class Node():
    def __init__(self, row, col, codebook_vector_size):
        self.row, self.col = row, col
        self.vector = [0 for _ in range(codebook_vector_size)]
        # labels is a frequency table for the labels of the vectors of all hits
        self.labels = Counter()

    def distance(self, other_vector):
        return self.distance_sq(other_vector) ** 0.5

    def distance_sq(self, other_vector):
        return sum((x - y) ** 2 for x, y in zip(self.vector, other_vector))

    def update(self, factor, input_vector):
        self.vector = [x + factor * (y - x)
                       for x, y in zip(self.vector, input_vector)]

    def hit(self, label=None):
        self.labels[label] += 1

    def get_hits(self):
        return sum(self.labels.values())

    def get_label(self):
        # TODO might not be the best label
        return self.labels.most_common()[0]

    def __repr__(self):
        return '%d,%d: %s (%s)' % (self.row, self.col, self.get_label(),
                                   self.vector)


def gaussian(node1, node2, variance):
    dist_sq = (node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2
    return exp(-dist_sq / (2 * variance * variance))


def test_bone_marrow():
    with open('data/BoneMarrow_Basal1.csv') as file_:
        data = [list(map(float, x)) for x in islice(csv.reader(file_), 1, None)]
    with open('./data/manualGating.csv') as file_:
        labels = list(islice(csv.reader(file_), 1, None))

    som = SOM(data, width=20, height=20, init_variance=7,
              init_learning_rate=.5, labels=labels)
    som.train(iterations=1)
    som.label_plot()


def test_colors():
    random.seed(2)
    data = [(random.random(), random.random(), random.random())
            for x in range(1000)]
    som = SOM(data, width=20, height=20, init_variance=10,
              init_learning_rate=.1)
    som.train(iterations=500)
    som.color_plot()
    #som.hit_map()
    som.distance_map()


if __name__ == '__main__':
    #test_bone_marrow()
    test_colors()

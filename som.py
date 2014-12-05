#!/usr/bin/env python
# encoding: utf-8
import random
import csv
from collections import Counter
from itertools import chain, islice
from math import exp

import matplotlib.pyplot as plt


class SOM():
    def __init__(self, data, width, height, init_variance=None,
                 init_learning_rate=0.1, labels=None):
        """Params:
          - data should be a list of numerical vectors
          - width and height should be the dimension of the map
          - init_variance (optional) should be the initial variance of the
            Gaussian distribution of the neighbourhood function
          - init_learning_rate (optional) should be the initial learning rate
          - labels (optional) should be the class labels for the data if these
            exist. We can give nice plots in this case.
        """
        self.width, self.height = width, height
        self.data = data
        # Initialize the codebook with a specified raster of zero-vectors
        self.codebook = [[Node(i, j, len(data[0]))
                          for j in range(width)]
                         for i in range(height)]
        # If no initial variance is given, make a guess based on the dimensions
        if init_variance is None:
            init_variance = ((width ** 2 + height ** 2) / 2) ** 0.5
        self.init_variance = init_variance
        self.init_learning_rate = init_learning_rate
        self.labels = labels

    def train(self, iterations=500):
        """Trains the SOM."""
        for iteration in range(iterations):
            # Select random data vector
            (input_index, input_vector) = self._random_vector()
            # Find the best matching unit in the codebook
            bmu = self.bmu(input_vector)
            # Count as a hit for the chosen vector (optionally with a label)
            if self.labels is not None:
                bmu.hit(self.labels[input_index])
            else:
                bmu.hit()
            # Update the neighbours of the bmu accordingly
            self.update_neighbours(iteration, iterations, input_vector, bmu)

    def bmu(self, input_vector):
        """Selects the best matching unit for an input vector."""
        return min(chain(*self.codebook),
                   key=lambda x: x.distance_sq(input_vector))

    def update_neighbours(self, iteration, iterations, input_vector, bmu):
        """Updates the neighbours of the bmu according to the new input."""

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
        """Predicts a test vector according to the class label of the bmu."""
        # TODO
        return self.bmu(test_vector).label()

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
        """If there are class labels available for the data, we plot the
        SOM with labels on the nodes where this class is the most frequent."""
        assert self.labels is not None

        gradient = [[[row / self.height, col / self.width, 0]
                     for col in range(self.width)]
                    for row in range(self.height)]
        plt.imshow(gradient, interpolation='none')

        for label in set(self.labels) - set([None]):
            class_node = max(chain(*self.codebook),
                             key=lambda node: node.labels[label])
            print(class_node.row, class_node.col)
            print(label)
            plt.text(class_node.row, class_node.col, label,
                     horizontalalignment='center', verticalalignment='center')

        plt.title('Label plot')
        plt.show()

    def polar_plots(self):
        """Shows for each node the attributes of the codebook vector as a polar
        plot."""

        import numpy as np
        fig, axes = plt.subplots(self.height, self.width,
                                 subplot_kw={'polar': True})
        normalized_codebook = normalize(x.vector for x in chain(*self.codebook))

        for ax, codebook_vector in zip(chain(*axes), normalized_codebook):
            n = len(codebook_vector)
            thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
            radii = codebook_vector
            bars = ax.bar(thetas, radii, width=2 * np.pi / n)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            for i, bar in enumerate(bars):
                bar.set_facecolor(plt.cm.jet(i / n))
                bar.set_alpha(0.5)

        fig.suptitle('Polar plots')
        plt.show()

    def hit_map(self):
        """Shows a heatmap of the codebook where the heat represents how many
        times the nodes were chosen as a BMU."""

        hits = [[node.get_hits() for node in row] for row in self.codebook]
        plt.imshow(hits, interpolation='none')
        plt.title('Hit map')
        plt.show()

    def distance_map(self):
        """Shows a plot of how far the vector of a node is from its
        neighbours. A warmer color means it's further away."""

        distances = []
        for row, nodes in enumerate(self.codebook):
            distance_row = []
            for col, node in enumerate(nodes):
                node_distances = []
                for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (row + drow < 0 or row + drow >= len(self.codebook) or
                            col + dcol < 0 or col + dcol >= len(nodes)):
                        continue
                    neighbour = self.codebook[row + drow][col + dcol].vector
                    node_distances.append(node.distance(neighbour))
                distance_row.append(sum(node_distances) / len(node_distances))
            distances.append(distance_row)

        plt.imshow(distances, interpolation='none')
        plt.title('Distance map / U-matrix')
        plt.show()

    def _random_vector(self):
        """Selects a random data vector."""
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
        most_common = self.labels.most_common(1)
        return most_common[0] if len(most_common) > 0 else None

    def __repr__(self):
        return '%d,%d: %s (%s)' % (self.row, self.col, self.get_label(),
                                   self.vector)


def gaussian(node1, node2, variance):
    dist_sq = (node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2
    return exp(-dist_sq / (2 * variance * variance))


def normalize(vectors):
    vectors = list(vectors)
    mins = [min(x) for x in zip(*vectors)]
    maxs = [max(x) for x in zip(*vectors)]
    for vector in vectors:
        yield [(number - min_) / (max_ - min_)
               for min_, max_, number in zip(mins, maxs, vector)]


def test_bone_marrow():
    with open('data/BoneMarrow_Basal1.csv') as file_:
        data = [list(map(float, x)) for x in islice(csv.reader(file_), 1, None)]
    with open('./data/manualGating.csv') as file_:
        labels = (x[1] for x in islice(csv.reader(file_), 1, None))
        labels = [None if x == 'Unknown' else x for x in labels]

    som = SOM(data, width=5, height=5, init_variance=7,
              init_learning_rate=.5, labels=labels)
    #som = SOM([[1]], width=20, height=20, init_variance=7,
    #som = SOM([[1]], width=5, height=5, init_variance=7,
              #init_learning_rate=.5, labels=[5])
    som.train(iterations=500)
    #som.label_plot()
    som.polar_plots()


def test_colors():
    random.seed(2)
    data = [(random.random(), random.random(), random.random())
            for x in range(1000)]
    som = SOM(data, width=20, height=20, init_variance=10,
              init_learning_rate=.1)
    som.train(iterations=500)
    som.color_plot()
    #som.hit_map()
    #som.distance_map()


if __name__ == '__main__':
    test_bone_marrow()
    #test_colors()

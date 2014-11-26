#!/usr/bin/env python
# encoding: utf-8
import random
import csv
from itertools import chain, islice
from math import exp

import matplotlib.pyplot as plt


class SOM():
    def __init__(self, data, width=10, height=10, init_variance=None,
                 init_learning_rate=0.1, labels=None):
        self.width, self.height = width, height
        self.data = data
        self.codebook = [[Node(i, j, self._random_vector())
                          for j in range(width)]
                         for i in range(height)]
        if init_variance is None:
            init_variance = ((width ** 2 + height ** 2) / 2) ** 0.5
        self.init_variance = init_variance
        self.init_learning_rate = init_learning_rate
        self.labels = labels

    def train(self, iterations=100):
        for iteration in range(iterations):
            input_vector = self._random_vector()
            bmu = self.bmu(input_vector)
            self.update_neighbours(iteration, iterations, input_vector, bmu)
            if iteration % 50 == 0:
                self.graphical_repr()

    def bmu(self, input_vector):
        winner = min(chain(*self.codebook),
                     key=lambda x: x.distance_sq(input_vector))
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

    def graphical_repr(self):
        #plt.figure(figsize=(5, 5))
        #min_ = np.mean(codebook.flatten())-1*np.std(codebook.flatten()),
        #max_ = np.mean(codebook.flatten())+1*np.std(codebook.flatten()),
        #norm = matplotlib.colors.normalize(vmin=min_, vmax=max_, clip=True)
        #plt.pcolor(codebookreshape, norm=norm)
        #plt.axis('off')
        #plt.axis([0, n, 0, m])
        values = [[x.vector for x in row] for row in self.codebook]
        plt.imshow(values, interpolation='none')
        #plt.text(2, 5, 'lolcatz')
        plt.show()

    def hit_map(self):
        #TODO
        pass

    def _random_vector(self):
        #random =
        return random.choice(self.data)


class Node():
    def __init__(self, row, col, vector):
        self.row, self.col = row, col
        self.vector = vector
        self.hits = 0

    def distance_sq(self, other_vector):
        return sum((x - y) ** 2 for x, y in zip(self.vector, other_vector))

    def update(self, factor, input_vector):
        self.vector = [x + factor * (y - x)
                       for x, y in zip(self.vector, input_vector)]
        norm = sum(self.vector) ** .5
        self.vector = [x / norm for x in self.vector]

    def hit(self):
        self.hits += 1

    def __repr__(self):
        return '%d,%d: %s' % (self.row, self.col, self.vector)


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
    som.train(iterations=200)

    #som.graphical_repr()


def test_colors():
    random.seed(2)
    data = [(random.random(), random.random(), random.random())
            for x in range(1000)]
    som = SOM(data, width=20, height=20, init_variance=6,
              init_learning_rate=.1)
    som.train(iterations=500)
    som.graphical_repr()


if __name__ == '__main__':
    #test_bone_marrow()
    test_colors()

#!/usr/bin/env python
# encoding: utf-8

"""A regular SOM."""

from collections import UserList
from .som import normalize
from .som import SOM, Topology, Node
from itertools import chain, islice
from random import choice
from math import exp

import numpy as np
import matplotlib.pyplot as plt


class BasicSOM(SOM):
    """A regular SOM."""

    def __init__(self, data, width, height, neighbourhood=None,
                 init_variance=None, **kwargs):
        """Initializes a new BasicSOM object.
        :data: should be a list of numerical vectors
        :width and :height: should be the dimension of the map
        :neighbourhood: (optional) should be a function which
            decides how much influence a bmu has on a specifed node at a
            certain moment in the training stage.
        :init_variance: (optional) should be the initial variance of the
            Gaussian distribution of the neighbourhood function (if no other
            neighbourhood function is given of course)
        """

        codebook = Grid(data, width, height, init_variance)
        super().__init__(data, codebook, **kwargs)

    def color_plot(self):
        """Shows a representation of the BasicSOM where every codebook vector
        is represented as a color.  Of course, this only works for 3- or
        4-dimensional data.
        """
        assert 3 <= self.data_vector_size <= 4

        values = [[x.vector for x in row] for row in self.codebook.data]
        plt.imshow(values, interpolation='none')
        plt.title('Color plot')
        plt.show()

    def label_plot(self):
        """If there are class labels available for the data, we plot the
        SOM with labels on the nodes where this class is the most frequent.
        """
        assert self.labels is not None

        normalized_codebook = normalize(node.vector for node in self.codebook)
        raster = split_generator(normalized_codebook, self.codebook.width)
        three_feats = [[vector[:3] + [.7] for vector in row] for row in raster]
        plt.imshow(three_feats, interpolation='none')

        for label in set(self.labels) - set([None]):
            class_node = max(self.codebook, key=lambda node: node.labels[label])
            plt.text(class_node.row, class_node.col, label,
                     horizontalalignment='center', verticalalignment='center')

        plt.title('Label plot')
        plt.show()

    def polar_plots(self):
        """Shows for each node the attributes of the codebook vector as a polar
        plot.
        """
        fig, axes = plt.subplots(self.codebook.height, self.codebook.width,
                                 subplot_kw={'polar': True})
        normalized_codebook = normalize(x.vector for x in self.codebook)

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
        times the nodes were chosen as a BMU.
        """
        hits = [[node.hits for node in row] for row in self.codebook.data]
        plt.imshow(hits, interpolation='none')
        plt.title('Hit map')
        plt.show()

    def distance_map(self):
        """Shows a plot of how far the vector of a node is from its
        neighbours. A warmer color means it's further away.
        """
        distances = []
        for row, nodes in enumerate(self.codebook.data):
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


class Grid(Topology, UserList):
    """A grid topology which is just a wrapper around a 2D list, satisfying the
    Topology interface.
    """

    def __init__(self, data, width, height, init_variance=None):
        """Initializes the grid."""

        self.width, self.height = width, height
        # If no initial variance is given, make a guess based on the dimensions
        if init_variance is None:
            init_variance = ((width ** 2 + height ** 2) / 2) ** 0.5
        self.init_variance = init_variance
        real_list = [[GridNode(i, j, choice(data)) for j in range(width)]
                     for i in range(height)]
        super().__init__(real_list)

    def __iter__(self):
        """Returns an iterator which flattens the 2D array to one dimension."""
        return chain(*self.data)

    def neighbourhood(self, node1, node2, t):
        """Returns the neighbourhood influence of node1 over node2 at
        iteration t. This uses a gaussian distribution.
        """
        return self._gaussian(node1, node2, t)
        # M-SOM
        # return max(0, 1 - self.distance_squared(node1, node2))

    def _gaussian(self, node1, node2, t):
        """Calculates a neighbourhood value following a Gaussian distribution.
        This assumes the nodes are GridNodes.
        """
        dist_sq = self.distance_squared(node1, node2)
        variance = self._gaussian_variance(t)
        return exp(-dist_sq / (2 * variance * variance))

    def _gaussian_variance(self, t):
        """A decreasing function for the variance of the gaussian distribution.
        """
        # return self.init_variance * (1 - t)
        return self.init_variance / (1 + t)
        # return self.init_variance ** (-t + 1)
        # return self.init_variance * (.001 / self.init_variance) ** t

    def distance_squared(self, node1, node2):
        """Calculates the squared distance between two nodes on the grid."""
        return (node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2

    def are_neighbours(self, node1, node2):
        """Checks whether two nodes are neighbouring on the grid."""
        return ((node1.row == node2.row and abs(node1.col - node2.col) <= 1)
                or (node1.col == node2.col and abs(node1.row - node2.row) <= 1))


class ToroidalGrid(Grid):
    """An extension of the regular grid using a torus."""

    def distance_squared(self, node1, node2):
        """Calculates the squared distance between two nodes on the torus."""
        dr = abs(node1.row - node2.row)
        dc = abs(node1.col - node2.col)
        return (min(dr, abs(dr - self.height)) ** 2
                + min(dc, abs(dc - self.width)) ** 2)


class GridNode(Node):
    """A node in the grid."""

    def __init__(self, row, col, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row, self.col = row, col

    def __repr__(self):
        """Representation: 'row,col: label (vector)'"""
        return '%d,%d: %s (%s)' % (self.row, self.col, self.get_label(),
                                   self.vector)


def split_generator(generator, n):
    """A helper function which splits a generator into multiple generators,
    cutting it off each time after n elements.
    """
    while True:
        part = list(islice(generator, n))
        if len(part) > 0:
            yield part
        if len(part) < n:
            break

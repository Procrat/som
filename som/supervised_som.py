#!/usr/bin/env python
# encoding: utf-8

from .som import SOM, Topology, Node
from random import choice
from collections import UserList
from itertools import product
import matplotlib.pyplot as plt
from math import exp


class GridlessSOM(SOM):
    def __init__(self, data, width, height, init_variance=.5, toroidal=False,
                 **kwargs):
        """Initializes a new GridlessSOM object.
        :data: should be a list of numerical vectors
        :width and :height: should be the dimensions of the initial grid
        :init_variance: the initial variance of the gaussian distribution of
            the neighbourhood function.
        :toroidal: whether to use a torus or a plane

        For other parameters, check the SOM documentation.
        """

        codebook_class = Torus if toroidal else Plane
        codebook = codebook_class(data, width, height, init_variance)
        super().__init__(data, codebook, **kwargs)

    def predict(self, test_vector):
        """Predicts a test vector according to the class label of the bmu."""
        # TODO
        pass

    def voronoi_plot(self):
        """Shows a representation of the SOM where the location of the nodes
        are marked and a Voronoi tesselation is made with this points."""

        from scipy.spatial import Voronoi, voronoi_plot_2d

        centroids = [node.location() for node in self.codebook]
        diagram = Voronoi(centroids)
        voronoi_plot_2d(diagram)

        for node in self.codebook:
            plt.text(node.x, node.y, '%.1f,%.1f,%.1f' % tuple(node.vector),
                     horizontalalignment='center', verticalalignment='center')

        for label, pos in (('red', [1,0,0]), ('green', [0,1,0]), ('blue', [0,0,1])):
            node = self.bmu(pos)
            plt.text(node.x, node.y, label,
                     horizontalalignment='center', verticalalignment='center',
                     color=label)

        plt.title('Voronoi plot')
        plt.show()

    def label_plot(self):
        """Some as voronoi plot, but if there are class labels available for
        the data, we plot the SOM with labels on the nodes where this class is
        the most frequent."""
        assert self.labels is not None

        from scipy.spatial import Voronoi, voronoi_plot_2d

        centroids = [node.location() for node in self.codebook]
        diagram = Voronoi(centroids)
        voronoi_plot_2d(diagram)

        for label in set(self.labels) - set([None]):
            class_node = max(self.codebook, key=lambda node: node.labels[label])
            plt.text(class_node.x, class_node.y, label,
                     horizontalalignment='center', verticalalignment='center')

        plt.title('Voronoi label plot')
        plt.show()

    def polar_plots(self):
        """Shows for each node the attributes of the codebook vector as a polar
        plot."""
        # TODO

        #import numpy as np

        #fig, axes = plt.subplots(self.height, self.width,
                                 #subplot_kw={'polar': True})
        #nolized_codebook = normalize(x.vector for x in chain(*self.codebook))

        #for ax, codebook_vector in zip(chain(*axes), normalized_codebook):
            #n = len(codebook_vector)
            #thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
            #radii = codebook_vector
            #bars = ax.bar(thetas, radii, width=2 * np.pi / n)

            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            #for i, bar in enumerate(bars):
                #bar.set_facecolor(plt.cm.jet(i / n))
                #bar.set_alpha(0.5)

        #fig.suptitle('Polar plots')
        #plt.show()

    def hit_map(self):
        """Shows a heatmap of the codebook where the heat represents how many
        times the nodes were chosen as a BMU."""

        # TODO
        #hits = [[node.hits for node in row] for row in self.codebook]
        #plt.imshow(hits, interpolation='none')
        #plt.title('Hit map')
        #plt.show()

    def distance_map(self):
        """Shows a plot of how far the vector of a node is from its
        neighbours. A warmer color means it's further away."""

        # TODO
        #distances = []
        #for row, nodes in enumerate(self.codebook):
            #distance_row = []
            #for col, node in enumerate(nodes):
                #node_distances = []
                #for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    #if (row + drow < 0 or row + drow >= len(self.codebook) or
                            #col + dcol < 0 or col + dcol >= len(nodes)):
                        #continue
                    #neighbour = self.codebook[row + drow][col + dcol].vector
                    #node_distances.append(node.distance(neighbour))
                #distance_row.append(sum(node_distances) / len(node_distances))
            #distances.append(distance_row)

        #plt.imshow(distances, interpolation='none')
        #plt.title('Distance map / U-matrix')
        #plt.show()


class PlaneNode(Node):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x, self.y = x, y

    def update(self, factor, input_vector, bmu):
        super().update(factor, input_vector, bmu)

        #vector_d = self.distance(bmu.vector)
        #if vector_d > 0:
            #factor /= vector_d
        print(factor)
        #factor -= .3
        self.x = self.x + factor * (bmu.x - self.x)
        self.y = self.y + factor * (bmu.y - self.y)

    def location(self):
        return (self.x, self.y)

    def __repr__(self):
        return '%f,%f: %s (%s)' % (self.x, self.y, self.get_label(),
                                   self.vector)


class Plane(Topology, UserList):
    """A 2D topology where the coordinates of the nodes are not bound to a
    grid."""
    NODE_CLASS = PlaneNode

    def __init__(self, data, width, height, init_variance):
        self.init_variance = init_variance
        real_list = [self.NODE_CLASS(i / width, j / height, choice(data))
                     for i, j in product(range(width), range(height))]
        super().__init__(real_list)

    def __iter__(self):
        return iter(self.data)

    def neighbourhood(self, node1, node2, t):
        return self._gaussian(node1, node2, t)

    def _gaussian(self, node1, node2, t):
        variance = self.init_variance / (1 + t)
        #variance = self.init_variance * (1 - t)
        #variance = self.init_variance ** (-t + 1)
        #variance = self.init_variance * (.001 / self.init_variance) ** t

        dist_sq = self.plane_distance_squared(node1, node2)
        return exp(-dist_sq / (2 * variance * variance))

    def plane_distance_squared(self, node1, node2):
        return (node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2


class TorusNode(PlaneNode):
    def update(self, factor, input_vector, bmu):
        super().update(factor, input_vector, bmu)

        vector_d = self.distance(bmu.vector)
        if vector_d > 0:
            factor /= vector_d
        self.x = self.lin_comb(self.x, bmu.x, factor)
        self.y = self.lin_comb(self.y, bmu.y, factor)

    @staticmethod
    def lin_comb(a, b, factor):
        if abs(a - b) > abs(a - b + 1):
            if a < b:
                a += 1
            else:
                b += 1
        c = (1 - factor) * a + factor * b
        if c >= 1:
            c -= 1
        return c


class Torus(Plane):
    NODE_CLASS = TorusNode

    def distance_squared(self, node1, node2):
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        return min(dx, abs(dx - 1)) ** 2 + min(dy, abs(dy - 1)) ** 2

#!/usr/bin/env python
# encoding: utf-8

"""An asymmetric SOM. This type of SOM doesn't use a grid, but the nodes are
freely positioned on a plane.
"""

from collections import UserList
from math import exp
import random
from random import choice
from scipy.spatial import Voronoi, voronoi_plot_2d
from .som import SOM, Topology, Node, normalize
import numpy as np
import matplotlib.pyplot as plt


class GridlessSOM(SOM):

    def __init__(self, data, width, height, topology_kwargs={}, **kwargs):
        """Initializes a new GridlessSOM object.
        :data: should be a list of numerical vectors
        :width and :height: should be the dimensions of the initial grid
        :init_variance: the initial variance of the gaussian distribution of
            the neighbourhood function.
        :toroidal: whether to use a torus or a plane

        For other parameters, check the SOM documentation.
        """
        toroidal = topology_kwargs.pop('toroidal', False)
        codebook_class = Torus if toroidal else Plane
        codebook = codebook_class(data, width, height, **topology_kwargs)
        super().__init__(data, codebook, **kwargs)

    def voronoi_plot(self):
        """Shows a representation of the SOM where the location of the nodes
        are marked and a Voronoi tesselation is made with this points."""

        centroids, voronoi = self.codebook.voronoi
        voronoi_plot_2d(voronoi)

        for node in self.codebook:
            plt.text(node.x, node.y, '%.1f,%.1f,%.1f' % tuple(node.vector),
                     horizontalalignment='center', verticalalignment='center')

        plt.title('Voronoi plot')
        plt.show()

    def color_plot(self):
        """Same as voronoi_plot, but assuming that the data is 3-dimensional,
        gives the regions the color corresponding to the weight vectors.
        """
        assert self.data_vector_size == 3

        centroids, vor = self.codebook.voronoi
        regions, vertices = voronoi_finite_polygons(vor)
        for node, region in zip(self.codebook, regions):
            polygon = vertices[region]
            plt.fill(*zip(*polygon), color=node.vector)

        plt.plot([x[0] for x in centroids], [x[1] for x in centroids], 'ko')
        plt.axis('equal')
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

        plt.title('Color plot')
        plt.show()

    def label_plot(self):
        """Some as voronoi plot, but if there are class labels available for
        the data, we plot the SOM with labels on the nodes where this class is
        the most frequent.
        """
        assert self.labels is not None

        centroids, vor = self.codebook.voronoi
        regions, vertices = voronoi_finite_polygons(vor)
        normalized_codebook = normalize(node.vector for node in self.codebook)
        for codebook_vector, region in zip(normalized_codebook, regions):
            polygon = vertices[region]
            plt.fill(*zip(*polygon), color=codebook_vector[:3] + [.6])

        xs, ys = zip(*centroids)
        plt.plot(xs, ys, 'ko', ms=1)
        plt.axis('equal')
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

        for label in set(self.labels) - set([None]):
            class_node = max(self.codebook, key=lambda node: node.labels[label])
            plt.text(class_node.x, class_node.y, label,
                     horizontalalignment='center', verticalalignment='center')

        plt.title('Voronoi label plot')
        plt.show()


class PlaneNode(Node):
    """A node on the plane."""

    def __init__(self, x, y, vector, push=.2, inhibition=15, **kwargs):
        super().__init__(vector, **kwargs)
        self.x, self.y = x, y
        self.push = push
        self.inhibition = inhibition

    def update(self, learning_rate, influence, input_vector, bmu):
        """The update rule is extended to also influence the position of the
        node on the plane. We only want this after a certain while though.
        """
        super().update(learning_rate, influence, input_vector, bmu)

        if learning_rate < .33:
            self.update_position(learning_rate, influence, bmu)

    def update_location(self, learning_rate, influence, bmu):
        """Updates the location of the node on the plane."""
        factor = learning_rate * (influence - self.push) / self.inhibition
        self.x = self.x + factor * (bmu.x - self.x)
        self.y = self.y + factor * (bmu.y - self.y)

    def location(self):
        """Returns the coordinates of the node on the plane as a tuple."""
        return (self.x, self.y)

    def __repr__(self):
        """Representation: 'x,y (vector)'"""
        return '%f,%f (%s)' % (self.x, self.y, self.vector)


class Plane(Topology, UserList):
    """A 2D topology where the coordinates of the nodes are not bound to a
    grid.
    """
    NODE_CLASS = PlaneNode

    def __init__(self, data, width, height, init_variance=.5, **kwargs):
        """`width * height` nodes are generated."""
        def new_node(x, y):
            return self.NODE_CLASS(x, y, choice(data), **kwargs)
        real_list = list(self._generate_nodes(width * height, new_node))
        super().__init__(real_list)
        self.init_variance = init_variance
        self._voronoi = None

    def _generate_nodes(self, n, new_node):
        """Generates nodes randomly distributed in a circle with radius .5
        around (.5, .5).
        """
        i = 0
        while i < n:
            x, y = random.random(), random.random()
            if (x - .5) ** 2 + (y - .5) ** 2 < .5 ** 2:
                yield new_node(x, y)
                i += 1

    def __iter__(self):
        return iter(self.data)

    def neighbourhood(self, node1, node2, t):
        """Calculates the neighbourhood influence using a Gaussian
        distribution.
        """
        return self._gaussian(node1, node2, t)

    def _gaussian(self, node1, node2, t):
        """Calculates the neighbourhood influence using a Gaussian
        distribution.
        """
        dist_sq = self.plane_distance_squared(node1, node2)
        variance = self._gaussian_variance(t)
        return exp(-dist_sq / (2 * variance * variance))

    def _gaussian_variance(self, t):
        """A decreasing function for the variance of the gaussian distribution.
        """
        # return self.init_variance * (1 - t)
        return self.init_variance / (1 + t)
        # return self.init_variance ** (-t + 1)
        # return self.init_variance * (.001 / self.init_variance) ** t

    def plane_distance_squared(self, node1, node2):
        """Calculates the distance between two nodes on the plane."""
        return (node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2

    @property
    def voronoi(self):
        """Wrapper around self._voronoi to make sure we don't calculate it
        multiple times unnecessarily.
        """
        if self._voronoi is None:
            self._calculate_voronoi()
        return self._voronoi[:2]

    def _calculate_voronoi(self):
        """Calculates the Voronoi tesselation of the nodes."""
        centroids = [node.location() for node in self]
        vor = Voronoi(centroids)
        self._voronoi = (centroids, vor, voronoi_finite_polygons(vor))

    def are_neighbours(self, node1, node2):
        """Checks whether two nodes are neighbouring in the Voronoi
        tesselation.
        """
        # Calculate the finite regions of the nodes in the voronoi tesselation
        self._calculate_voronoi()
        regions, _ = self._voronoi[2]
        nodes = list(self)
        region1 = regions[nodes.index(node1)]
        region2 = regions[nodes.index(node2)]

        # Check if regions have borders in common
        def ridges(region):
            n = len(region)
            ridges = {(region[i], region[(i + 1) % n]) for i in range(n)}
            return ridges | {(y, x) for x, y in ridges}
        ridges_in_common = ridges(region1) & ridges(region2)
        return len(ridges_in_common) > 0


class TorusNode(PlaneNode):
    """A Node implementation for the Torus topology."""

    def update_location(self, learning_rate, influence, bmu):
        """Updates the location of the node on the torus."""
        factor = learning_rate * (influence - self.push) / self.inhibition
        self.x = self.lin_comb(self.x, bmu.x, factor)
        self.y = self.lin_comb(self.y, bmu.y, factor)

    @staticmethod
    def lin_comb(a, b, factor):
        if b < a:
            a, b = b, a
            factor = 1 - factor
        if abs(a - b) > abs(a + 1 - b):
            a += 1
        c = (1 - factor) * a + factor * b
        if c >= 1:
            c -= 1
        return c


class Torus(Plane):
    """An extension of the plane using a torus."""

    NODE_CLASS = TorusNode

    def plane_distance_squared(self, node1, node2):
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        return min(dx, abs(dx - 1)) ** 2 + min(dy, abs(dy - 1)) ** 2


# Adapted from common code of Pauli Virtanen
# (https://gist.github.com/pv/8036995)
def voronoi_finite_polygons(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    :vor: Voronoi diagram
    :radius: (optional) distance to 'points at infinity'
    Returns
    :regions: Indices of vertices in each revised Voronoi regions
    :vertices: Coordinates for revised Voronoi vertices. Same as coordinates of
               input vertices, with 'points at infinity' appended to the end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # Finite region
            new_regions.append(vertices)
            continue

        # Reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # Finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # Finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

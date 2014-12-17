#!/usr/bin/env python
# encoding: utf-8

"""The base classes and functions of the framework."""

import collections
from collections import Counter
import random


class SOM:
    """A base class for SOMs. It can be directly instantiated when provided
    with a codebook.
    """

    def __init__(self, data, codebook, init_learning_rate=0.1, labels=None):
        """Initializes a new SOM object.
        :data: should be a list of numerical vectors
        :codebook: should be an initial collections of codebook nodes,
            satisfying the Topology interface.
        :init_learning_rate: (optional) should be the initial learning rate
        :labels: (optional) should be the class labels for the data if these
            exist. We can give nice plots in this case.
        """
        self.data = data
        assert len(data) > 0
        self.data_vector_size = len(data[0])
        self.codebook = codebook
        self.init_learning_rate = init_learning_rate
        self.labels = labels
        self.ready_for_prediction = False

    def train(self, iterations=500):
        """Trains the SOM."""

        self.ready_for_prediction = False

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
        return min(self.codebook, key=lambda x: x.distance_sq(input_vector))

    def update_neighbours(self, iteration, iterations, input_vector, bmu):
        """Updates the neighbours of the BMU according to the new input."""

        t = iteration / iterations
        learning_rate = self.learning_rate(t)
        for node in self.codebook:
            influence = self.codebook.neighbourhood(node, bmu, t)
            node.update(learning_rate, influence, input_vector, bmu)

    def learning_rate(self, t):
        """Calculates the learning rate value for the given iteration, which is
        a number between 0 and 1.
        """
        # return self.init_learning_rate * (1 - t)
        return self.init_learning_rate / (1 + t)
        # return self.init_learning_rate * exp(-t)
        # return self.init_learning_rate * (.005 / self.init_learning_rate) ** t

    def predict(self, test_vectors):
        """Predicts the class label of a test vector or multiple test vectors,
        according to the class label of the bmu.
        """
        # Calculate the best matching label for each node
        if not self.ready_for_prediction:
            # totals = sum((node.labels for node in self.codebook), Counter())
            for node in self.codebook:
                # # Take into account small clusters. A frequency approach
                # freq_counter = Counter({label: count / totals[label]
                #                    for label, count in node.labels.items()})
                # if len(freq_counter) > 0:
                #     node.label = freq_counter.most_common(1)[0][0]
                # else:
                #     node.label = ''
                # Or ignore small clusters and just aim for accuracy
                if len(node.labels) > 0:
                    node.label = node.labels.most_common(1)[0][0]
                else:
                    node.label = ''
            self.ready_for_prediction = True

        # Return the label of the best matching unit for the given test_vectors
        if isinstance(test_vectors, collections.Iterable):
            return [self.bmu(test_vector).label for test_vector in test_vectors]
        else:
            return self.bmu(test_vectors).label

    def quantization_error(self, test_vectors):
        """Calculates the quantization_error for one or multiple vectors."""

        if not isinstance(test_vectors, collections.Iterable):
            test_vectors = [test_vectors]

        distances = (self.bmu(test_vector).distance(test_vector)
                     for test_vector in test_vectors)
        return sum(distances) / len(test_vectors)

    def topology_error(self, test_vectors):
        """Calculates the topology error, a simple topology preservation
        measure. It calculates for a test_vector or multiple test_vectors
        whether the bmu and the second bmu are neighbouring nodes.
        """
        if not isinstance(test_vectors, collections.Iterable):
            test_vectors = [test_vectors]

        def are_bmus_neighbours(test_vector):
            bmu = self.bmu(test_vector)
            nodes_wo_bmu = (node for node in self.codebook if node is not bmu)
            bmu2 = min(nodes_wo_bmu, key=lambda x: x.distance_sq(test_vector))
            return self.codebook.are_neighbours(bmu, bmu2)

        return (sum(not(are_bmus_neighbours(vec)) for vec in test_vectors) /
                len(test_vectors))

    def __repr__(self):
        return repr(self.codebook)

    def _random_vector(self):
        """Selects a random data vector."""
        index = random.randrange(len(self.data))
        return (index, self.data[index])


class Topology:
    """An abstract base class for codebook classes."""

    def neighbourhood(self, node1, node2, t):
        """The neighbourhood function of the topology. Should return the
        influence of one node over the other at time t."""
        raise NotImplementedError

    def __iter__(self):
        """Should return an iterable over all nodes."""
        raise NotImplementedError


class Node:
    """A base class for codebook nodes."""

    def __init__(self, vector):
        """Initializes a new node with a given vector."""

        self.vector = vector
        # labels is a frequency table for the labels of the vectors of all hits
        self.labels = Counter()

    def distance(self, other_vector):
        """Calculates the distance between the weight vector of this node and
        another one.
        """
        return self.distance_sq(other_vector) ** 0.5

    def distance_sq(self, other_vector):
        """Calculates the squared distance between the weight vector of this
        node and another one.
        """
        return sum((x - y) ** 2 for x, y in zip(self.vector, other_vector))

    def update(self, learning_rate, influence, input_vector, bmu):
        """Updates the weight vector of this node to become more like the input
        vector, according to a certain factor.
        """
        factor = learning_rate * influence
        self.vector = [x + factor * (y - x)
                       for x, y in zip(self.vector, input_vector)]

    def hit(self, label=None):
        """Adds a hit to the frequency counter."""
        self.labels[label] += 1

    @property
    def hits(self):
        """Returns the total amount of hits this node recieved, i.e. the amount
        of times this node was chosen as BMU.
        """
        return sum(self.labels.values())


def normalize(vectors):
    """Normalizes a collection of vectors feature-wise."""

    vectors = list(vectors)
    mins = [min(x) for x in zip(*vectors)]
    maxs = [max(x) for x in zip(*vectors)]
    for vector in vectors:
        yield [(number - min_) / (max_ - min_)
               for min_, max_, number in zip(mins, maxs, vector)]

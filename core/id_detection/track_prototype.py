from __future__ import division
from __future__ import unicode_literals
from builtins import object
from past.utils import old_div
import numpy as np
from numpy.linalg import norm


class TrackPrototype(object):
    def __init__(self, descriptor, std, weight=1):
        self.descriptor = descriptor
        self.weight = weight
        self.std = std

    def update(self, new_prototype):
        alpha = old_div(self.weight, float(self.weight + new_prototype.weight))
        self.descriptor = alpha * self.descriptor + (1 - alpha) * new_prototype.descriptor
        self.weight += new_prototype.weight

    def distance(self, prototype):
        return norm(self.descriptor - prototype.descriptor)

    def distance_and_weight(self, prototype):
        return self.distance(prototype), self.weight

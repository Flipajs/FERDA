from __future__ import unicode_literals
from builtins import object
__author__ = 'fnaiser'

class AnimalGroup(object):
    """
    Animal group is based on "functionality"... example infected, healthy, males, females...
    """
    def __init__(self, name, description='', id=-1):
        self.name = name
        self.description = description
        self.id = id
        self.members = []

    def add_member(self, animal):
        self.members.append(animal)
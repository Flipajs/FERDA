from __future__ import unicode_literals
from builtins import object
__author__ = 'fnaiser'


class AnimalClass(object):
    """
    example of AnimalClass is 'worker', 'queen'. It is based on appearance.
    """
    def __init__(self, name, description='', id=-1):
        self.name = name
        self.description = description
        self.id = id
        self.members = []

    def add_member(self, animal):
        self.members.append(animal)
__author__ = 'fnaiser'

class AnimalGroup():
    """
    Animal group is based on "functionality"... example infected, healthy, males, females...
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.members = []

    def add_member(self, animal):
        self.members.append(animal)
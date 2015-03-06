__author__ = 'fnaiser'


class AnimalClass():
    """
    example of AnimalClass is 'worker', 'queen'. It is based on appearance.
    """
    def __init__(self, name, description='', id=0):
        self.name = name
        self.description = description
        self.id = id
        self.members = []

    def add_member(self, animal):
        self.members.append(animal)
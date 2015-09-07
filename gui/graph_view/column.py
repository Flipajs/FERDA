__author__ = 'Simon Mandlik'

from PyQt4 import Qt, QtCore, QtGui

# MAX_NUM_OBJECTS = 8

class Column():

    def __init__(self, frame, empty=False):

        self.empty = empty
        self.x = 0
        self.frame = frame
        self.objects = []

        if self.empty:
            self.compress_marker = QtGui.QGraphicsTextItem('...')
            self.objects.append(0)

    # def is_full(self):
    #     return True if len(self.objects) > MAX_NUM_OBJECTS else False

    def add_object(self, object, position):
        self.objects[position] = object

    def is_free(self, position=0):
        return True if self.objects[position] is None else False

    def draw(self):
        if self.empty:
            self.compress_marker.setPos()
            self.compress_marker.show()
        else:
            for object in self.objects:
                object.draw()




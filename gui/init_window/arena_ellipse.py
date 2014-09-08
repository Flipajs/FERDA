__author__ = 'flipajs'
from gui.init_window.arena_circle import *


class ArenaEllipse(ArenaCircle):
    def __init__(self):
        super(ArenaEllipse, self).__init__()
        self.b = None

    def add_points(self, c_center, a, b):
        self.c = c_center
        self.a = a
        self.b = b
        self.is_ready = True

    def update_geometry(self):
        cx = self.c.pos().x()
        cy = self.c.pos().y()
        a = math.sqrt((cx-self.a.pos().x())**2 + (cy-self.a.pos().y())**2)
        b = math.sqrt((cx-self.b.pos().x())**2 + (cy-self.b.pos().y())**2)

        self.setRect(cx-a+10, cy-a+10, 2*a, 2*b)

__author__ = 'fnaiser'

import numpy as np

male_names_ = ["Arnold", "Bob", "Cenek", "Dusan", "Emil", "Ferda", "Gustav", "Hugo", "Igor", "Julius", "Kamil",
                 "Ludek", "Milos", "Narcis", "Oliver", "Prokop", "Quido", "Rene", "Servac", "Tadeas", "1", "2",
                 "3", "4", "5", "6", "7", "8", "9", "10"]

colors_ = [(145, 95, 22), (54, 38, 227), (0, 191, 255), (204, 102, 153), (117, 149, 105),
                  (0, 182, 141), (255, 255, 0), (32, 83, 78), (0, 238, 255), (128, 127, 0),
                  (190, 140, 238), (32, 39, 89), (99, 49, 222), (139, 0, 0), (0, 0, 139),
                  (60, 192, 3), (0, 79, 255), (128, 128, 128), (255, 255, 255), (255,160,122),
                  (124,252,0), (102,205,170), (0,206,209), (30,144,255), (238,130,238), (75,0,130),
                  (210,180,140), (240,255,240), (184,134,11), (0,255,127), (220,20,60), (128,0,0)]


class Animal():
    def __init__(self, id=0, name=None, colormark=None, color=None):
        self.id = id
        self.name = male_names_[id]
        if name:
            self.name = name

        if colormark:
            self.color_ = colormark.color_
        else:
            self.color_ = colors_[id]

        # Workaround...
        if color:
            self.color_ = color


        self.init_pos_center_ = None
        self.init_pos_head_ = None

        self.colormark_ = None

    def set_colormark(self, colormark):
        self.colormark_ = colormark

        self.color_ = colormark.color_

    def reset_id(self, id):
        self.id = id

    def set_init_pos(self, center, head):
        self.init_pos_center_ = np.array([center.y(), center.x()])
        self.init_pos_head_ = np.array([head.y(), head.x()])
import random
from PyQt4 import QtGui
from matplotlib import colors

from PyQt4.QtGui import QApplication

from gui.graph_widget import OPACITY


class InfoManager():

    def __init__(self, loader):
        self.clipped = []
        self.loader = loader
        self.last_color = None

    def add(self, item):
        self.clipped.append(item)

    def remove(self, item):
        self.clipped.remove(item)

    def show_all_info(self):
        for item in self.clipped:
            self.show_info(item)
        QApplication.processEvents()

    def show_info(self, item):
        if item not in self.clipped:
            self.clipped.append(item)
        if not item.clipped:
            if self.last_color:
                color = hex2rgb_opacity_tuple(inverted_hex_color_str(self.last_color))
                self.last_color = None
            else:
                self.last_color = random_hex_color_str()
                color = hex2rgb_opacity_tuple(self.last_color)
            item.set_color(color)
        item.show_info(self.loader)

    def hide_info(self, item):
        if item.shown:
            item.hide_info()

    def hide_all_info(self):
        for item in self.clipped:
            self.hide_info(item)

    def remove_info_all(self):
        while self.clipped:
            item = self.clipped.pop()
            if item.clipped:
                self.hide_info(item)
                item.decolor_margins()
                item.clipped = False;


def random_hex_color_str():
    rand_num = random.randint(1, 3)
    l1 = "0123456789ab"
    color = "#"
    for i in range(1, 4):
        if i == rand_num:
            color += "ff"
        else:
            color += (l1[random.randint(0, len(l1)-1)] + l1[random.randint(0, len(l1)-1)])

    return color


def inverted_hex_color_str(color):
    string = str(color).lower()
    code = {}
    l1 = "#;0123456789abcdef"
    l2 = "#;fedcba9876543210"

    for i in range(len(l1)):
        code[l1[i]] = l2[i]

    inverted = ""

    for j in string:
        inverted += code[j]

    return inverted


def hex2rgb_opacity_tuple(color):
    rgb = colors.hex2color(color)
    rgb_list = [int(255 * x) for x in rgb]
    rgb_list.append(OPACITY)
    return QtGui.QColor(rgb_list[0], rgb_list[1], rgb_list[2], rgb_list[3])

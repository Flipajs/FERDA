import numpy as np
import matplotlib.cm as cm
import matplotlib.pylab as plt


def generate_colors(count):
    cm = plt.get_cmap('gist_rainbow')
    return np.array([cm(1. * i / count) for i in range(count)]).astype(float)


def get_q_color(id, ant_num):
    from PyQt4 import QtGui
    r, g, b = get_color(id, ant_num)
    return QtGui.QColor(r, g, b)


def get_color(id, ant_num):
    colors = cm.rainbow(np.linspace(0, 1, ant_num))
    return int(colors[id][0] * 255), int(colors[id][1] * 255), int(colors[id][2] * 255)


def get_opacity(current_depth, max_depth):
    return float((max_depth - current_depth) + float(current_depth/max_depth))/max_depth/2


def get_contrast_color(r, g, b):
    if (r+g+b)/3 < 128:
        return 250, 250, 255
    return 5, 0, 5
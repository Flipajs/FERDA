__author__ = 'fnaiser'

from PyQt5 import QtCore, QtGui

#meaning of these parameters can be found in tooltips
default_settings = {
    # CACHE
    'cache_use': True,
    'cache_mser': True,

    'blur_distance': 10,
    'square_line_width': 5,
    'copy_square_color': QtGui.QColor("lime"),
    'position_square_color': QtGui.QColor("yellow"),
    'open_image': QtGui.QKeySequence(QtCore.Qt.Key_O),
    'save_image': QtGui.QKeySequence(QtCore.Qt.Key_S),
    'fix_image': QtGui.QKeySequence(QtCore.Qt.Key_F),
    'settings': QtGui.QKeySequence(),
    'cancel_fixing': QtGui.QKeySequence(QtCore.Qt.Key_Escape),

    #MSER
    'mser_max_area': 100000,
    'mser_min_margin': 5,
    'mser_min_area': 5,
}

tooltips = {
    'cache_use': 'There will be stored information in working directory to speed up mainly the results tool.',
    'cache_mser': 'Storing MSERs have huge impact on speed but it also needs huge space amount.',
}

def get_default(key):
    return default_settings[key]

def get_tooltip(key):
    if key in tooltips:
        return tooltips[key]

    return ''

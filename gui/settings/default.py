__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore

from core.settings import Settings as S_

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
    'mser_max_area': 0.005,
    'mser_min_margin': 5,
    'mser_min_area': 5,

    #COLORMARKS
    'igbr_i_norm': (255*3 + 1) * 3,
    'igbr_i_weight': 0.5,

    'colormarks_use': True,
    'colormarks_mser_max_area': 200,
    'colormarks_mser_min_area': 5,
    'colormarks_mser_min_margin': 5,
    'colormarks_avg_radius': 10,
    'colormarks_debug': True,
}

tooltips = {
    'cache_use': 'There will be stored information in working directory to speed up mainly the correction tool.',
    'cache_mser': 'Storing MSERs have huge impact on speed but it also needs huge space amount.',

    'igbr_i_weight': 'Used in Igbr space. Defines I component weight compared to gbr, this is useful during computation of distance in Igbr space when you want to have this distance predominantly based on color',

    'colormarks_use': '...',
    'colormarks_mser_max_area': 'Used in colormark detection process to ignore all regions bigger then this parameter (in pixels)',
    'colormarks_mser_min_area': 'Used in colormark detection process to ignore all regions lower then this parameter (in pixels)',
    'colormarks_mser_min_margin': 'Used in colormark detection process to ignore all regions with margin (stability) lower then this parameter',
    'colormarks_avg_radius': 'is automatically set by algorithm, you should not modify this value...',
    'colormarks_debug': 'More values will be stored for easier debug...'
}

def get_default(key):
    return default_settings[key]

def get_tooltip(key):
    if key in tooltips:
        return tooltips[key]

    return ''
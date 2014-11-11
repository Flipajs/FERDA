from PyQt4 import QtGui, QtCore

"""A rather clumsy implementation of default settings. All settings are stored as a String:value pair"""

default_settings = {
    'length_tolerance': 2,
    'minimal_certainty': .2,
    'proximity_tolerance': 5,
    'angular_tolerance': 45,
    'correction_mode': 'individual',
    'len_test': True,
    'certainty_test': True,
    'proximity_test': False,
    'angular_test': False,
    'lost_test': True,
    'collision_test': True,
    'overlap_test': True,
    'history_depth': 30,
    'forward_depth': 30,
    'markers_shown_history': 'center',
    'head_detection': False,
    'autosave_count': 5,
    'zoom_on_faults': True,
    'switches_first': True,
    'undo_redo_mode': 'separate',
    'view_mode': 'individual',
    'center_marker_size': 7,
    'head_marker_size': 10,
    'tail_marker_size': 10,
    'bottom_panel_height': 40,
    'side_panel_width': 100,
    'marker_opacity': .7,

    'forward': QtGui.QKeySequence(QtCore.Qt.Key_B),
    'backward': QtGui.QKeySequence(QtCore.Qt.Key_V),
    'play pause': QtGui.QKeySequence(QtCore.Qt.Key_Space),
    'open data': QtGui.QKeySequence(),
    'open video': QtGui.QKeySequence(),
    'save data': QtGui.QKeySequence(),
    'settings': QtGui.QKeySequence(),
    'load changes': QtGui.QKeySequence(),
    'save changes': QtGui.QKeySequence(),
    'undo change': QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Z),
    'redo change': QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.SHIFT + QtCore.Qt.Key_Z),
    'show history': QtGui.QKeySequence(QtCore.Qt.Key_H),
    'swap ants': QtGui.QKeySequence(QtCore.Qt.Key_S),
    'swap tail and head': QtGui.QKeySequence(),
    'show faults': QtGui.QKeySequence(),
    'next fault': QtGui.QKeySequence(QtCore.Qt.Key_N),
    'previous fault': QtGui.QKeySequence(),
    'toggle highlight': QtGui.QKeySequence(QtCore.Qt.Key_T),
    'cancel correction': QtGui.QKeySequence()
}


def get_default(key):
    return default_settings[key]
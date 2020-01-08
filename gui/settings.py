"""
In this file all global settings are stored.
"""
__author__ = 'fnaiser'
from PyQt4 import QtCore, QtGui


class SettingsType(type):
    """
    Each class which uses this as metaclass can be used in following way

    Settings.attr will call Settings.attr.get()
    Settings.attr_tooltip call Settings.attr.tooltip()
    or Settings.attr = val, sets val to atrr = Item().
    """
    def __new__(meta, classname, bases, classDict):
        meta.caller_name_ = str(classname)
        return type.__new__(meta, classname, bases, classDict)

    def __getattribute__(self, attr):
        if len(attr) > 9 and attr[-9:] == '__tooltip':
            return type.__getattribute__(self, attr[:-9]).tooltip()
        else:
            return type.__getattribute__(self, attr).get()

    def __setattr__(self, attr, val):
        return type.__getattribute__(self, attr).set(val)


class Item:
    def __init__(self, key, val, tooltip=''):
        self.key_ = key
        self.val_ = val
        self.tooltip_ = tooltip

        self.default_type = type(val)
        if type(val) == QtGui.QColor:
            self.val_ = str([val.red(), val.green(), val.blue(), val.alpha()])

    def get(self, type_=None):
        t = self.default_type
        if type_:
            t = type_

        settings = QtCore.QSettings('ferda1')

        if t == QtGui.QColor:
            from ast import literal_eval
            val = str(settings.value(self.key_, self.val_, str))
            val = literal_eval(val)
            return QtGui.QColor(val[0], val[1], val[2], val[3])

        return settings.value(self.key_, self.val_, t)

    def set(self, val):
        settings = QtCore.QSettings('ferda1')
        if self.default_type == QtGui.QColor:
            val = str([val.red(), val.green(), val.blue(), val.alpha()])

        settings.setValue(self.key_, val)

    def tooltip(self):
        return self.tooltip_


# class Cache(object):
#     __metaclass__ = SettingsType
#     use = Item('cache/use', True, 'There will be stored information in working directory to speed up mainly the results tool.')
#     mser = Item('cache/mser', True, 'Storing MSERs have huge impact on speed but it also needs huge space amount.')
#     img_manager_size_MB = Item('cache/img_manager_size_MB', 500, '')
#     region_manager_num_of_instances = Item('cache/region_manager_num_of_instances', 0, '')

class Visualization(metaclass=SettingsType):
    default_region_color = Item('visualization/default_region_color', QtGui.QColor(0, 255, 255, 50), '')
    basic_marker_opacity = Item('visualization/basic_marker_opacity', 0.8, '...')
    segmentation_alpha = Item('visualization/segmentation_alpha', 230, '...')
    no_single_id_filled = Item('visualization/no_single_id_filled', True, '...')
    trajectory_history = Item('visualization/trajectory_history', True, '...')
    history_depth = Item('visualization/history_depth', 10, '...')
    history_depth_step = Item('visualization/history_depth_step', 1, '...')
    history_alpha = Item('visualization/history_alpha', 2.0, '...')
    tracklet_len_per_px = Item('visualization/tracklet_len_per_px_sb', 1, '...')


class Temp(metaclass=SettingsType):
    last_vid_path = Item('temp/last_vid_path', '')
    last_wd_path = Item('temp/last_wd_path', '')
    last_gt_path = Item('temp/last_gt_path', '')


class Controls(metaclass=SettingsType):
    show_settings = Item('controls/show_settings', QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Comma), 'Show settings tab')

    # step by step results
    next_case = Item('controls/sbs/next_case', QtGui.QKeySequence(QtCore.Qt.Key_N))
    prev_case = Item('controls/sbs/prev_case', QtGui.QKeySequence(QtCore.Qt.Key_B))
    confirm = Item('controls/sbs/confirm', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_Space))
    partially_confirm = Item('controls/sbs/partially_confirm', QtGui.QKeySequence(QtCore.Qt.Key_C))
    confirm_path = Item('controls/sbs/confirm_path', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_C))
    fitting_from_left = Item('controls/sbs/fitting_from_left', QtGui.QKeySequence(QtCore.Qt.Key_F))
    fitting_from_right = Item('controls/sbs/fitting_from_right', QtGui.QKeySequence(QtCore.Qt.Key_G))
    remove_region = Item('controls/sbs/remove_region', QtGui.QKeySequence(QtCore.Qt.Key_Backspace))
    remove_chunk = Item('controls/remove_chunk', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_Backspace))
    join_regions = Item('controls/join_regions', QtGui.QKeySequence(QtCore.Qt.Key_J))
    new_region = Item('controls/new_region', QtGui.QKeySequence(QtCore.Qt.Key_R))
    ignore_case = Item('controls/ignore_case', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_I))

    undo_fitting = Item('controls/undo_fitting', QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_F))
    undo_whole_fitting = Item('controls/undo_whole_fitting', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_F))

    stop_action = Item('controls/stop_action', QtGui.QKeySequence(QtCore.Qt.Key_Escape))
    save = Item('controls/save', QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_S))
    save_only_long_enough = Item('controls/save_only_long_enough', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.CTRL + QtCore.Qt.Key_S))
    undo = Item('controls/undo', QtGui.QKeySequence(QtCore.Qt.CTRL + QtCore.Qt.Key_Z))
    get_info = Item('controls/get_info', QtGui.QKeySequence(QtCore.Qt.Key_I))
    hide_show = Item('controls/hide_show', QtGui.QKeySequence(QtCore.Qt.Key_H))

    video_next = Item('controls/video_next', QtGui.QKeySequence(QtCore.Qt.Key_N))
    video_prev = Item('controls/video_prev', QtGui.QKeySequence(QtCore.Qt.Key_B))
    video_play_pause = Item('controls/video_play_pause', QtGui.QKeySequence(QtCore.Qt.Key_Space))
    video_random_frame = Item('controls/video_random_frame', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_R))

    chunk_alpha_blending = Item('controls/chunk_alpha_blending', QtGui.QKeySequence(QtCore.Qt.Key_A))
    chunk_interpolation_fitting = Item('controls/chunk_interpolation_fitting', QtGui.QKeySequence(QtCore.Qt.SHIFT + QtCore.Qt.Key_P))

    # global view
    global_view_join_chunks = Item('controls/gv/join_chunks', QtGui.QKeySequence(QtCore.Qt.Key_J))
    global_view_stop_following = Item('controls/gv/stop_following', QtGui.QKeySequence(QtCore.Qt.Key_S))


class Settings:
    visualization = Visualization
    temp = Temp
    controls = Controls

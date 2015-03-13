__author__ = 'fnaiser'

from PyQt4 import QtCore, QtGui
# from core.settings_ import *

"""
In this file all global settings are stored.
"""


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


class Item():
    def __init__(self, key, val, tooltip=''):
        self.key_ = key
        self.val_ = val
        self.tooltip_ = tooltip
        self.default_type = type(val)

    def get(self, type_=None):
        t = self.default_type
        if type_:
            t = type_

        settings = QtCore.QSettings('FERDA')
        return settings.value(self.key_, self.val_, t)

    def set(self, val):
        settings = QtCore.QSettings('FERDA')
        settings.setValue(self.key_, val)

    def tooltip(self):
        return self.tooltip_


class Cache(object):
    __metaclass__ = SettingsType
    use = Item('cache/use', True, 'There will be stored information in working directory to speed up mainly the correction tool.')
    mser = Item('cache/mser', True, 'Storing MSERs have huge impact on speed but it also needs huge space amount.')


class MSER:
    __metaclass__ = SettingsType
    max_area = Item('mser/max_area', 0.005)
    min_area = Item('mser/min_area', 5)
    min_margin = Item('mser/min_margin', 5)


class Colormarks:
    __metaclass__ = SettingsType
    use = Item('colormarks/use', True)
    mser_max_area = Item('colormarks/mser_max_area', 200, 'Used in colormark detection process to ignore all regions bigger then this parameter (in pixels)')
    mser_min_area = Item('colormarks/mser_min_area', 5, 'Used in colormark detection process to ignore all regions lower then this parameter (in pixels)')
    mser_min_margin = Item('colormarks/mser_min_margin', 5, 'Used in colormark detection process to ignore all regions with margin (stability) lower then this parameter')
    avg_radius = Item('colormarks/avg_radius', 10, 'is automatically set by algorithm, you should not modify this value...')
    debug = Item('colormarks/debug', True, 'More values will be stored for easier debug...')
    igbr_i_norm = Item('colormarks/igbr_i_norm', (255*3 + 1) * 3)
    igbr_i_weight = Item('colormarks/igbr_i_weight', 0.5)


class Visualization:
    __metaclass__ = SettingsType
    default_region_color = Item('visualization/default_region_color', QtGui.QColor(0, 255, 255, 50), '')

class Boo(object):
    __metaclass__ = SettingsType
    use = Item('cache/use', True, 'There will be stored information in working directory to speed up mainly the correction tool.')
    max_area = Item('mser/max_area', 0.005, 'toooltip max_areas')


class Settings:
    cache = Cache
    mser = MSER
    colormarks = Colormarks
    visualization = Visualization


if __name__ == '__main__':
    print Settings.colormarks.use
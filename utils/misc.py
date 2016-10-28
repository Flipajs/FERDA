import os
import sys

from gui.settings.default import get_default

__author__ = 'filip@naiser.cz'
import pickle
import numpy as np
from PyQt4 import QtCore

def get_pickle_data(filepath):
    f = open(filepath, "rb")
    data = pickle.load(f)
    f.close()

    return data


def fill_spaces_up_to(str_, up_to, prefix=False):
    if prefix:
        while len(str_) < up_to:
            str_ = " " + str_
    else:
        while len(str_) < up_to:
            str_ = str_ + " "

    return str_


def float2str(d, precision):
    if np.allclose(d, 0.):
        return "0"

    str_ = "{:."+str(precision)+"f}"
    return str_.format(d)


def print_exception(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print "EXCEPTION RAISED: ", e.message, "\nin: " + f_name + " on line: " + str(exc_tb.tb_lineno)


def is_flipajs_pc():
    """
    This function test whether FERDA is runned on Flipajs pc. It can be used for fast start of program (e.g. loading predefined video etc.) and for
    """
    p = os.path.realpath(__file__)

    if '/Users/flipajs/Documents/dev/ferda/' in p or '/Users/flipajs/Documents/dev/ferda/core/' in p:
        return True
    else:
        return False

def get_settings(key, type=str):
    settings = QtCore.QSettings('FERDA')
    return settings.value(key, get_default(key), type)


def set_settings(key, value):
    settings = QtCore.QSettings('FERDA')
    settings.setValue(key, value)


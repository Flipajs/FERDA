__author__ = 'flipajs'
import pickle
import numpy as np


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
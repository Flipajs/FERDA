__author__ = 'filip@naiser.cz'
import os
import pickle
import numpy as np
import sys
import errno


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise e


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
    print("EXCEPTION RAISED: " + e.message + "\nin: " + f_name + " on line: " + str(exc_tb.tb_lineno))


def is_flipajs_pc():
    """
    This function test whether FERDA is runned on Flipajs pc. It can be used for fast start of program (e.g. loading predefined video etc.) and for
    """
    p = os.path.realpath(__file__)

    if '/Users/flipajs/Documents/dev/ferda/' in p or '/Users/flipajs/Documents/dev/ferda/core/' in p:
        return True
    else:
        return False


def is_matejs_pc():
    """
    This function tests whether FERDA runs under user matej.
    """
    import getpass
    return getpass.getuser() == 'matej'


# Print iterations progress
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    taken from: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """

    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


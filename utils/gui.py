__author__ = 'fnaiser'

from PyQt4 import QtGui


def file_names_dialog(window, text='Select files', path='', filter_=''):
    file_names = QtGui.QFileDialog.getOpenFileNames(window, text, path, filter=filter_)

    names = []
    for f in file_names:
        names.append(str(f))

    return names

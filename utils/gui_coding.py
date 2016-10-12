from PyQt4 import QtCore, QtGui


def add_action(widget, name, shortcut, trigger):
    """
    adds action to widget
    """
    setattr(widget, name, QtGui.QAction(widget))
    a = getattr(widget, name)
    a.setShortcut(QtGui.QKeySequence(shortcut))
    a.triggered.connect(trigger)
    widget.addAction(a)
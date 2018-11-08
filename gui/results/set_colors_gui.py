from __future__ import unicode_literals
from builtins import str
from PyQt4 import QtGui

class SetColorsGui(QtGui.QWidget):
    def __init__(self, project):
        super(SetColorsGui, self).__init__()

        self.project = project

        self.fbox = QtGui.QFormLayout()
        self.setLayout(self.fbox)

        self.buttons = []
        from functools import partial
        for i, a in enumerate(project.animals):
            button = QtGui.QPushButton('change color')
            b, g, r = project.animals[i].color_
            button.setStyleSheet("background-color: rgb({}, {}, {});".format(r, g, b))
            button.clicked.connect(partial(self.change_color, i))
            self.buttons.append(button)

            self.fbox.addRow(str(i), button)

        self.save_b = QtGui.QPushButton('save')
        self.save_b.clicked.connect(self.save)

        self.fbox.addRow(self.save_b)

    def change_color(self, i):
        color = QtGui.QColorDialog.getColor()

        self.project.animals[i].color_ = (color.blue(), color.green(), color.red())
        self.buttons[i].setStyleSheet("background-color: %s;" % color.name())

    def save(self):
        self.project.save()
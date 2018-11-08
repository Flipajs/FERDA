from __future__ import print_function
__author__ = 'fnaiser'

import sys
from functools import partial

from PyQt4 import QtGui, QtCore

from core import animal_group

class GroupsWidget(QtGui.QWidget):
    updated = QtCore.pyqtSignal()

    def __init__(self):
        super(GroupsWidget, self).__init__()

        self.groups = []
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.setAlignment(QtCore.Qt.AlignTop)
        self.vbox.setSpacing(0)
        self.setLayout(self.vbox)

        self.groups.append(animal_group.AnimalGroup('default group'))
        self.list_widget_place = QtGui.QWidget()
        self.list_widget_place.setLayout(QtGui.QVBoxLayout())
        self.list_widget_place.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        self.vbox.addWidget(self.list_widget_place)
        self.list_widget = self.get_list_widget()
        self.list_widget_place.layout().addWidget(self.list_widget)

        self.new_class_layout = QtGui.QFormLayout()
        self.new_class_widget = QtGui.QWidget()
        self.new_class_widget.setLayout(self.new_class_layout)
        self.name = QtGui.QLineEdit()
        self.description = QtGui.QPlainTextEdit()
        self.create_class_button = QtGui.QPushButton('add')
        self.create_class_button.clicked.connect(self.create_new_class)
        self.new_class_layout.addRow('name', self.name)
        self.new_class_layout.addRow('description', self.description)
        self.new_class_layout.addRow(None, self.create_class_button)
        self.vbox.addWidget(self.new_class_widget)
        self.new_class_widget.hide()

        self.new_class_button = QtGui.QPushButton('+ new group')
        self.new_class_button.clicked.connect(self.new_class_widget.show)
        self.new_class_button.clicked.connect(self.name.setFocus)

        # self.name.returnPressed.connect(self.)
        self.connect(self.name, QtCore.SIGNAL('returnPressed()'), self.create_class_button, QtCore.SIGNAL('clicked()'))

        self.vbox.addWidget(self.new_class_button)

    def create_new_class(self):
        c = animal_group.AnimalGroup(self.name.text(), str(self.description.toPlainText()), len(self.groups))
        self.groups.append(c)
        self.name.setText('')
        self.description.setPlainText('')
        self.new_class_widget.hide()
        self.list_widget.hide()
        self.list_widget = self.get_list_widget()
        self.list_widget_place.layout().addWidget(self.list_widget)

        QtGui.QApplication.processEvents()

        self.updated.emit()

    def get_list_widget(self):
        w = QtGui.QWidget()

        vbox = QtGui.QVBoxLayout()
        vbox.setSpacing(0)
        vbox.setMargin(0)
        w.setLayout(vbox)

        id = 0
        for c in self.groups:
            r = QtGui.QWidget()
            r_layout = QtGui.QHBoxLayout()
            r_layout.setMargin(0)

            r.setLayout(r_layout)

            l = QtGui.QLabel(c.name)
            r_layout.addWidget(l)
            b = QtGui.QPushButton('edit')
            b.clicked.connect(partial(self.edit, id))
            r_layout.addWidget(b)
            b = QtGui.QPushButton('remove')
            b.clicked.connect(partial(self.remove_class, id))
            r_layout.addWidget(b)

            vbox.addWidget(r)
            id += 1

        return w

    def edit(self, id):
        print(id)
        self.new_class_widget.show()
        self.name.setText(self.groups[id].name)
        self.description.setPlainText(self.groups[id].description)
        self.create_class_button.clicked.disconnect()
        self.create_class_button.clicked.connect(partial(self.finish_edditing, id))
        self.create_class_button.setText('edit')
        self.name.setFocus()

    def remove_class(self, id):
        m = "Are you sure you want to delete class: "+str(self.groups[id].name)
        reply = QtGui.QMessageBox.question(self, 'Message', m, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            self.groups.remove(self.groups[id])
            self.list_widget.hide()
            self.list_widget = self.get_list_widget()
            self.list_widget_place.layout().addWidget(self.list_widget)

        self.updated.emit()

    def finish_edditing(self, id):
        self.groups[id].name = str(self.name.text())
        self.groups[id].description = str(self.description.toPlainText())

        self.name.setText('')
        self.description.setPlainText('')
        # reconnect(self.create_class_button, self.create_new_class)
        self.create_class_button.clicked.disconnect()
        self.create_class_button.clicked.connect(self.create_new_class)
        self.create_class_button.setText('add')
        self.new_class_widget.hide()

        self.list_widget.hide()
        self.list_widget = self.get_list_widget()
        self.list_widget_place.layout().addWidget(self.list_widget)

        QtGui.QApplication.processEvents()
        self.updated.emit()
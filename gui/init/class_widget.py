__author__ = 'fnaiser'

import sys
from functools import partial

from PyQt4 import QtGui, QtCore

from core import animal_class


class ClassWidget(QtGui.QWidget):
    def __init__(self):
        super(ClassWidget, self).__init__()

        self.classes = []
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.vbox)

        #LABEL
        # self.title_label = QtGui.QLabel('Animal classes: ')
        # font = self.title_label.font()
        # font.setPointSize(40)
        # self.title_label.setFont(font)
        # self.vbox.addWidget(self.title_label)

        self.classes.append(animal_class.AnimalClass('default'))
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



        self.new_class_button = QtGui.QPushButton('+ new class')
        self.new_class_button.clicked.connect(self.new_class_widget.show)
        self.new_class_button.clicked.connect(self.name.setFocus)
        self.vbox.addWidget(self.new_class_button)

        self.show()

    def create_new_class(self):
        c = animal_class.AnimalClass(self.name.text(), str(self.description.toPlainText()), len(self.classes))
        self.classes.append(c)
        self.name.setText('')
        self.description.setPlainText('')
        self.new_class_widget.hide()
        self.list_widget.hide()
        self.list_widget = self.get_list_widget()
        self.list_widget_place.layout().addWidget(self.list_widget)

        QtGui.QApplication.processEvents()

    def get_list_widget(self):
        w = QtGui.QWidget()

        vbox = QtGui.QVBoxLayout()
        w.setLayout(vbox)

        id = 0
        for c in self.classes:
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
        print id
        self.new_class_widget.show()
        self.name.setText(self.classes[id].name)
        self.description.setPlainText(self.classes[id].description)
        self.create_class_button.clicked.disconnect()
        self.create_class_button.clicked.connect(partial(self.finish_edditing, id))
        self.create_class_button.setText('edit')
        self.name.setFocus()

    def remove_class(self, id):
        m = "Are you sure you want to delete class: "+str(self.classes[id].name)
        reply = QtGui.QMessageBox.question(self, 'Message', m, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            self.classes.remove(self.classes[id])
            self.list_widget.hide()
            self.list_widget = self.get_list_widget()
            self.list_widget_place.layout().addWidget(self.list_widget)

    def finish_edditing(self, id):
        print "FINISH"
        self.classes[id].name = str(self.name.text())
        self.classes[id].description = str(self.description.toPlainText())

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

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    ex = ClassWidget()

    app.exec_()
    app.deleteLater()
    sys.exit()
from __future__ import unicode_literals
from PyQt4 import QtGui, QtCore

class MyPopup(QtGui.QWidget):
    def __init__(self, img1):
        super(MyPopup, self).__init__()

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)
        self.scene.addPixmap(QtGui.QPixmap.fromImage(img1))
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().addWidget(self.view)
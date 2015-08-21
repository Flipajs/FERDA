from PyQt4 import QtGui, QtCore
from gui.img_controls import utils

class MyPopup(QtGui.QWidget):
    def __init__(self, img1, img2):
        super(MyPopup, self).__init__()

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()
        self.view.setScene(self.scene)
        self.scene.addPixmap(QtGui.QPixmap.fromImage(img1))
        self.scene.addPixmap(QtGui.QPixmap.fromImage(img2))
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().addWidget(self.view)
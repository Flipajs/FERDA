from PyQt5 import QtCore, QtGui, QtWidgets

class MyScene(QtWidgets.QGraphicsScene):
    def __init__(self, update_callback_release=None):
        super(MyScene, self).__init__()
        self.update_callback_release = update_callback_release

    def mouseReleaseEvent(self, e):
        super(MyScene, self).mouseReleaseEvent(e)

        if self.update_callback_release:
            self.update_callback_release(e)

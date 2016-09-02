from PyQt4 import QtGui, QtCore

class MyScene(QtGui.QGraphicsScene):
    def __init__(self, update_callback_release=None):
        super(MyScene, self).__init__()
        self.update_callback_release = update_callback_release

    def mouseReleaseEvent(self, e):
        super(MyScene, self).mouseReleaseEvent(e)

        if self.update_callback_release:
            self.update_callback_release(e)

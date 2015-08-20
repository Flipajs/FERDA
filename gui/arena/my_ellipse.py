__author__ = 'filip@naiser.cz'
from PyQt4 import QtGui, QtCore

class MyEllipse(QtGui.QGraphicsEllipseItem):
    def __init__(self, ellipse, update_callback=None, radius=10.0):
        super(MyEllipse, self).__init__(-radius/2, -radius/2, radius, radius)
        self.ellipse = ellipse
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)
        self.update_callback = update_callback

    def mouseReleaseEvent(self, e):

        # self.update_geometry()

        if self.update_callback:
            self.update_callback(self)


        return super(MyEllipse, self).mouseReleaseEvent(e)
"""
    def update_geometry(self):
        #print "updating..."
        super(MyEllipse, self).update_geometry()"""

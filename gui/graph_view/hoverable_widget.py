__author__ = 'simon'
# UNFNISHED!!!

from PyQt4 import QtCore, QtGui

class HoverableWidget(QtGui.QWidget):

    def __init__(self, parent):
        super(HoverableWidget, self).__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        self.parent.setMouseTracking(True)
        self.parent.installEventFilter(self)

        self.animation = QtCore.QPropertyAnimation(self, "geometry")
        self.animation.setDuration(250)

    def eventFilter(self, object, event):
        if(self.parent == object) and (event.type() == QtCore.QEvent.MouseMove):
            mouseEvent = QtCore.QEvent.MouseMove(event)
            if event.pos().x() < 10:
                if (self.isHidden()) and (self.animation.state() != QtCore.QPropertyAnimation.Running):
                    self.animation.setStartValue(QtCore.QRect(0, 0, 0, 50))
                    self.animation.setEndValue(QtCore.QRect(0, 0, 40, 50))
                    self.animation.finished().disconnect(self.hide())
                    self.animation.start()
                    self.show()
            elif self.animation.state() != QtCore.QPropertyAnimation.Running:
                if not self.isHidden():
                    self.animation.setStartValue(QtCore.QRect(0, 0, 0, 50))
                    self.animation.setEndValue(QtCore.QRect(0, 0, 40, 50))
                    self.animation.finished().connect(self.hide())
                    self.animation.start()
        eventFilter = QtGui.QWidget.eventFilter(object, event)
        return eventFilter

    def enterEvent(self, QEvent):

        self.show()

    def leaveEvent(self, QEvent):
        self.hide()



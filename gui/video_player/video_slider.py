from PyQt5 import QtCore, QtGui, QtWidgets

class VideoSlider(QtWidgets.QSlider):
    """A slider that changes it's value directly to the part where it was clicked instead of slowly sliding there.
    Also, it's nice! """

    def __init__(self, parent=None):
        super(VideoSlider, self).__init__(parent)
        self.usercontrolled = False
        self.recentlyreleased = False
        self.setPageStep(0)
        self.setSingleStep(0)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: white;
            }

            QSlider::sub-page:horizontal {
                background: green;
                background: qlineargradient(x1: 1, y1: 0,    x2: 0, y2: 0, stop: 0 #085700, stop: 1 #5DA556);
            }

            QSlider::handle:horizontal {
                background: #5DA556;
                border: 0px;
                width: 0px;
                margin-top: 0px;
                margin-bottom: 0px;
                border-radius: 0px;
            }
        """)

    def mousePressEvent(self, QMouseEvent):
        super(VideoSlider, self).mousePressEvent(QMouseEvent)
        self.usercontrolled = True
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        sr = self.style().subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)

        if QMouseEvent.button() == QtCore.Qt.LeftButton and not sr.contains(QMouseEvent.pos()):
            if self.orientation() == QtCore.Qt.Vertical:
                newVal = self.minimum() + ((self.maximum() - self.minimum()) * (self.height()-QMouseEvent.y()))/self.height()
            else:
                newVal = self.minimum() + (self.maximum() - self.minimum()) * QMouseEvent.x() / self.width()
            if self.invertedAppearance():
                self.setValue(self.maximum() - newVal)
            else:
                self.setValue(newVal)

    def mouseReleaseEvent(self, QMouseEvent):
        self.usercontrolled = False
        self.recentlyreleased = True
        super(VideoSlider, self).mouseReleaseEvent(QMouseEvent)

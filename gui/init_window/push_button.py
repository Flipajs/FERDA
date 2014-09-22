__author__ = 'flipajs'

from PyQt4 import QtGui


class PushButton(QtGui.QPushButton):
    def __init__(self, id, window_p):
        super(PushButton, self).__init__()
        self.clicked.connect(self.set_actual_focus)
        self.window_p = window_p
        self.setText(str(-1))
        self.id = id
        c = self.window_p.ants[id].color
        s = c[0]+c[1]+c[2]
        text_c = 'rgb(0, 0, 0)'
        if s < (255*3) / 2:
            text_c = 'rgb(255, 255, 255)'

        self.setStyleSheet("QPushButton { background-color: rgb("+str(c[2])+", "+str(c[1])+", "+str(c[0])+"); color: "+text_c+";}")

    def set_actual_focus(self):
        print self.id
        self.window_p.actual_focus = self.id

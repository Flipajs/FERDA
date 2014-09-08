__author__ = 'flipajs'

from PyQt4 import QtGui

class SpinBox(QtGui.QSpinBox):
    def __init__(self, maxVal, id, window_p, parent=None):
        super(SpinBox, self).__init__(parent)
        self.setMinimum(-1)
        self.setValue(-1)
        self.setMaximum(maxVal)
        self.editingFinished.connect(self.assign_ant)
        self.window_p = window_p
        self.id = id
        c = self.window_p.ants[id].color
        s = c[0]+c[1]+c[2]
        text_c = 'rgb(0, 0, 0)'
        if s < (255*3) / 2:
            text_c = 'rgb(255, 255, 255)'

        self.setStyleSheet("QSpinBox { background-color: rgb("+str(c[2])+", "+str(c[1])+", "+str(c[0])+"); color: "+text_c+";}")

    def assign_ant(self):
        self.window_p.assign_ant(self.id, self.value())

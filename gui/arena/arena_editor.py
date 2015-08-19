__author__ = 'flipajs'

from PyQt4 import QtGui, QtCore
import cv2
import sys
from core.project.project import Project
from gui.img_controls import utils


class ArenaEditor(QtGui.QWidget):
    def __init__(self, img, project):
        super(ArenaEditor, self).__init__()

        self.img = img
        self.project = project

        self.setLayout(QtGui.QVBoxLayout())

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)
        self.scene.addPixmap(utils.cvimg2qtpixmap(img))

        # ukazka toho jak pridat klavesovou zkratku...
        # pri zmacknuti T se zavola funkce test_fce
        self.test_action = QtGui.QAction('test', self)
        self.test_action.triggered.connect(self.test_fce)
        self.test_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_T))
        self.addAction(self.test_action)

        self.layout().addWidget(self.view)

    def test_fce(self):
        print "TEST"


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    # TODO: change path...
    # TODO: pokud se chces podivat do puvodni implementace s kruhovou areno, mrkni do gui/init/init_where_widget.py

    # im = cv2.imread('/home/dita/PycharmProjects/sample2.png')
    im = cv2.imread('/Users/flipajs/Desktop/red_vid.png')
    p = Project()

    ex = ArenaEditor(im, p)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
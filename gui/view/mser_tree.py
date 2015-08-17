__author__ = 'flipajs'

import sys
import cv2
from core.region.mser import get_msers_
from core.project.project import Project
from PyQt4 import QtGui, QtCore


class MSERTree(QtGui.QWidget):
    def __init__(self, img, project):
        super(MSERTree, self).__init__()

        self.img = img
        self.project = project

        self.setLayout(QtGui.QVBoxLayout())

        self.view = QtGui.QGraphicsView()
        self.scene = QtGui.QGraphicsScene()

        self.view.setScene(self.scene)
        self.layout().addWidget(self.view)

        # TODO: z gui/view/graph_visualizer.py kolem line 60... vykresleni oblasti
        # z scripts/region_graph3.py kolem line 307 fce show_node... tam se da zjistit jak
        # pridat obrazek do sceny na nejakou pozici a jak ho mit klikaci...

        pass

    def get_regions(self):
        regions = get_msers_(self.img, self.project)

        for r in regions:
            print r.area(), r.label()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    # change for some test image...
    im = cv2.imread('/Users/flipajs/Desktop/red_vid.png')
    p = Project()
    p.mser_parameters.min_area = 30
    p.mser_parameters.min_margin = 5

    ex = MSERTree(im, p)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
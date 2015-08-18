from gui.pixmap_selectable import Pixmap_Selectable

__author__ = 'flipajs'

import sys
import cv2
import numpy as np

from PyQt4 import QtGui
from PyQt4 import QtCore
from skimage.transform import resize
from core.region.mser import get_msers_
from core.project.project import Project
from gui.img_controls.utils import cvimg2qtpixmap
from scripts.region_graph3 import visualize_nodes


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

        regions = self.get_regions()

        # set image size manually
        image_width = 60
        # set image spacing manually
        image_spacing = 10

        lab = 0
        tmp = 0
        # loop through all regions
        for i in range(0, len(regions)):

            # prepare the image
            # TODO: could be replaced with a function
            r = regions[i]
            vis = visualize_nodes(self.img, r)
            vis = np.asarray(resize(vis, (image_width, image_width)) * 255, dtype=np.uint8)
            pix_map= cvimg2qtpixmap(vis)
            it = self.scene.addPixmap(pix_map)

            # go to next column if 'label' changed
            if r.label() != lab:
                lab = r.label()
                tmp = 0

            # draw image at specified position
            pos_x = (image_spacing + image_width)*r.label()
            pos_y = (2*image_spacing + image_width)*tmp
            it.setPos(pos_x, pos_y)

            # add 'margin = ...' text under each image
            text = QtCore.QString("m = %s" % r.margin_)
            text_item = QtGui.QGraphicsSimpleTextItem(text, scene=self.scene)
            text_item.setPos(pos_x, pos_y + image_width)

            tmp += 1


        # TODO: z gui/view/graph_visualizer.py kolem line 60... vykresleni oblasti
        # z scripts/region_graph3.py kolem line 307 fce show_node... tam se da zjistit jak
        # pridat obrazek do sceny na nejakou pozici a jak ho mit klikaci...


        pass

    def get_regions(self):
        regions = get_msers_(self.img, self.project)

        for r in regions:
            print r.area(), r.label()

        return regions


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    # change for some test image...
    im = cv2.imread('/home/dita/PycharmProjects/sample.png')
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
__author__ = 'flipajs'

import sys
from PyQt4 import QtCore
from PyQt4 import QtGui

import cv2
import numpy as np
from skimage.transform import resize

from core.project.project import Project
from core.region.mser import get_msers_
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

        find_ids = [66, 0, 1, 2]
        qt_points = []

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


            """
            # add 'margin = ...' text under each image
            text = QtCore.QString("m = %s" % r.margin_)
            text_item = QtGui.QGraphicsSimpleTextItem(text, scene=self.scene)
            text_item.setPos(pos_x, pos_y + image_width)
            """
            text = QtCore.QString("id = %s" % r.id_)
            text_item = QtGui.QGraphicsSimpleTextItem(text, scene=self.scene)
            text_item.setPos(pos_x, pos_y + image_width)

            if r.id_ in find_ids:
                print "match found, id: %s [%d, %d]" % (r.id_, pos_x, pos_y)
                focus = QtCore.QPointF(pos_x, pos_y)
                qt_points.append(it)



            tmp += 1



        qt_points[0], qt_points[3] = qt_points[3], qt_points[0]
        # TODO: z gui/view/graph_visualizer.py kolem line 60... vykresleni oblasti
        # z scripts/region_graph3.py kolem line 307 fce show_node... tam se da zjistit jak
        # pridat obrazek do sceny na nejakou pozici a jak ho mit klikaci...
        self.view.update()
        self.scene.update()
        QtGui.QApplication.processEvents()
        self.focus_on(qt_points)

        self.b = QtGui.QPushButton("test")
        from functools import partial
        self.b.clicked.connect(partial(self.focus_on, qt_points))
        self.layout().addWidget(self.b)

    def get_regions(self):
        regions = get_msers_(self.img, self.project)

        for r in regions:
            print r.area(), r.label()

        return regions

    def focus_on(self, items):
        for item in items[:]:
            if item is not isinstance(item, QtGui.QGraphicsPixmapItem):
                items.remove(item)

        # create a rectangle with covering all points
        rect = self.find_rect(items)

        # check if it fits into the view
        if rect.width() < self.view.width() and rect.height() < self.view.height():
            point = rect.center()
            text = QtCore.QString("x")
            text_item = QtGui.QGraphicsSimpleTextItem(text, scene=self.scene)
            text_item.setPos(point)
            print "focusing on rectangle [%s, %s]" % (point.x(), point.y())
            self.view.centerOn(point)
        # if it doesn't, center on the first point
        else:
            # find center of gravity of all points
            sum_x = 0
            sum_y = 0
            for item in items:
                sum_x += item.x()
                sum_y += item.y()
            sum_x /= len(items)
            sum_y /= len(items)
            center_pt = QtCore.QPointF(sum_x, sum_y)

            print "rectangle is too big, centering on point [%s, %s]" % (sum_x, sum_y)
            # point with highest priority
            item = items[0]

            # move the view so that the first point is there
            if center_pt.x() > item.x():
                new_x = item.x() + self.view.width()/2
            else:
                new_x = item.x() - self.view.width()/2 + item.pixmap().width()

            if center_pt.y() > item.y():
                new_y = item.y() + self.view.height()/2
            else:
                new_y = item.y() - self.view.height()/2 + item.pixmap().height()

            text = QtCore.QString("x")
            text_item = QtGui.QGraphicsSimpleTextItem(text, scene=self.scene)
            new = QtCore.QPointF(new_x, new_y)
            text_item.setPos(new)
            print "focusing on point [%s, %s]" % (new_x, new_y)
            self.view.centerOn(new)

    def find_rect(self, items):
        min_x = 10000000
        max_x = -1
        min_y = 10000000
        max_y = -1
        for i in items:
            if i.x() > max_x:
                max_x = i.x()
            if i.x() < min_x:
                min_x = i.x()
            if i.y() > max_y:
                max_y = i.y()
            if i.y() < min_y:
                min_y = i.y()
        result = QtCore.QRectF(min_x, min_y, max_x-min_x, max_y-min_y)
        return result


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    im = cv2.imread('/home/dita/PycharmProjects/sample2.png')
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
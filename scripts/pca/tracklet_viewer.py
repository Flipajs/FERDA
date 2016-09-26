from PyQt4 import QtCore
from PyQt4 import QtGui

from core.graph.region_chunk import RegionChunk
from gui.gui_utils import cvimg2qtpixmap


class TrackletViewer(QtGui.QWidget):
    """
    This class serves for only purpose to view tracklets and extracting them to ground truth
    """
    WIDTH = HEIGHT = 300

    def __init__(self, im, ch, chm, gm, rm):
        super(TrackletViewer, self).__init__()
        self.im = im
        self.regions = list(self.get_regions(ch, chm, gm, rm))
        self.setLayout(QtGui.QVBoxLayout())
        self.buttons = QtGui.QHBoxLayout()
        self.next_b = QtGui.QPushButton('next')
        self.prev_b = QtGui.QPushButton('prev')
        self.img = QtGui.QLabel()
        self.current = -1
        self.prepare_layout()
        self.next_action()
        self.prev_b.setDisabled(True)
        if len(self.regions) == 1:
            self.next_b.setDisabled(True)

    def prepare_layout(self):
        self.layout().addWidget(self.img)
        self.layout().addLayout(self.buttons)
        self.buttons.addWidget(self.prev_b)
        self.buttons.addWidget(self.next_b)
        self.connect(self.prev_b, QtCore.SIGNAL('clicked()'), self.prev_action)
        self.connect(self.next_b, QtCore.SIGNAL('clicked()'), self.next_action)

    def view_region(self):
        region = self.regions[self.current]
        img = self.im.get_crop(region.frame(), region, width=self.WIDTH, height=self.HEIGHT, margin=200)
        pixmap = cvimg2qtpixmap(img)
        self.img.setPixmap(pixmap)
        # plt.scatter(contour[:, 0], contour[:, 1])
        # plt.scatter(contour[0, 0], contour[0, 1], c='r')
        # plt.scatter(region.centroid()[0], region.centroid()[1])
        # plt.show()

    def get_regions(self, ch, chm, gm, rm):
        chunk = chm[ch]
        print chunk
        r_ch = RegionChunk(chunk, gm, rm)
        return r_ch

    def next_action(self):
        print self.current
        if self.current != len(self.regions) - 1:
            self.current += 1
            self.view_region()
            self.prev_b.setDisabled(False)
            if self.current == len(self.regions) - 1:
                self.next_b.setDisabled(True)

    def prev_action(self):
        print self.current
        if self.current != 0:
            self.current -= 1
            self.view_region()
            self.next_b.setDisabled(False)
            if self.current == 0:
                self.prev_b.setDisabled(True)

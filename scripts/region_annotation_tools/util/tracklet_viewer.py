from PyQt6 import QtCore, QtWidgets
from PyQt6 import QtGui, QtWidgets

from core.graph.region_chunk import RegionChunk
from gui.gui_utils import cvimg2qtpixmap
from scripts.region_annotation_tools import TrackletTypes


class TrackletViewer(QtWidgets.QWidget):
    """
    This class serves for only purpose to view tracklets and extracting them to ground truth
    """
    WIDTH = HEIGHT = 600

    def __init__(self, project, tracklets, label_callback, save_callback):
        super(TrackletViewer, self).__init__()
        self.im = project.img_manager
        self.chm = project.chm
        self.gm = project.gm
        self.rm = project.rm

        self.setLayout(QtWidgets.QVBoxLayout())
        self.labels = QtWidgets.QHBoxLayout()
        self.current_id_label = QtWidgets.QLabel()
        self.current_region_label = QtWidgets.QLabel()
        self.previous_id_label = QtWidgets.QLabel()
        self.buttons = QtWidgets.QHBoxLayout()
        self.next_b = QtWidgets.QPushButton('next frame (n)')
        self.prev_b = QtWidgets.QPushButton('prev frame (p)')
        self.blob_b = QtWidgets.QPushButton('blob')
        self.single_b = QtWidgets.QPushButton('single')
        self.other_b = QtWidgets.QPushButton('other')
        self.save_b = QtWidgets.QPushButton('save and exit')
        self.img = QtWidgets.QLabel()
        self.current_frame = -1
        self.tracklets = tracklets
        self.current_tracklet = None
        self.tracklet_callback = label_callback
        self.save_callback = save_callback
        self.prepare_layout()
        self.next_tracklet()


    def prepare_layout(self):
        self.layout().addLayout(self.labels)
        self.layout().addWidget(self.img)
        self.layout().addLayout(self.buttons)
        self.labels.addWidget(self.current_id_label)
        self.labels.addWidget(self.current_region_label)
        self.labels.addWidget(self.previous_id_label)
        self.buttons.addWidget(self.prev_b)
        self.buttons.addWidget(self.next_b)
        self.buttons.addWidget(self.blob_b)
        self.buttons.addWidget(self.single_b)
        self.buttons.addWidget(self.other_b)
        self.buttons.addWidget(self.save_b)
        self.prev_b.clicked.connect(self.prev_action)
        self.next_b.clicked.connect(self.next_action)
        self.blob_b.clicked.connect(self.blob_action)
        self.single_b.clicked.connect(self.single_action)
        self.other_b.clicked.connect(self.other_action)
        self.save_b.clicked.connect(self.save_callback)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_N), self).activated.connect(self.next_action)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_P), self).activated.connect(self.prev_action)
        # self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_B), self), QtCore.SIGNAL('activated()'),
        #              self.blob_action)
        # self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_S), self), QtCore.SIGNAL('activated()'),
        #              self.single_action)
        # self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_O), self), QtCore.SIGNAL('activated()'),
        #              self.other_action)

    def view_region(self):
        region = self.regions[self.current_frame]
        self.current_region_label.setText("Current region id: {0}".format(region.id()))
        img = self.im.get_crop(region.frame(), region, width=self.WIDTH, height=self.HEIGHT, margin=200)
        pixmap = cvimg2qtpixmap(img)
        self.img.setPixmap(pixmap)
        # plt.scatter(contour[:, 0], contour[:, 1])
        # plt.scatter(contour[0, 0], contour[0, 1], c='r')
        # plt.scatter(region.centroid()[0], region.centroid()[1])
        # plt.show()

    def get_regions(self, chunk, gm, rm):
        r_ch = RegionChunk(chunk, gm, rm)
        return r_ch

    def next_tracklet(self):
        self.previous_id_label.setText("Previous tracklet id: {0}".format(
            self.current_tracklet.id() if self.current_tracklet is not None else '-'))
        self.current_tracklet = self.tracklets.pop()
        self.regions = list(self.get_regions(self.current_tracklet, self.gm, self.rm))
        self.current_frame = -1
        self.next_action()
        self.prev_b.setDisabled(True)
        if len(self.regions) == 1:
            self.next_b.setDisabled(True)
        else:
            self.next_b.setDisabled(False)
        self.current_id_label.setText("Current tracklet id: {0}".format(self.current_tracklet.id()))

    def next_action(self):
        if self.current_frame != len(self.regions) - 1:
            self.current_frame += 1
            self.view_region()
            self.prev_b.setDisabled(False)
            if self.current_frame == len(self.regions) - 1:
                self.next_b.setDisabled(True)

    def prev_action(self):
        if self.current_frame != 0:
            self.current_frame -= 1
            self.view_region()
            self.next_b.setDisabled(False)
            if self.current_frame == 0:
                self.prev_b.setDisabled(True)

    def blob_action(self):
        self.tracklet_callback(self.current_tracklet.id(), TrackletTypes.BLOB)
        self.next_tracklet()

    def single_action(self):
        self.tracklet_callback(self.current_tracklet.id(), TrackletTypes.SINGLE)
        self.next_tracklet()

    def other_action(self):
        self.tracklet_callback(self.current_tracklet.id(), TrackletTypes.OTHER)
        self.next_tracklet()


import logging
from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtGui, QtWidgets

import sys

from core.graph.region_chunk import RegionChunk
from core.project.project import Project
import numpy as np
from gui.gui_utils import cvimg2qtpixmap
from gui.segmentation.my_scene import MyScene
from gui.segmentation.my_view import MyView
from gui.segmentation.painter import mask2qimage
from utils.drawing.points import get_contour_without_holes

__author__ = 'simon'


def convex_hull(points):
    if len(points) < 3: return points
    ret = []
    comp = lambda A, B: -1 if A[0] < B[0] or A[0] == B[0] and A[1] < B[1] else 1
    points = sorted(points, comp)

    ret.append(points[0])
    ret.append(points[1])

    for p in points[2:] + list(reversed(points)):
        while len(ret) > 1 and wedge_product(ret[-2], ret[-1], p) <= 0:
            ret.pop()
        ret.append(p)

    ret.pop() # first == last

    return np.array(ret)


def wedge_product(A, B, X):
    return (X[0] - A[0]) * (B[1] - A[1]) - (X[1] - A[1]) * (B[0] - A[0])


class BlobWidget(QtWidgets.QWidget):
    width = 1000
    height = 1000

    def __init__(self, project, tracklets, examples_from_tracklet, save_callback, exit_callback, contains_callback, threshold=.1):
        QtWidgets.QWidget.__init__(self)
        self.save_callback = save_callback
        self.exit_callback = exit_callback
        self.contains_callback = contains_callback
        self.examples_from_tracklet = examples_from_tracklet
        self.project = project
        self.threshold = threshold
        self.img_viewer = ImgPainter(self.project.img_manager, threshold)
        self.region_generator = self.regions_gen(tracklets)
        self.init_gui()
        self.set_threshold(0.1)

        self.r = None
        self.tr_id = None
        self.ants = []

    def next_ant(self):
        bm = (self.img_viewer.get_current_ant_bitmap())

        points = set()
        a, b = np.nonzero(bm)
        for (y, x) in zip(a, b):
            points.add((x,y))

        hull = convex_hull(list(points))

        contour = get_contour_without_holes(np.array(list(points)))

        self.ants.append(contour)
        self.img_viewer.reset_view()

    def next_region(self):
        self.img_viewer.reset()
        self.show_current_selected_ants()
        self.save_region()

        self.last_id.setText("Last ID: {0}".format(self.r.id() if self.r is not None else '-'))
        try:
            self.r = next(self.region_generator)
        except StopIteration:
            self.exit_callback()
            self.close()
            return
        self.curr_id.setText("Current ID: {0}".format(self.r.id()))
        self.img_viewer.set_next(self.r)
        self.roi_tickbox.setChecked(True)

    def show_current_selected_ants(self):
        if len(self.ants) == 0: return
        import matplotlib.pyplot as plt
        for c in self.ants:
            plt.plot(c[:, 0], c[:, 1])

        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.show()

    def save_region(self):
        if self.r is not None and len(self.ants) > 0:
            self.save_callback(self.r.id(), self.r.frame_, self.tr_id, self.ants)
            self.ants = []

    def reset_region(self):
        self.ants = []
        self.reset_ant()

    def reset_ant(self):
        self.img_viewer.reset_view()

    def get_results(self):
        return self.results

    def toggle_roi(self):
        if self.roi_tickbox.isChecked():
            self.img_viewer.show_roi()
        else:
            self.img_viewer.hide_roi()

    def toggle_mode(self):
        if self.img_viewer.mode == 1:
            self.mode_button.setText("GREEN")
        else:
            self.mode_button.setText("RED")
        self.img_viewer.mode = 1 - self.img_viewer.mode

    def regions_gen(self, tracklets):
        for tracklet in tracklets:
            self.tr_id = tracklet.id()
            rch = RegionChunk(tracklet, self.project.gm, self.project.rm)
            step = len(rch)/(self.examples_from_tracklet-1)
            if len(rch) < self.examples_from_tracklet:
                idcs = [0]
            else:
                idcs = [i*step for i in range(self.examples_from_tracklet-1)] + [len(rch) - 1]
            for i in idcs:
                region = rch[i]
                if self.contains_callback(region.id(), region.frame, tracklet.id()):
                    logging.info("Skipping region id {0} from tracklet {1} as it is already labeled".format(
                        region.id(), tracklet.id()
                    ))
                else:
                    yield region

    def set_threshold(self, value):
        value /= 100.0
        value *= value
        self.threshold = value
        self.img_viewer.set_threshold(value)

    def init_gui(self):
        self.showMaximized()

        self.layout = QtWidgets.QHBoxLayout()
        self.left_part = QtWidgets.QWidget()
        self.left_part.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.left_part.setLayout(QtWidgets.QVBoxLayout())
        self.left_part.layout().setAlignment(QtCore.Qt.AlignTop)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.setTickInterval(10)
        self.slider.setValue(self.threshold * 100)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.set_threshold)

        self.roi_tickbox = QtWidgets.QCheckBox("Roi")
        self.roi_tickbox.clicked.connect(self.toggle_roi)
        self.roi_tickbox.toggled.connect(self.toggle_roi)

        self.buttons = QtWidgets.QWidget()
        self.buttons.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.buttons.setLayout(QtWidgets.QVBoxLayout())
        self.buttons.layout().setAlignment(QtCore.Qt.AlignBottom)
        self.next_region_button = QtWidgets.QPushButton('Next Region')
        self.next_region_button.clicked.connect(self.next_region)

        self.next_ant_button = QtWidgets.QPushButton('Next Ant')
        self.next_ant_button.clicked.connect(self.next_ant)

        self.reset_ant_button = QtWidgets.QPushButton('Reset Ant')
        self.reset_ant_button.clicked.connect(self.reset_ant)

        self.reset_region_button = QtWidgets.QPushButton('Reset Region')
        self.reset_region_button.clicked.connect(self.reset_region)

        self.mode_button = QtWidgets.QPushButton('RED')
        self.mode_button.clicked.connect(self.toggle_mode)

        self.show_selected_button = QtWidgets.QPushButton('Show selected')
        self.show_selected_button.clicked.connect(self.show_current_selected_ants)

        self.quit = QtWidgets.QPushButton('save and quit', self)
        self.quit.clicked.connect(self.exit_callback)


        self.buttons.layout().addWidget(self.mode_button)
        self.buttons.layout().addWidget(self.next_ant_button)
        self.buttons.layout().addWidget(self.reset_ant_button)
        self.buttons.layout().addWidget(self.show_selected_button)
        self.buttons.layout().addWidget(self.next_region_button)
        self.buttons.layout().addWidget(self.reset_region_button)
        self.buttons.layout().addWidget(self.quit)

        self.help = QtWidgets.QLabel("Scroll to change sensitivity")
        self.curr_id = QtWidgets.QLabel("")
        self.last_id = QtWidgets.QLabel("")

        self.left_part.layout().addWidget(self.slider)
        self.left_part.layout().addWidget(self.roi_tickbox)
        self.left_part.layout().addWidget(self.buttons)
        self.left_part.layout().addWidget(self.help)
        self.left_part.layout().addWidget(self.curr_id)

        self.layout.addWidget(self.left_part)
        self.layout.addWidget(self.img_viewer)

        self.setLayout(self.layout)


class ImgPainter(MyView):
    WB_dist = np.math.sqrt(3 * np.math.pow(255, 2))
    img = None
    img_roi = None
    img_pixmap = None
    roi_pixmap = None

    img_z_value = 0
    roi_z_value = 1
    bitmaps_z_value = 2

    GREEN = 1
    RED = 0

    def __init__(self, img_manager, threshold=.1):
        super(ImgPainter, self).__init__()
        self.setMouseTracking(False) #override parent
        self.img_manager = img_manager
        self.scene = MyScene()
        self.setScene(self.scene)

        self.threshold = threshold
        self.last_x = None
        self.last_y = None
        self.x = []
        self.y = []

        self.visited = np.array((0,0))
        self.selected = np.array((0,0))
        self.excluded = np.array((0,0))
        self.tmp = np.array((0,0))
        self.bitmask_pixmap = None
        self.exclude_pixmap = None
        self.points_pixmap = None

        self.mode = self.GREEN
        self.pen_size = 1

    def set_threshold(self, threshold):
        # TODO another bug : when user has some points selected, sets threshold and then clicks green point,
        # data are saved
        self.threshold = threshold
        self.update_last()
        self.draw()

    def update_last(self):
        if self.last_x is not None and self.last_y is not None:
            self.visited.fill(False)
            self.tmp.fill(False)
            self.floodfill(self.img[self.last_x, self.last_y], self.last_x, self.last_y)

    def update_all(self):
        self.selected.fill(False)
        self.tmp.fill(False)
        for x, y in zip(self.x, self.y):
            self.visited.fill(False)
            self.floodfill(self.img[x, y], x, y)

    def save_results(self):
        self.selected = (self.selected | self.tmp) & (1 - self.excluded)

    def get_current_ant_bitmap(self):
        self.save_results()
        return self.selected

    def reset_view(self):
        self.reset()
        self.draw()

    def reset(self):
        self.x = []
        self.y = []
        self.last_x = None
        self.last_y = None
        self.visited.fill(False)
        self.selected.fill(False)
        self.excluded.fill(False)
        self.tmp.fill(False)
        # self.bitmask_pixmap = None
        # self.exclude_pixmap = None
        # self.points_pixmap = None

    def floodfill(self, color, x, y):
        stack = [self.find_line_segment(color, x, y)]
        while len(stack) > 0:
            segment = stack.pop()
            x = segment[0]
            for y in range(segment[1], segment[2] + 1):
                if x - 1 >= 0:
                    if self.is_similar(color, x - 1, y):
                        stack.append(self.find_line_segment(color, x - 1, y))
                    else:
                        self.visited[x - 1, y] = True
                if x + 1 < self.img.shape[0]:
                    if self.is_similar(color, x + 1, y):
                        stack.append(self.find_line_segment(color, x + 1, y))
                    else:
                        self.visited[x + 1, y] = True

    def find_line_segment(self, color, x, y):
        y1 = y2 = y
        self.visited[x, y] = True
        self.tmp[x, y] = True
        while y1 - 1 >= 0:
            self.visited[x, y1] = True
            if self.is_similar(color, x, y1 - 1):
                self.tmp[x, y1] = True
                y1 -= 1
            else:
                break
        while y2 + 1 < self.img.shape[1]:
            self.visited[x, y2] = True
            if self.is_similar(color, x, y2 + 1):
                self.tmp[x, y2] = True
                y2 += 1
            else:
                break
        return x, y1, y2

    def is_similar(self, color, x, y):
        return not self.visited[x, y] and not self.excluded[x, y] and self.are_colors_similar(color, self.img[x, y])

    def are_colors_similar(self, color1, color2):
        # euclidian distance
        dist = 0
        for c1, c2 in zip(color1, color2):
            dist += np.math.pow(int(c1) - int(c2), 2)
        return self.threshold * self.WB_dist >= np.math.sqrt(dist)

    def add_point(self, point):
        if type(point) == QtCore.QPointF:
            point = point.toPoint()
        # different canvas and array indexing
        x = point.x()
        y = point.y()
        x, y = y, x

        if self.mode == self.GREEN:
            if not self.excluded[x, y]:
                self.last_x = x
                self.last_y = y
                self.x.append(x)
                self.y.append(y)
                self.update_last()
        else:
            self.excluded[x, y] = True
            self.update_all()

        self.draw()

    def show_img(self):
        self.img_pixmap.setZValue(self.img_z_value)
        self.exclude_pixmap.setZValue(self.img_z_value)
        self.points_pixmap.setZValue(self.img_z_value)

    def show_roi(self):
        self.roi_pixmap.setZValue(self.roi_z_value)

    def hide_roi(self):
        self.roi_pixmap.setZValue(-1)

    def set_next(self, region):
        self.scene.clear()
        self.exclude_pixmap = None
        self.bitmask_pixmap = None
        self.points_pixmap = None
        self.img = self.img_manager.get_crop(region.frame(), region,
                                             default_color=(255, 255, 255, 0))
        self.img_roi = self.img_manager.get_crop(region.frame(), region)
        self.visited = np.zeros(self.img.shape[:2], dtype=bool)
        self.selected = np.zeros(self.img.shape[:2], dtype=bool)
        self.excluded = np.zeros(self.img.shape[:2], dtype=bool)
        self.tmp = np.zeros(self.img.shape[:2], dtype=bool)

        self.img_pixmap = self.scene.addPixmap(cvimg2qtpixmap(self.img))
        self.roi_pixmap = self.scene.addPixmap(cvimg2qtpixmap(self.img_roi))

        self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def draw_bitmask(self, mask, r=0, g=0, b=0):
        r = np.asarray(mask * r, dtype=np.uint8)
        g = np.asarray(mask * g, dtype=np.uint8)
        b = np.asarray(mask * b, dtype=np.uint8)
        a = np.full_like(mask, 80, dtype=np.uint8)
        rgb = np.dstack((r, g, b, a))

        qimg = mask2qimage(mask, rgb)
        pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        return pixmap

    def draw_points(self, x, y):
        r = g =  np.zeros_like(self.selected, dtype=np.uint8)
        b = np.zeros_like(self.selected, dtype=np.uint8)
        mask = np.zeros_like(self.selected)

        for x, y in zip(x, y):
            mask[x, y] = True
            b[x, y] = 255

        a = np.full_like(mask, 255, dtype=np.uint8)
        rgb = np.dstack((r, g, b, a))
        qimg = mask2qimage(mask, rgb)
        pixmap = self.scene.addPixmap(QtGui.QPixmap.fromImage(qimg))
        return pixmap

    def draw(self):
        if self.bitmask_pixmap is not None:
            self.scene.removeItem(self.bitmask_pixmap)
        if self.exclude_pixmap is not None:
            self.scene.removeItem(self.exclude_pixmap)
        if self.points_pixmap is not None:
            self.scene.removeItem(self.points_pixmap)
        self.bitmask_pixmap = self.draw_bitmask((self.selected | self.tmp) & (1 - self.excluded), g=255)
        self.bitmask_pixmap.setZValue(self.bitmaps_z_value)
        self.exclude_pixmap = self.draw_bitmask(self.excluded, r=255)
        self.exclude_pixmap.setZValue(self.bitmaps_z_value)
        self.points_pixmap = self.draw_points(self.x, self.y)
        self.points_pixmap.setZValue(self.bitmaps_z_value)

    def mousePressEvent(self, e):
        super(ImgPainter, self).mousePressEvent(e)
        point = self.mapToScene(e.pos())
        if self.scene.itemsBoundingRect().contains(point):
            if self.mode == self.GREEN:
                self.save_results()
            self.add_point(point)

    def mouseMoveEvent(self, e):
        super(ImgPainter, self).mouseMoveEvent(e)
        point = self.mapToScene(e.pos())
        if self.scene.itemsBoundingRect().contains(point):
            if self.mode == self.GREEN:
                self.save_results()
            self.add_point(point)

    def mouseReleaseEvent(self, QMouseEvent):
        self.draw()

    def wheelEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers != QtCore.Qt.ControlModifier:
            val = np.sqrt(self.threshold)
            if event.angleDelta().y() > 0:
                val += 0.01
            else:
                val -= 0.01
            self.set_threshold(max(0, min(1, val * val)))
        # TODO disabled for the time being due to the bug
        # else:
        #     self.zoomAction(event, scale_factor=1.04)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    chunks = project.gm.chunk_list()

    chunks_with_clusters = [6, 10, 12, 13, 17, 18, 26, 28, 29, 32, 37, 39, 40, 41, 43, 47, 51, 54, 57, 58, 60, 61, 65,
                            67, 69, 73, 75, 78, 81, 84, 87, 90, 93, 94, 96, 99, 102, 105]
    chunks_with_clusters = [chunks[x] for x in chunks_with_clusters]

    app = QtWidgets.QApplication(sys.argv)

    gt = BlobWidget(project, chunks_with_clusters)
    gt.show()

    app.exec_()

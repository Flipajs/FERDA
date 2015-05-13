__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.img_controls.my_scene import MyScene
from gui.gui_utils import cvimg2qtpixmap
import numpy as np
from skimage.transform import resize
from utils.img import get_roi, ROI
from gui.gui_utils import get_image_label
from utils.drawing.points import draw_points_crop, draw_points
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_msers_, get_all_msers
from skimage.transform import resize
from gui.img_controls.my_view import MyView
from gui.img_controls.my_scene import MyScene
import sys
from PyQt4 import QtGui, QtCore
from gui.img_controls.utils import cvimg2qtpixmap
import numpy as np
import pickle
from functools import partial
from core.animal import colors_
from core.region.fitting import Fitting

class ConfigWidget(QtGui.QWidget):
    def __init__(self, G, c, vid):
        super(ConfigWidget, self).__init__()

        self.G = G
        self.c = c

        self.node_size = 70
        self.frame_visu_margin = 100

        self.config_lines = []
        self.node_positions = []
        self.h_ = self.node_size + 3
        self.w_ = self.node_size + 100

        self.user_actions = []

        self.active_node = None

        self.sub_g = self.G.subgraph(self.c['c1'] + self.c['c2'])

        self.it_nodes = {}

        self.active_config = 0
        self.im_t1 = None
        self.im_t2 = None
        self.crop_t1_widget = None
        self.crop_t2_widget = None
        self.crop_visualize = True

        self.node_positions = {}

        frame_t = self.G.node[self.c['c1'][0]]['t']

        if not self.im_t1:
            self.im_t1 = vid.seek_frame(frame_t)
            self.im_t2 = vid.move2_next()

        self.pop_menu_node = QtGui.QMenu(self)
        self.action_remove_node = QtGui.QAction('remove', self)
        self.action_remove_node.triggered.connect(self.remove_node)

        self.action_partially_confirm = QtGui.QAction('confirm this connection', self)
        self.action_partially_confirm.triggered.connect(self.partially_confirm)

        self.action_mark_merged = QtGui.QAction('merged', self)
        self.action_mark_merged.triggered.connect(self.mark_merged)

        self.pop_menu_node.addAction(self.action_remove_node)
        self.pop_menu_node.addAction(self.action_mark_merged)
        self.pop_menu_node.addAction(self.action_partially_confirm)

        self.pop_menu_else = QtGui.QMenu(self)
        self.pop_menu_else.addAction(QtGui.QAction('add node', self))

        self.setLayout(QtGui.QHBoxLayout())
        self.v = QtGui.QGraphicsView()
        self.scene = MyScene()

        self.edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen.setColor(QtGui.QColor(0, 0, 0, 0x38))
        self.edge_pen.setWidth(1)

        self.strong_edge_pen = QtGui.QPen(QtCore.Qt.SolidLine)
        self.strong_edge_pen.setColor(QtGui.QColor(0, 255, 0, 0x78))
        self.strong_edge_pen.setWidth(2)

        self.layout().addWidget(self.v)
        self.v.setScene(self.scene)
        self.layout().addWidget(QtGui.QLabel(str(0 if c['certainty'] < 0.001 else c['certainty'])[0:5]))


        self.draw_scene()
        self.draw_frame()

        self.score_list = QtGui.QVBoxLayout()
        self.layout().addLayout(self.score_list)

        for i in range(min(10, len(self.c['configurations']))):
            s = self.c['scores'][i]
            b = QtGui.QPushButton(str(0 if abs(s) < 0.001 else s)[0:6])
            b.clicked.connect(partial(self.show_configuration, i))
            self.score_list.addWidget(b)
            # self.score_list.addWidget(QtGui.QLabel(str(0 if abs(s) < 0.001 else s)[0:6]))

        self.confirm_b = QtGui.QPushButton('confirm')
        self.confirm_b.clicked.connect(self.confirm_clicked)

        self.layout().addWidget(self.confirm_b)
        self.v.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.connect(self.v, QtCore.SIGNAL('customContextMenuRequested(const QPoint&)'), self.on_context_menu)


    def on_context_menu(self, point):
        it = self.scene.itemAt(self.v.mapToScene(point))

        if isinstance(it, QtGui.QGraphicsPixmapItem):
            self.active_node = self.it_nodes[it]
            self.pop_menu_node.exec_(self.v.mapToGlobal(point))
        else:
            self.pop_menu_else.exec_(self.v.mapToGlobal(point))
            self.active_node = None

    def confirm_clicked(self):
        self.setDisabled(True)

    def get_im(self, n, t1=True):
        if t1:
            im = self.im_t1
        else:
            im = self.im_t2

        vis = draw_points_crop(im, n.pts(), color=self.get_node_color(n), square=True)
        # vis = self.G.node[n]['img']

        if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
            vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)

        return cvimg2qtpixmap(vis)

    def remove_node(self):
        print "REMOVE", self.active_node

    def get_node_color(self, n):
        opacity = 0.5
        c = colors_[self.node_positions[n] + len(self.c['c1'])] + (opacity, )
        for c1, c2 in self.c['configurations'][self.active_config]:
            if c1 == n:
                c = colors_[self.node_positions[n]] + (opacity, )
                break
            if c2 == n:
                c = colors_[self.node_positions[c1]] + (opacity, )
                break

        return c

    def draw_frame(self):
        centroids = []
        for n in self.c['c1']:
            centroids.append(n.centroid())

        roi = get_roi(np.array(centroids))
        m = self.frame_visu_margin

        h_, w_, _ = self.im_t1.shape

        im = self.im_t1
        if self.crop_visualize:
            im = self.im_t1.copy()

            for r in self.c['c1']:
                im = draw_points(im, r.pts(), color=self.get_node_color(r))

        roi = ROI(max(0, roi.y() - m), max(0, roi.x() - m), min(roi.height() + 2*m, h_), min(roi.width() + 2*m, w_))
        crop = np.copy(im[roi.y():roi.y()+roi.height(), roi.x():roi.x()+roi.width(), :])
        self.crop_t1_widget = get_image_label(crop)
        self.layout().insertWidget(0, self.crop_t1_widget)

        if self.crop_visualize:
            im = self.im_t2.copy()

            for r in self.c['c2']:
                im = draw_points(im, r.pts(), color=self.get_node_color(r))

        crop = np.copy(im[roi.y():roi.y()+roi.height(), roi.x():roi.x()+roi.width(), :])
        self.crop_t2_widget = get_image_label(crop)
        self.layout().insertWidget(1, self.crop_t2_widget)

    def mark_merged(self):
        if len(self.c['c1']) > 0 and len(self.c['c2']) > 0:
            avg_area_c1 = 0
            for c1 in c['c1']:
                avg_area_c1 += c1.area()
            avg_area_c1 /= float(len(self.c['c1']))

            avg_area_c2 = 0
            for c2 in c['c2']:
                avg_area_c2 += c2.area()
            avg_area_c2 /= float(len(self.c['c2']))

            t1_ = self.c['c1']
            t2_ = self.c['c2']
            t_reversed = False
            if avg_area_c1 > avg_area_c2:
                t1_ = self.c['c2']
                t2_ = self.c['c1']
                t_reversed = True

            reg = []
            for c2 in t2_:
                if not reg:
                    reg = c2
                else:
                    reg.pts_ = np.append(reg.pts_, c2.pts_, axis=0)

            objects = []
            for c1 in t1_:
                objects.append(c1)

            print self, self.c['c1']
            f = Fitting(reg, objects, num_of_iterations=10)
            f.fit()

            # results = f.animals

    def show_configuration(self, id):
        self.active_config = id
        for it in self.config_lines:
            self.scene.removeItem(it)

        self.config_lines = []

        for n1, n2 in self.c['configurations'][id]:
            if n2 is None:
                continue

            line_ = QtGui.QGraphicsLineItem(self.node_size, self.node_positions[n1]*self.h_ + self.h_/2, self.w_, self.node_positions[n2]*self.h_ + self.h_/2)
            line_.setPen(self.strong_edge_pen)
            self.config_lines.append(line_)
            self.scene.addItem(line_)

    def partially_confirm(self):
        conf = self.c['configurations'][self.active_config]
        n1 = self.active_node

        i = 0
        for n1_, n2_ in conf:
            if n1_ == n1:
                n2 = n2_
                break

            if n2_ == n1:
                n1 = n1_
                n2 = n2_
                break

            i += 1

        for i, n_ in zip(range(len(self.c['c1'])), self.c['c1']):
            if n_ is n1:
                self.c['c1'].pop(i)
                break

        for i, n_ in zip(range(len(self.c['c2'])), self.c['c2']):
            if n_ is n2:
                self.c['c2'].pop(i)
                break

        self.sub_g.remove_node(n1)
        self.sub_g.remove_node(n2)

        self.config_lines = []

        self.scene = MyScene()
        self.v.setScene(self.scene)
        self.draw_scene()

    def draw_scene(self):
        h_pos = 0
        for n in self.c['c1']:
            self.node_positions[n] = h_pos
            it = self.scene.addPixmap(self.get_im(n))
            self.it_nodes[it] = n
            it.setPos(0, h_pos * self.h_)
            h_pos += 1

        max_h_pos = h_pos

        h_pos = 0
        for n in self.c['c2']:
            self.node_positions[n] = h_pos
            it = self.scene.addPixmap(self.get_im(n, t1=False))
            self.it_nodes[it] = n
            it.setPos(self.w_, h_pos * self.h_)
            h_pos += 1

        max_h_pos = max(max_h_pos, h_pos)
        self.v.setFixedHeight(max_h_pos * self.h_)
        for n in self.c['c1']:
            for _, n2 in self.sub_g.out_edges(n):
                line_ = QtGui.QGraphicsLineItem(self.node_size, self.node_positions[n]*self.h_ + self.h_/2, self.w_, self.node_positions[n2]*self.h_ + self.h_/2)
                line_.setPen(self.edge_pen)
                self.scene.addItem(line_)

                # self.show_configuration(0)


class CertaintyVisualizer(QtGui.QWidget):
    def __init__(self, G, vid):
        super(CertaintyVisualizer, self).__init__()
        self.setLayout(QtGui.QVBoxLayout())
        self.scenes_widget = QtGui.QWidget()
        self.scenes_widget.setLayout(QtGui.QVBoxLayout())
        self.scroll_ = QtGui.QScrollArea()
        self.scroll_.setWidgetResizable(True)
        self.scroll_.setWidget(self.scenes_widget)
        self.layout().addWidget(self.scroll_)

        self.G = G
        self.vid = vid
        self.ccs = []
        self.ccs_sorted = False

    def visualize_n_sorted(self, n, start=0):
        if not self.ccs_sorted:
            self.ccs = sorted(self.ccs, key=lambda k: k['certainty'])
            self.ccs_sorted = True

        for i in range(start, min(start+n, len(self.ccs))):
            cw = ConfigWidget(self.G, self.ccs[i], self.vid)
            self.scenes_widget.layout().addWidget(cw)

    def add_configuration(self, cc):
        self.ccs_sorted = False
        self.ccs.append(cc)
        cw = ConfigWidget(self.G, cc, self.vid)
        self.scenes_widget.layout().addWidget(cw)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    with open('/Volumes/Seagate Expansion Drive/mser_svm/eight/certainty_visu.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g = up.load()
        ccs = up.load()
        vid_path = up.load()

    cv = CertaintyVisualizer(g, get_auto_video_manager(vid_path))

    i = 0
    for c in ccs:
        if i == 10:
            break

        cv.add_configuration(c)
        i += 1

    cv.showMaximized()


    app.exec_()
    sys.exit()
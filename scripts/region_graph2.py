__author__ = 'fnaiser'

import networkx as nx
import matplotlib.pyplot as plt
from utils.img import get_safe_selection
from utils.drawing.points import draw_points_crop, get_contour, draw_points
import cv2
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_msers_, get_all_msers
from skimage.transform import resize
from gui.img_controls.my_view import MyView
from gui.img_controls.my_scene import MyScene
import sys
from PyQt4 import QtGui, QtCore
from gui.img_controls.utils import cvimg2qtpixmap
import numpy as np
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from matplotlib.mlab import normpdf
import math
from copy import copy
from functools import partial
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pickle
from core.animal import colors_

# max speed of #px / frame
MAX_SPEED = 200
MIN_AREA = 50

AVG_AREA = 240
UNDEFINED_POS_SCORE = 0.1


USE_UNDEFINED = False
S_THRESH = 0.8


vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
working_dir = '/Users/fnaiser/Documents/graphs'
# vid_path = '/Users/fnaiser/Documents/chunks/NoPlasterNoLid800.m4v'
# working_dir = '/Users/fnaiser/Documents/graphs2'
# AVG_AREA = 150
# MAX_SPEED = 120
n_frames = 10


def select_msers(im):
    msers = get_msers_(im)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    ids = area_filter(msers, ids, MIN_AREA)
    ids = children_filter(msers, ids)
    ids = sort_by_distance_from_origin(msers, ids)

    return [msers[i] for i in ids]


def select_msers_cached(frame):
    msers = get_all_msers(frame, vid_path, working_dir)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    ids = area_filter(msers, ids, MIN_AREA)
    ids = children_filter(msers, ids)
    ids = sort_by_distance_from_origin(msers, ids)

    return [msers[i] for i in ids]


def sort_by_distance_from_origin(regions, ids):
    dists = [np.linalg.norm(regions[i].centroid()) for i in ids]

    indices = np.argsort(dists)
    ids = [ids[i] for i in indices]

    return ids


def add_edges(region, region_y, region_x, prev_msers, scene, G):
    to_x = region_x * (width + x_margin)
    to_y = top_offset + region_y * (height + y_margin) + height / 2
    y = 0

    if prev_msers:
        for r in prev_msers:
            if np.linalg.norm(region.centroid() - r.centroid()) < MAX_SPEED:
                s = position_score(r, region)
                s *= area_diff_score(r, region)

                if s > S_THRESH:
                    from_x = to_x - x_margin
                    from_y = top_offset + y * (height + y_margin) + height / 2


                    pen = QtGui.QPen(QtCore.Qt.DashLine)
                    pen.setColor(QtGui.QColor(0, 0, 0, 0x88))
                    pen.setWidth(1.5)

                    line_ = QtGui.QGraphicsLineItem(from_x, from_y, to_x, to_y)
                    line_.setPen(pen)
                    scene.addItem(line_)
                    G.add_edge(r, region)

                    if False:
                        x_ = (region.area() - r.area()) / float(AVG_AREA)
                        c_ = int(round(x_))
                        d_ = x_ - c_

                        std_ = 2 / 3.
                        n_ = normpdf(0, 0, std_)
                        p_ = normpdf(d_, 0, std_) / n_

                        G.add_edge(region, r, weight=p_, c=c_)

                        t_ = QtGui.QGraphicsTextItem('a = ' + str(r.area()) + ' a_ = ' + str(r.a_) + ' b_ = ' + str(r.b_))
                        r_ = line_.boundingRect()

                        text_pos = 0.7
                        text_pos_h_ = 0.7
                        if from_y > to_y:
                            text_pos_h_ = 0.3

                        t_.setPos(r_.x() + r_.width() * text_pos, r_.y() + r_.height() * text_pos_h_)
                        scene.addItem(t_)

                        if r in edges_:
                            edges_[r].append([line_, region])
                        else:
                            edges_[r] = [[line_, region]]
                    else:
                        if r in edges_:
                            edges_[r].append([line_, region])
                        else:
                            edges_[r] = [[line_, region]]

            y += 1


def emphasize_edges(reg, select=True, go_deep=False):
    if reg in visited:
        return
    else:
        visited[reg] = True

    pen = QtGui.QPen(QtCore.Qt.DashLine)
    pen.setColor(QtGui.QColor(0, 0, 0, 0x88))
    pen.setWidth(3)
    bold = False
    color = QtGui.QColor(0, 0, 0, 0xff)
    z = 0

    if not select:
        pen.setWidth(1)
        try:
            selected.remove(reg)
        except:
            pass
    else:
        selected.append(reg)
        bold = True
        color = QtGui.QColor(40, 215, 40, 0xff)
        z = 1000

    if reg in edges_:
        for e, r in edges_[reg]:
            e.setPen(pen)
            if go_deep:
                emphasize_edges(r, select, go_deep)

        for t in texts_[reg]:
            f = t.font()
            f.setBold(bold)
            t.setFont(f)
            t.setZValue(z)
            t.setDefaultTextColor(color)


def dist_to_nearest(pt, ids):
    pass

def movement_prediction():
    pass

def scene_clicked(pos):
    global visited

    visited = {}

    modifiers = QtGui.QApplication.keyboardModifiers()
    go_deep = False
    if modifiers == QtCore.Qt.ControlModifier:
        go_deep = True

    it = scene.itemAt(pos)
    if isinstance(it, QtGui.QGraphicsPixmapItem):
        reg = regions[it]
        select = True
        if reg in selected:
            select = False

        emphasize_edges(reg, select, go_deep=go_deep)

def position_score(r1, r2):
    x = np.linalg.norm(r1.centroid() - r2.centroid())
    n_ = normpdf(0, 0, MAX_SPEED / 3)
    return normpdf(x, 0, MAX_SPEED / 3) / n_

def area_diff_score(r1, r2):
    min_ = min(r1.area(), r2.area())
    x = (r1.area() - r2.area()) / float(min_)

    std = 1.5 / 3.

    return normpdf(x, 0, std) / normpdf(0, 0, std)

def final_area_score(classes_num, reg):
    # TOOD: replace
    class_avg_area = [250]

    supposed_area = 0

    if isinstance(classes_num, list):
        for i in classes_num:
            supposed_area = class_avg_area[i] * classes_num
    else:
        supposed_area = class_avg_area[0] * classes_num

    if supposed_area == 0:
        print "supposed_area = 0", classes_num

    x = (reg.area() - supposed_area) / float(supposed_area)
    # TODO : set std based on some experiment...

    # smaller
    if x < 0:
        std = 0.1 / 3
    else:
        std = 0.7 / 3

    return normpdf(x, 0, std) / normpdf(0, 0, std)

def remove_in_edges(G, n):
    to_remove = []
    for e_ in G.in_edges(n):
        G.remove_edge(e_[0], e_[1])
        to_remove.append(e_[0])

    for n_ in to_remove:
        remove_out_edges(G, n_)

def remove_out_edges(G, n):
    to_remove = []
    for e_ in G.out_edges(n):
        G.remove_edge(e_[0], e_[1])
        to_remove.append(e_[1])

    for n_ in to_remove:
        remove_in_edges(G, n_)

def simplify_g(G):
    print nx.number_of_nodes(G)
    print nx.number_of_edges(G)

    for n in G.nodes():
        if G.out_degree(n) > 1:
            remove_out_edges(G, n)
        elif G.in_degree(n) > 1:
            remove_in_edges(G, n)

    for n in G.nodes():
        if G.out_degree(n) == 1 and G.in_degree(n) == 1:
            n_in = G.in_edges([n])[0][0]
            n_out = G.out_edges([n])[0][1]
            G.remove_node(n)
            G.add_edge(n_in, n_out)

def get_chunk(G, n, n2=None):
    ch = []

    if n2:
        while True:
            c = n.centroid()
            ch.append([c[0], c[1], G.node[n]['t'], n.area()])
            e_ = G.out_edges([n])
            if e_ and n != n2:
                n = e_[0][1]
            else:
                break
    else:
        ch = [[n.centroid()[0], n.centroid()[1], G.node[n]['t'], n.area()]]

    return ch

class VisualizeGraph():
    def __init__(self, G):
        self.G = G
        self.w = QtGui.QWidget()
        self.v = QtGui.QGraphicsView()
        self.w.setLayout(QtGui.QVBoxLayout())
        self.w.layout().addWidget(self.v)

        self.scene = QtGui.QGraphicsScene()
        self.v.setScene(self.scene)

        self.max_t = 1000

    def item_moved(self, pos):
        print pos

    def visualize_id_lines(self, ids, height, width, y_margin, x_margin, top_offset):
        y = 0
        y_offset = 0
        for id in ids:
            c_ = colors_[id]
            pen = QtGui.QPen(QtCore.Qt.SolidLine)
            pen.setColor(QtGui.QColor(c_[0], c_[1], c_[2], 0x88))
            pen.setWidth(2)

            y_ = top_offset + y*(height+y_margin) + y_offset
            line_ = QtGui.QGraphicsLineItem(0, y_, (self.max_t + 1) * (width + x_margin), y_)
            line_.setPen(pen)
            self.scene.addItem(line_)
            y += 1

            for i in range(self.max_t):
                x_ = (width + x_margin) * i
                r_ = 1
                el_ = QtGui.QGraphicsEllipseItem(QtCore.QRectF(x_ - r_, y_ - r_, 2*r_, 2*r_))
                self.scene.addItem(el_)

    def visualize_graph(self, height=50, width=50, y_margin=3, x_margin=10, top_offset=0):
        ys = {}
        positions = {}
        items = {}

        self.visualize_id_lines([i for i in range(8)], height, width, y_margin, x_margin,top_offset)

        for n, d in self.G.nodes(data=True):
            t = d['t']
            ys[t] = ys.get(t, -1) + 1
            y = ys[t]

            it = self.scene.addPixmap(cvimg2qtpixmap(d['image']))
            it.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True)
            p = (t * (width + x_margin), y * (height + y_margin) + top_offset)
            it.setPos(p[0], p[1])

            positions[n] = p
            items[n] = it

        pen = QtGui.QPen(QtCore.Qt.DashLine)
        pen.setColor(QtGui.QColor(0, 0, 0, 0x88))
        pen.setWidth(1)

        for e in self.G.edges():
            p1 = positions[e[0]]
            p2 = positions[e[1]]

            w2 = p2[0]-p1[0]
            line_ = QtGui.QGraphicsLineItem(width, height/2, w2, height/2)
            line_.setPen(pen)

            self.scene.addItem(line_)

            line_.setParentItem(items[e[0]])
            items[e[1]].setParentItem(items[e[0]])
            items[e[1]].setPos(w2, 0)
            items[e[1]].setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)

            # group = QtGui.QGraphicsItemGroup(scene=self.scene)
            # group.addToGroup(items[e[0]])
            # group.addToGroup(items[e[1]])
            # group.addToGroup(line_)
            # group.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
            # line_.setFlag(QtGui.QGraphicsItem.ItemIsMovable, False)
            # line_.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, False)

            # self.scene.createItemGroup(group)

        return self.w

    def graph_scene_clicked(self, pos):
        it = self.scene.itemAt(pos)
        if isinstance(it, QtGui.QGraphicsPixmapItem):
            reg = regions[it]
            select = True
            if reg in selected:
                select = False

            emphasize_edges(reg, select)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    with open(working_dir+'/g.pkl', 'rb') as f:
        G = pickle.load(f)

    simplify_g(G)
    vg = VisualizeGraph(G)

    w = vg.visualize_graph()
    w.showMaximized()


    app.exec_()
    sys.exit()


    #
    #
    # vid = get_auto_video_manager(vid_path)
    #
    # selected = []
    #
    # visited = {}
    #
    # edges_ = {}
    # texts_ = {}
    # regions = {}
    # app = QtGui.QApplication(sys.argv)
    #
    #
    # w = QtGui.QWidget()
    #
    # v = QtGui.QGraphicsView()
    #
    # w.setLayout(QtGui.QVBoxLayout())
    # w.layout().addWidget(v)
    #
    #
    # scene = MyScene()
    # scene.clicked.connect(scene_clicked)
    # v.setScene(scene)
    #
    # w.showMaximized()
    #
    #
    # w2 = QtGui.QWidget()
    #
    # v2 = QtGui.QGraphicsView()
    #
    # w2.setLayout(QtGui.QVBoxLayout())
    # w2.layout().addWidget(v2)
    #
    #
    # scene2 = MyScene()
    # # scene2.clicked.connect(scene_clicked2)
    # v2.setScene(scene2)
    #
    # w2.show()
    #
    # # w_hyps = QtGui.QWidget()
    # # w_hyps.setLayout(QtGui.QHBoxLayout())
    # # w_hyps.show()
    #
    # G = nx.DiGraph()
    # y_margin = 3
    # x_margin = 300
    # height = 50
    # width = 50
    #
    # top_offset = height + x_margin + 40
    #
    # pos = {}
    #
    # prev_msers = None
    #
    # frame_offset = 0
    # vid.seek_frame(frame_offset)
    #
    # Rs = []
    #
    # for frame in range(1, n_frames+1):
    #     print frame
    #
    #     im = vid.move2_next()
    #     msers = select_msers_cached(frame + frame_offset)
    #
    #     y = 0
    #     h_, w_, _ = im.shape
    #     if h_ <= w_:
    #         h_ = (width + x_margin) * (h_ / float(w_))
    #     w_ = width + x_margin
    #     top_offset = h_ + 40
    #
    #     im_ = cvimg2qtpixmap(np.asarray(resize(im, (h_, w_)) * 255, dtype=np.uint8))
    #     it = scene.addPixmap(im_)
    #     it.setPos(frame * (width + x_margin), 20)
    #
    #     t_ = QtGui.QGraphicsTextItem('frame = ' + str(frame + frame_offset))
    #     t_.setPos(frame * (width + x_margin), 0)
    #     scene.addItem(t_)
    #
    #     Rs.append(msers)
    #
    #     im_ = np.copy(im)
    #     for r in msers:
    #         node_name = str(frame) + '_' + str(y)
    #         G.add_node(r, t=frame)
    #
    #         im_crop = draw_points_crop(im_, get_contour(r.pts()), color=(0, 0, 255, 0.7), square=True)
    #
    #         im_crop = np.asarray(resize(im_crop, (height, width)) * 255, dtype=np.uint8)
    #         G.node[r]['image'] = im_crop
    #
    #         pos[node_name] = (frame * (width + x_margin), y * (height + y_margin) + top_offset)
    #         it = scene.addPixmap(cvimg2qtpixmap(im_crop))
    #         it.setPos(pos[node_name][0], pos[node_name][1])
    #
    #         regions[it] = r
    #
    #         add_edges(r, y, frame, prev_msers, scene, G)
    #
    #         y += 1
    #
    #     prev_msers = msers
    #
    # with open(working_dir+'/g.pkl', 'wb') as f:
    #     pickle.dump(G, f)
    #
    # G_sim = nx.DiGraph()
    #
    # for n, d in G.nodes(data=True):
    #     G_sim.add_node(n, d)
    #
    # G_sim.add_edges_from(G.edges())
    # # G_sim = G.copy()
    # simplify_g(G_sim)
    #
    # y = 0
    # pen = QtGui.QPen(QtCore.Qt.DashLine)
    # pen.setColor(QtGui.QColor(0, 0, 0, 0x88))
    # pen.setWidth(1)
    #
    # x_margin_ = 10
    #
    # chunks = []
    #
    # lengths = []
    # items = []
    #
    # for n, d in G_sim.nodes(data=True):
    #     if len(G_sim.in_edges(n)) == 0:
    #         # it = scene2.addPixmap(cvimg2qtpixmap(d['image']))
    #         frame = d['t']
    #         # from_x = frame * (width + x_margin_)
    #         # from_x = 0
    #         #
    #         # from_y = y * (height + y_margin) + top_offset
    #         # it.setPos(from_x, from_y)
    #
    #         e = G_sim.out_edges(n)
    #         if e:
    #             n2 = e[0][1]
    #
    #             d2 = G_sim.node[n2]
    #             # it2 = scene2.addPixmap(cvimg2qtpixmap(d2['image']))
    #             frame2 = d2['t']
    #             # to_x = (frame2-frame) * (width + x_margin_)
    #             # to_y = y * (height + y_margin) + top_offset
    #             # it2.setPos(to_x, to_y)
    #             #
    #             # line_ = QtGui.QGraphicsLineItem(from_x+width, from_y+height/2, to_x, to_y+height/2)
    #             # line_.setPen(pen)
    #             # scene2.addItem(line_)
    #
    #
    #             #text
    #             # t_ = QtGui.QGraphicsTextItem(str(frame)+' - '+str(frame2))
    #             #
    #             # t_.setPos((from_x + width + to_x) / 2, (from_y + to_y) / 2)
    #             # scene2.addItem(t_)
    #
    #             chunks.append(get_chunk(G, n, n2))
    #
    #             lengths.append(frame2-frame)
    #             items.append({'im1': d['image'], 'im2': d2['image'], 'frame': frame, 'frame2': frame2})
    #         else:
    #             chunks.append(get_chunk(G, n))
    #             lengths.append(0)
    #             items.append({'im1': d['image'], 'frame': frame})
    #
    #         y += 1
    #
    #
    # ids = np.argsort(-np.array(lengths))
    # y = 0
    # for id in ids:
    #     v_ = items[id]
    #     it = scene2.addPixmap(cvimg2qtpixmap(v_['im1']))
    #     frame = v_['frame']
    #     from_x = 0
    #
    #     from_y = y * (height + y_margin) + top_offset
    #     it.setPos(from_x, from_y)
    #
    #     if 'im2' in v_:
    #         it2 = scene2.addPixmap(cvimg2qtpixmap(v_['im2']))
    #         to_x = (v_['frame2'] - v_['frame']) * (width + x_margin_)
    #         to_y = y * (height + y_margin) + top_offset
    #         it2.setPos(to_x, to_y)
    #
    #         line_ = QtGui.QGraphicsLineItem(from_x+width, from_y+height/2, to_x, to_y+height/2)
    #         line_.setPen(pen)
    #         scene2.addItem(line_)
    #
    #         t_ = QtGui.QGraphicsTextItem(str(frame)+' - '+str(v_['frame2']))
    #
    #         t_.setPos((from_x + width), (from_y + to_y) / 2)
    #         scene2.addItem(t_)
    #     y += 1
    #
    # # chunks = np.array(chunks)
    #
    #
    # with open(working_dir+'/chunks.pkl', 'wb') as f:
    #     pickle.dump(chunks, f)
    #
    # print len(chunks)
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # # z = np.linspace(-2, 2, 100)
    # # r = z**2 + 1
    # # x = r * np.sin(theta)
    # # y = r * np.cos(theta)
    #
    # for ch in chunks:
    #     ch =  np.array(ch)
    #     if ch.shape[0] == 1:
    #         # ax.plot([ch[0, 1], ch[0, 1]], [ch[0, 0], ch[0, 0]], [ch[0, 2] - 0.25, ch[0, 2]+0.25])
    #         ax.scatter(ch[0, 1], ch[0, 0], ch[0, 2])
    #     else:
    #         avg_a = np.mean(ch[:, 3])
    #         w_ = max(1, math.log(avg_a)-4)
    #         ax.plot(ch[:, 1], ch[:, 0], ch[:, 2], linewidth=w_)
    #
    #     plt.hold(True)
    #
    # plt.hold(False)
    # plt.show()
    #
    # app.exec_()
    # sys.exit()
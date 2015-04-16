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
from scripts.similarity_test import similarity_loss
from methods.bg_model.max_intensity import MaxIntensity
import time
from scipy.ndimage import gaussian_filter


# max speed of #px / frame
MAX_SPEED = 200
MIN_AREA = 50

AVG_AREA = 240
UNDEFINED_POS_SCORE = 0.1
UNDEFINED_EDGE_THRESH = -0.5
UNDEFINED_EDGE_THRESH = -2.0

MIN_I_DISTANCE_COEF = 20.0

USE_UNDEFINED = False
S_THRESH = 0.8


CACHE_IMGS = False




SIMILARITY = 'sim'
STRONG = 's'
CONFIRMED = 'c'
MERGED = 'm'
SPLIT = 'split'

TSD_CONFIRM_THRESH = 30
TSDI_CONFIRM_THRESH = UNDEFINED_EDGE_THRESH
SCORE_CONFIRM_THRESH = UNDEFINED_EDGE_THRESH

USE_BG_SUB = False

with open('/Users/fnaiser/Documents/graphs/log_hists.pkl', 'rb') as f:
    log_hists = pickle.load(f)

NODE_SIZE = 50
MIN_AREA = 30

AVG_MAIN_A = 40
vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
working_dir = '/Users/fnaiser/Documents/graphs'

vid_path = '/Users/fnaiser/Documents/chunks/NoPlasterNoLid800.m4v'
working_dir = '/Users/fnaiser/Documents/graphs2'
AVG_AREA = 150
AVG_MAIN_A = 25
MAX_SPEED = 60
NODE_SIZE = 30
MIN_AREA = 25
#
vid_path = '/Users/fnaiser/Documents/Camera 1_biglense1.avi'
working_dir = '/Users/fnaiser/Documents/graphs5'
MAX_SPEED = 100
AVG_MAIN_A = 50
NODE_SIZE = 60
MIN_AREA = 500

USE_BG_SUB = False

CACHE = True
n_frames = 900

#    ALPHA = 19.52
#     BETA = 18.48
#
#     pred = np.array([0, 0])
#     if G.in_degree(n1) == 1:
#         for m1, m2, d in G.in_edges(n1, data=True):
#             if d['type'] == CONFIRMED:
#                 pred = n1.centroid() - m1.centroid()
#
#
#     t = abs(n1.theta_ - n2.theta_)
#     t *= ALPHA
#
#     s = abs(n1.area() - n2.area()) / float(min(n1.area(), n2.area()))
#     if s < .5:
#         if n1.area() < n2.area():
#             s = similarity_loss(n2, n1)
#         else:
#             s = similarity_loss(n1, n2)
#     else:
#         s *= 2
#
#     s *= BETA
#     d = np.linalg.norm(n1.centroid() + pred - n2.centroid())
#

def get_hist_val(data, bins, query):
    if query > bins[-2]:
        return np.min(data)
    else:
        return data[np.searchsorted(bins, query)]


def d_lhist_score(G, hists, n1, n2):
    pred = np.array([0, 0])
    if G.in_degree(n1) == 1:
        for m1, _ in G.in_edges(n1):
            pred = n1.centroid() - m1.centroid()

    d = np.linalg.norm(n1.centroid() + pred - n2.centroid()) / AVG_MAIN_A

    bins = hists['distances']['bins']
    data = hists['distances']['data']

    return get_hist_val(data, bins, d)


def o_lhist_score(hists, n1, n2):
    t = abs(n1.theta_ - n2.theta_)

    bins = hists['thetas']['bins']
    data = hists['thetas']['data']

    return get_hist_val(data, bins, t)


def s_lhist_score(hists, n1, n2):
    s = abs(n1.area() - n2.area()) / float(min(n1.area(), n2.area()))
    if s < .5:
        if n1.area() < n2.area():
            s = similarity_loss(n2, n1)
        else:
            s = similarity_loss(n1, n2)
    else:
        s *= 2

    return get_hist_val(hists['similarities']['data'], hists['similarities']['bins'], s)


def m_lhist_score(hists, n1, n2):
    m = n1.min_intensity_ - n2.min_intensity_

    return get_hist_val(hists['minI']['data'], hists['minI']['bins'], m-1)


def get_hist_score(G, hists, n1, n2):
    d = d_lhist_score(G, hists, n1, n2)
    s = s_lhist_score(hists, n1, n2)
    o = o_lhist_score(hists, n1, n2)
    m = m_lhist_score(hists, n1, n2)

    return -(d + s + o + m)

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

def num_strong_in_edges(G, n):
    num = 0
    last_n = None
    for n_, _, d in G.in_edges(n, data=True):
        if d['type'] == STRONG or SIMILARITY:
            num += 1
            last_n = n_

    return num, last_n

def num_strong_out_edges(G, n):
    num = 0
    last_n = None

    for _, n_, d in G.out_edges(n, data=True):
        if d['type'] == STRONG or SIMILARITY:
            num += 1
            last_n = n_

    return num, last_n

def simplify_g(G):
    for n in G.nodes():
        in_num, in_n = num_strong_in_edges(G, n)
        out_num, out_n = num_strong_out_edges(G, n)

        if out_num == 1 and in_num == 1:
            G.remove_node(n)
            G.add_edge(in_n, out_n, type='s')

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


def tsd_vals(G, n1, n2):
    # weights in distance
    ALPHA = 19.52
    BETA = 18.48

    pred = np.array([0, 0])
    if G.in_degree(n1) == 1:
        for m1, m2, d in G.in_edges(n1, data=True):
            if d['type'] == CONFIRMED:
                pred = n1.centroid() - m1.centroid()


    t = abs(n1.theta_ - n2.theta_)
    t *= ALPHA

    s = abs(n1.area() - n2.area()) / float(min(n1.area(), n2.area()))
    if s < .5:
        if n1.area() < n2.area():
            s = similarity_loss(n2, n1)
        else:
            s = similarity_loss(n1, n2)
    else:
        s *= 2

    s *= BETA
    d = np.linalg.norm(n1.centroid() + pred - n2.centroid())


    return t, s, d

def tsd_distance(G, n1, n2):
    t, s, d = tsd_vals(G, n1, n2)

    return (t**2 + s**2 + d**2)**0.5

def tsdi_distance(G, n1, n2):
    GAMMA = 1/3.
    t, s, d = tsd_vals(G, n1, n2)
    i = GAMMA * (abs(n1.min_intensity_ - n2.min_intensity_))

    return (t**2 + s**2 + d**2 + i**2)**0.5

class NodeGraphVisualizer():
    def __init__(self, G, imgs, regions):
        self.G = G
        self.imgs = imgs
        self.regions = regions
        self.vid = get_auto_video_manager(vid_path)

        self.w = QtGui.QWidget()
        self.v = QtGui.QGraphicsView()
        self.w.setLayout(QtGui.QVBoxLayout())
        self.info_layout = QtGui.QHBoxLayout()

        self.info_label = QtGui.QLabel('INFO: ')
        self.info_layout.addWidget(self.info_label)

        self.score_label = QtGui.QLabel('')
        self.info_layout.addWidget(self.score_label)

        self.score_d_label = QtGui.QLabel('')
        self.info_layout.addWidget(self.score_d_label)

        self.score_o_label = QtGui.QLabel('')
        self.info_layout.addWidget(self.score_o_label)

        self.score_s_label = QtGui.QLabel('')
        self.info_layout.addWidget(self.score_s_label)

        self.score_m_label = QtGui.QLabel('')
        self.info_layout.addWidget(self.score_m_label)

        self.others_label = QtGui.QLabel('')
        self.info_layout.addWidget(self.others_label)

        self.w.layout().addLayout(self.info_layout)
        self.w.layout().addWidget(self.v)


        # self.scene = QtGui.QGraphicsScene()
        self.scene = MyScene()
        self.scene.clicked.connect(self.scene_clicked)

        self.v.setScene(self.scene)

        self.used_rows = {}
        self.positions = {}
        self.node_displayed = {}
        self.node_size = NODE_SIZE
        self.y_step = self.node_size + 2
        self.x_step = self.node_size + 150

        self.edge_pen_dist = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen_dist.setColor(QtGui.QColor(0, 0, 0, 0x38))
        self.edge_pen_dist.setWidth(1)

        self.edge_pen_similarity = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen_similarity.setColor(QtGui.QColor(0, 0, 255, 0x68))
        self.edge_pen_similarity.setWidth(2)

        self.edge_pen_strong = QtGui.QPen(QtCore.Qt.SolidLine)
        self.edge_pen_strong.setColor(QtGui.QColor(0, 180, 0, 0x68))
        self.edge_pen_strong.setWidth(2)

        self.edge_pen_merged = QtGui.QPen(QtCore.Qt.DashLine)
        self.edge_pen_merged.setColor(QtGui.QColor(255, 0, 0, 0x68))
        self.edge_pen_merged.setWidth(2)

        self.edge_pen_split = QtGui.QPen(QtCore.Qt.DashLine)
        self.edge_pen_split.setColor(QtGui.QColor(0, 255, 0, 0x68))
        self.edge_pen_split.setWidth(2)

        self.r_color = (0, 255, 0, 0.35)
        self.availability = np.zeros(len(regions))

        self.edges_obj = {}
        self.show_frames_number = True

    def scene_clicked(self, click_pos):
        item = self.scene.itemAt(click_pos)
        for j in [-1, 1, -2, 2]:
            if not item:
                item = self.scene.itemAt(QtCore.QPointF(click_pos.x(), click_pos.y()-j))

        if item and isinstance(item, QtGui.QGraphicsLineItem):
            item.setSelected(True)
            e_ = self.edges_obj[item]

            e = self.G[e_[0]][e_[1]]
            prec = 7
            self.score_label.setText('score: '+str(e['score'])[0:prec])
            self.score_d_label.setText('dist_s: '+str(e['d'])[0:prec])
            self.score_o_label.setText('orient_s: '+str(e['o'])[0:prec])
            self.score_s_label.setText('overlap_s: '+str(e['s'])[0:prec])
            self.score_m_label.setText('minI_s: '+str(e['m'])[0:prec])
            val = np.linalg.norm(e_[0].centroid()-e_[1].centroid())
            self.others_label.setText(str(val)[0:prec])

            print "score: ", e['score'], "d: ", e['d'], "o: ", e['o'], "s: ", e['s'], "m: ", e['m'], e_[0].min_intensity_, e_[1].min_intensity_


    def get_nearest_free_slot(self, t, pos):
        if t in self.used_rows:
            step = 0
            while True:
                test_pos = pos-step
                if test_pos > -1 and test_pos not in self.used_rows[t]:
                    self.used_rows[t][test_pos] = True
                    return test_pos
                if pos+step not in self.used_rows[t]:
                    self.used_rows[t][pos+step] = True
                    return pos+step

                step += 1
        else:
            self.used_rows[t] = {pos: True}
            return pos

    def show_node_with_edges(self, n, prev_pos=0, with_descendants=True):
        if n in self.node_displayed or n not in self.G.node:
            return

        self.node_displayed[n] = True

        t = self.G.node[n]['t']

        if n in self.positions:
            pos = self.positions[n]
        else:
            pos = self.get_nearest_free_slot(t, prev_pos)
            self.positions[n] = pos

        vis = self.G.node[n]['img']
        if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
            vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)
        else:
            z = np.zeros((self.node_size, self.node_size, 3), dtype=np.uint8)
            z[0:vis.shape[0], 0:vis.shape[1]] = vis
            vis = z

        it = self.scene.addPixmap(cvimg2qtpixmap(vis))
        it.setPos(self.x_step * t, self.y_step * pos)

        if with_descendants:
            edges = self.G.out_edges(n)
            for e in edges:
                self.show_node_with_edges(e[1], prev_pos=pos, with_descendants=with_descendants)
                self.draw_edge(n, e[1])


    def draw_edge(self, n1, n2):
        t1 = self.G.node[n1]['t']
        t2 = self.G.node[n2]['t']

        from_x = self.x_step * t1 + self.node_size
        to_x = self.x_step * t2

        from_y = self.y_step * self.positions[n1] + self.node_size/2
        to_y = self.y_step * self.positions[n2] + self.node_size/2

        line_ = QtGui.QGraphicsLineItem(from_x, from_y, to_x, to_y)
        if self.G[n1][n2]['type'] == 's':
            line_.setPen(self.edge_pen_strong)

            for t in range(t1, t2):
                if t not in self.used_rows:
                    self.used_rows[t] = {self.positions[n1]: True}
                else:
                    self.used_rows[t][self.positions[n1]] = True

                self.availability[t] += 1

            c_ = QtGui.QGraphicsEllipseItem(to_x-2, to_y-2, 4, 4)
            c_.setPen(self.edge_pen_strong)
            self.scene.addItem(c_)

            c_ = QtGui.QGraphicsEllipseItem(from_x-2, from_y-2, 4, 4)
            c_.setPen(self.edge_pen_strong)
            self.scene.addItem(c_)

        elif self.G[n1][n2]['type'] == CONFIRMED:
            line_.setPen(self.edge_pen_similarity)
        elif self.G[n1][n2]['type'] == MERGED:
            line_.setPen(self.edge_pen_merged)
        elif self.G[n1][n2]['type'] == SPLIT:
            line_.setPen(self.edge_pen_split)
        else:
            line_.setPen(self.edge_pen_dist)

        self.scene.addItem(line_)
        line_.setFlag(QtGui.QGraphicsItem.ItemIsSelectable, True)

        self.edges_obj[line_] = (n1, n2)

    def prepare_positions(self, frames):
        for f in frames:
            for n1 in self.regions[f]:
                if n1 not in self.G.node:
                    continue

                if n1 in self.positions:
                    continue

                for _, n2, d in self.G.out_edges(n1, data=True):
                    if d['type'] == STRONG:
                        if n2 in self.positions:
                            continue

                        t1 = self.G.node[n1]['t']
                        t2 = self.G.node[n2]['t']

                        p1 = self.get_nearest_free_slot(t1, 0)
                        p2 = self.get_nearest_free_slot(t2, p1)

                        self.positions[n1] = p1
                        self.positions[n2] = p2

                        for t in range(t1+1, t2):
                            if t in self.used_rows:
                                self.used_rows[t][p1] = True
                            else:
                                self.used_rows[t] = {p1: True}

    def visualize(self):
        k = np.array(self.regions.keys())
        frames = np.sort(k)

        # self.prepare_positions(frames)

        for f in frames:
            for r in self.regions[f]:
                self.show_node_with_edges(r)

            if self.show_frames_number:
                t_ = QtGui.QGraphicsTextItem(str(f))

                t_.setPos(self.x_step * f + self.node_size*0.3, -20)
                self.scene.addItem(t_)

        return self.w

def get_cc(G, n):
    t1 = G.node[n]['t']
    s_t1 = set()
    s_t2 = set()

    process = [n]

    while True:
        if not process:
            break

        n_ = process.pop()
        t_ = G.node[n_]['t']

        s_test = s_t2
        if t_ == t1:
            s_test = s_t1

        if n_ in s_test:
            continue

        s_test.add(n_)

        if t_ == t1:
            for _, n2 in G.out_edges(n_):
                process.append(n2)
        else:
            for n2, _ in G.in_edges(n_):
                process.append(n2)

    return list(s_t1), list(s_t2)


def g_add_frame(G, frame, regions, prev_nodes, max_speed=MAX_SPEED):
    for r in regions:
        G.add_node(r, t=frame)

    for r in regions:
        for prev_r in prev_nodes:
            d = np.linalg.norm(r.centroid() - prev_r.centroid())

            if d < max_speed:

                d = -d_lhist_score(G, log_hists, prev_r, r)
                s = -s_lhist_score(log_hists, prev_r, r)
                o = -o_lhist_score(log_hists, prev_r, r)
                m = -m_lhist_score(log_hists, prev_r, r)

                G.add_edge(prev_r, r, type='d', score=d+s+o+m, d=d, s=s, o=o, m=m)


def create_g(num_frames, vid, bg_model=None):
    G = nx.DiGraph()

    prev_nodes = []
    regions = {}

    for f in range(num_frames):
        # msers = select_msers_cached(f)
        im = vid.move2_next()
        if bg_model:
            im = bg_model.get_model().bg_subtraction(im)

        msers = select_msers(im)
        regions[f] = msers

        g_add_frame(G, f, msers, prev_nodes)
        prev_nodes = msers

    return G, regions

def test_similarity(g, max_loss):
    for n in g.nodes():
        edges = g.out_edges(n)
        for e in edges:
            if (n.area() - e[1].area()) / float(n.area()) < 0.5:
                if abs(n.min_intensity_ - e[1].min_intensity_) < 15:
                    s = similarity_loss(n, e[1])

                    if s < max_loss:
                        g[n][e[1]]['type'] = 'sim'


def simplify(G, rules):
    queue = G.nodes()

    while queue:
        n = queue.pop()
        for r in rules:
            affected = r(G, n)
            queue.extend(affected)

def confirmed_rule(G, n):
    if G.out_degree(n) == 1:
        _, n_, d = G.out_edges(n, data=True)[0]
        if G.in_degree(n_) == 1 and d['score'] < SCORE_CONFIRM_THRESH:
            G[n][n_]['type'] = CONFIRMED

    return []


def get_configurations(G, nodes1, nodes2, c, s, configurations, conf_scores):
    if nodes1:
        n1 = nodes1.pop(0)
        for i in range(len(nodes2)):
            n2 = nodes2.pop(0)
            if n2 in G[n1]:
                get_configurations(G, nodes1, nodes2, c + [(n1, n2)], s+G[n1][n2]['score'], configurations, conf_scores)
            nodes2.append(n2)

        # undefined state
        get_configurations(G, nodes1, nodes2, c + [(n1, None)], s + UNDEFINED_EDGE_THRESH, configurations, conf_scores)

        nodes1.append(n1)
    else:
        configurations.append(c)
        conf_scores.append(s)

def cc_optimization(G, nodes1, nodes2):
    configurations = []
    conf_scores = []

    get_configurations(G, nodes1, nodes2, [], 0, configurations, conf_scores)

    if len(conf_scores) == 0:
        return [], []

    ids = np.argsort(conf_scores)

    c = 1 + 1.0/len(nodes1)
    prev = conf_scores[ids[0]]

    final_s = [prev]
    final_c = [configurations[ids[0]]]


    for id in ids:
        s = conf_scores[id]

        if prev < 0:
            if prev > c*s:
                break
        else:
            if prev < c*s:
                break


        prev = s
        final_s.append(s)
        final_c.append(configurations[id])

    return final_s, final_c


def cc_solver(G, n):
    if G.out_degree(n) > 1:
        for _, n2 in G.out_edges(n):
            if G[n][n2]['type'] == MERGED or G[n][n2]['type'] == SPLIT:
                return []

    s1, s2 = get_cc(G, n)
    if len(s1) > 1 or len(s2) > 1:
        scores, configs = cc_optimization(G, s1, s2)

        if len(scores) == 1:
            for n1, n2 in configs[0]:
                # undefined state
                if not n2:
                    continue

                for _, n2_ in G.out_edges(n1):
                    if n2_ != n2:
                        G.remove_edge(n1, n2_)

                for n1_, _ in G.in_edges(n2):
                    if n1_ != n1:
                        G.remove_edge(n1_, n2)

        else:
            print scores, configs

    return []


def merge_detector(G, n):
    if G.in_degree(n) > 1:
        merge = True
        for n1, _ in G.in_edges(n):
            if G.out_degree(n1) != 1:
                merge = False
                break

        if merge:
            for n1, _ in G.in_edges(n):
                G[n1][n]['type'] = MERGED


    return []


def split_detector(G, n):
    if G.out_degree(n) > 1:
        split = True

        for _, n2 in G.out_edges(n):
            if G.in_degree(n2) != 1:
                split = False
                break

        if split:
            for _, n2 in G.out_edges(n):
                G[n][n2]['type'] = SPLIT

    return []

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    start = time.time()

    try:
        if not CACHE:
            raise Exception

        with open(working_dir+'/g'+str(n_frames)+'.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            g = up.load()
            regions = up.load()
    except:
        bg_model = None
        if USE_BG_SUB:
            bg_model = MaxIntensity(vid_path)
            bg_model.compute_model()
            bg_model.bg_model = gaussian_filter(bg_model.bg_model, sigma=3)

        vid = get_auto_video_manager(vid_path)
        g, regions = create_g(n_frames, vid, bg_model)
        vid = get_auto_video_manager(vid_path)

        for frame in regions:
            im = vid.move2_next()
            for r in regions[frame]:
                vis = draw_points_crop(im, r.pts(), square=True, color=(0, 255, 0, 0.35))
                g.node[r]['img'] = vis

        with open(working_dir+'/g'+str(n_frames)+'.pkl', 'wb') as f:
            p = pickle.Pickler(f)
            p.dump(g)
            p.dump(regions)

    end = time.time()

    print "GRAPH CREATED, takes ", end - start, " seconds which is ", (end-start) / n_frames, " seconds per frame"
    start = time.time()
    # simplify(g, [merge_detector, split_detector, cc_solver, confirmed_rule])
    simplify(g, [cc_solver, confirmed_rule])
    end = time.time()
    print "SIMPLIFIED, takes ", end - start, " seconds which is ", (end-start) / n_frames, " seconds per frame"

    imgs = {}
    vid = get_auto_video_manager(vid_path)

    ngv = NodeGraphVisualizer(g, imgs, regions)

    w = ngv.visualize()

    w.showMaximized()

    app.exec_()
    sys.exit()
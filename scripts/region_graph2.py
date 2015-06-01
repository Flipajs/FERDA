__author__ = 'fnaiser'

import networkx as nx
import matplotlib.pyplot as plt
from utils.img import get_safe_selection
from utils.drawing.points import draw_points_crop, draw_points
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
from core.antlikeness import Antlikeness
import assignment_svm
import ant_number_svm
import cProfile
import multiprocessing as mp
from gui.correction.certainty import CertaintyVisualizer
from core.graph.solver import Solver

# max speed of #px / frame
MAX_SPEED = 200
MIN_AREA = 50

AVG_AREA = 240.0
AVG_MARGIN = 0.0
UNDEFINED_POS_SCORE = 0.1
UNDEFINED_EDGE_THRESH = -0.5
UNDEFINED_EDGE_THRESH = 0
WEAK_OVERLAP_THRESH = -1.2

MIN_I_DISTANCE_COEF = 20.0

USE_UNDEFINED = True
S_THRESH = 0.8


CERT_THRESHOLD = 0.5

CACHE_IMGS = True


SIMILARITY = 'sim'
STRONG = 's'
CONFIRMED = 'c'
MERGED = 'm'
SPLIT = 'split'

TSD_CONFIRM_THRESH = 30
TSDI_CONFIRM_THRESH = UNDEFINED_EDGE_THRESH
# SCORE_CONFIRM_THRESH = UNDEFINED_EDGE_THRESH
SCORE_CONFIRM_THRESH = -0.3
ZERO_EPS_THRESHOLD = -0.1

USE_BG_SUB = False

# with open('/Users/fnaiser/Documents/graphs/log_hists.pkl', 'rb') as f:
#     log_hists = pickle.load(f)

from configs.eight import *

# from configs.noplaster import *
# from configs.colormarks1 import *
# from configs.colormarks2 import *
# from configs.bilenses1 import *
# from configs.colony1 import *

USE_BG_SUB = False

CACHE = False
n_frames = 1500

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

    d = np.linalg.norm(n1.centroid() + pred - n2.centroid()) / float(AVG_MAIN_A)
    #
    # bins = hists['distances']['bins']
    # data = hists['distances']['data']
    #
    # return get_hist_val(data, bins, d)

    return max(0, 2-d)


def o_lhist_score(hists, n1, n2):
    t1 = n1.theta_
    t2 = n2.theta_

    if t1 < 0:
        t1 += np.pi
    if t2 < 0:
        t2 += np.pi

    t_ = max(t1, t2) - min(t1, t2)

    if t_ > np.pi/2:
        t_ = np.pi - t_

    bins = hists['thetas']['bins']
    data = hists['thetas']['data']

    return get_hist_val(data, bins, t_)


def s_lhist_score(hists, n1, n2):
    s = abs(n1.area() - n2.area()) / float(min(n1.area(), n2.area()))
    # if s < .5:
    #     if n1.area() < n2.area():
    #         s = similarity_loss(n2, n1)
    #     else:
    #         s = similarity_loss(n1, n2)
    # else:
    #     s *= 2

    return get_hist_val(hists['similarities']['data'], hists['similarities']['bins'], s)


def m_lhist_score(hists, n1, n2):
    m = n1.min_intensity_ - n2.min_intensity_

    return get_hist_val(hists['minI']['data'], hists['minI']['bins'], m-1)

def m_intensity_weight(n1, n2):
    m = n1.min_intensity_ - n2.min_intensity_

    if abs(m) < 10:
        return 1
    else:
        return 0.1


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


def select_msers_cached(frame, use_area_filter=True, use_sort=True):
    msers = get_all_msers(frame, vid_path, working_dir)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    if use_area_filter:
        ids = area_filter(msers, ids, MIN_AREA)

    ids = children_filter(msers, ids)

    if use_sort:
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

# def simplify_g(G):
#     for n in G.nodes():
#         in_num, in_n = num_strong_in_edges(G, n)
#         out_num, out_n = num_strong_out_edges(G, n)
#
#         if out_num == 1 and in_num == 1:
#             G.remove_node(n)
#             G.add_edge(in_n, out_n, type=SIMILARITY)

# def simplify_g(G):
#     for n in G.nodes():
#         if G.in_degree(n) == 1 and G.out_degree(n) == 1:
#             in_e_ = G.in_edges(n)
#             out_e_ = G.out_edges(n)
#
#             G.remove_node(n)
#
#             G.add_edge(in_e_[0][0], out_e_[0][1], type='s')

def get_chunk(G, n, n2=None):
    ch = []

    if n2:
        while True:
            c = n.centroid()
            ch.append([c[0], c[1], n.frame_, n.area()])
            e_ = G.out_edges([n])
            if e_ and n != n2:
                n = e_[0][1]
            else:
                break
    else:
        ch = [[n.centroid()[0], n.centroid()[1], n.frame_, n.area()]]

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

        self.availability = np.zeros(len(regions))

        self.edges_obj = {}
        self.nodes_obj = {}
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

            score = e['score']
            if abs(score) < 0.0001:
                score = 0

            d = e['d']
            if abs(d) < 0.0001:
                d = 0

            o = e['o']
            if abs(o) < 0.0001:
                o = 0

            s = e['s']

            if abs(s) < 0.0001:
                s = 0

            m = e['m']
            if abs(m) < 0.0001:
                m = 0

            self.score_label.setText('score: '+str(score))
            self.score_d_label.setText('dist_s: '+str(d)[0:prec])
            self.score_o_label.setText('multi: '+str(o)[0:prec])
            if 'threshold' in e:
                self.score_s_label.setText('a_thresh: '+str(e['threshold'])[0:prec])
            else:
                self.score_s_label.setText('overlap_s: '+str(s)[0:prec])
            self.score_m_label.setText('antlike: '+str(m)[0:prec])
            # val = np.linalg.norm(e_[0].centroid()-e_[1].centroid())

            cert = -1
            if 'certainty' in e:
                cert = e['certainty']

            if abs(cert) < 0.0001:
                cert = 0

            self.others_label.setText('certainty: '+ str(cert)[0:prec])

        if item and isinstance(item, QtGui.QGraphicsPixmapItem):
            n = self.nodes_obj[item]
            self.score_label.setText('pos: '+str(n.centroid()[0])+', '+str(n.centroid()[1]))
            c_len = len(n.contour())
            self.score_d_label.setText('contour len: '+str(c_len))
            self.score_o_label.setText('clen/area: '+str(c_len/float(n.area())))
            # self.score_s_label.setText('antlikeness: '+str(self.G.node[n]['antlikeness']))
            self.score_m_label.setText('')


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

        t = n.frame_

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
        self.nodes_obj[it] = n

        # if with_descendants:
        #     edges = self.G.out_edges(n)
        #     for e in edges:
        #         self.show_node_with_edges(e[1], prev_pos=pos, with_descendants=with_descendants)
        #         self.draw_edge(n, e[1])


    def draw_edge(self, n1, n2):
        t1 = n1.frame_
        t2 = n2.frame_

        from_x = self.x_step * t1 + self.node_size
        to_x = self.x_step * t2

        from_y = self.y_step * self.positions[n1] + self.node_size/2
        to_y = self.y_step * self.positions[n2] + self.node_size/2

        line_ = QtGui.QGraphicsLineItem(from_x, from_y, to_x, to_y)
        if self.G[n1][n2]['type'] == SIMILARITY:
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
                    if d['type'] == SIMILARITY:
                        if n2 in self.positions:
                            continue

                        t1 = n1.frame_
                        t2 = n2.frame_

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
        self.prepare_positions(frames)

        nodes_queue = []

        visited = {}
        for f in frames:
            for r in self.regions[f]:
                if r in visited or r not in self.G.node:
                    continue

                temp_queue = [r]

                while True:
                    if not temp_queue:
                        break

                    n = temp_queue.pop()
                    if n in visited:
                        continue

                    visited[n] = True
                    nodes_queue.append(n)
                    for e_ in self.G.out_edges(n):
                        temp_queue.append(e_[1])

            if self.show_frames_number:
                t_ = QtGui.QGraphicsTextItem(str(f))

                t_.setPos(self.x_step * f + self.node_size*0.3, -20)
                self.scene.addItem(t_)

        for n in nodes_queue:
           self.show_node_with_edges(n)

        for e in self.G.edges():
            self.draw_edge(e[0], e[1])

        return self.w

def g_add_frame(G, frame, regions, prev_nodes, max_speed=MAX_SPEED):
    for r in regions:
        G.add_node(r, t=frame)

    for r in regions:
        for prev_r in prev_nodes:
            d = np.linalg.norm(r.centroid() - prev_r.centroid())

            if d < max_speed:
                G.add_edge(prev_r, r)


def create_g(num_frames, vid, bg_model=None):
    G = nx.DiGraph()

    prev_nodes = []
    regions = {}

    r_id = 0

    for f in range(num_frames):
        if f % 100 == 0:
            print f

        msers = select_msers_cached(f)
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
            G[n][n_]['type'] = SIMILARITY

    return []

def weak_overlap(G, n):
    if G.out_degree(n) == 1:
        _, n_, d = G.out_edges(n, data=True)[0]
        if G.in_degree(n_) == 1 and d['d'] + d['o'] < WEAK_OVERLAP_THRESH:
            G[n][n_]['type'] = SIMILARITY

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
        if USE_UNDEFINED:
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


def cc_optimization_2best(G, nodes1, nodes2):
    configurations = []
    conf_scores = []

    get_configurations(G, nodes1, nodes2, [], 0, configurations, conf_scores)

    if len(conf_scores) < 2:
        return conf_scores, configurations

    scores = [0, 0]
    confs = [None, None]
    ids = np.argsort(conf_scores)
    for i in range(2):
        scores[i] = conf_scores[ids[i]]
        confs[i] = configurations[ids[i]]

    return scores, confs

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

def get_best_out(G, n):
    best = 0
    best_n = None
    for _, n2, d in G.out_edges(n, data=True):
        if best > d['score']:
            best = d['score']
            best_n = n2

    return best, best_n

def get_best_in(G, n):
    best = 0
    best_n = None
    for n1, _, d in G.in_edges(n, data=True):
        if best > d['score']:
            best = d['score']
            best_n = n1

    return best, best_n


def get_best_nodes(G, regions, frame, positions):
    nodes = []
    for p in positions:
        nearest_r = None
        nearest_d = 2*AVG_MAIN_A
        for r in regions[frame]:
            if r not in G.node:
                continue

            d = np.linalg.norm(p - r.centroid())
            if d < nearest_d:
                nearest_d = d
                nearest_r = r

        nodes.append(nearest_r)

    return nodes


def get_positions(data):
    positions = []
    for id in range(8):
        positions.append(np.array([data[id]['cy'], data[id]['cx']]))

    return positions

def check_node_consistency(G, frame, prev_nodes, nodes, crumbs, swaps):
    for prev_n, n in zip(prev_nodes, nodes):
        if not prev_n:
            continue
        if not n:
            continue

        if n not in G[prev_n]:
            if G.in_degree(n) == 0:
                crumbs.append((frame, prev_n, n))
            else:
                for e_ in G.in_edges(n):
                    for pn_ in prev_nodes:
                        if e_[0] == pn_:
                            swaps.append((frame, prev_n, n))


def check_gt(G, regions, n_frames, gt_data):
    positions = get_positions(gt_data[0])
    nodes = get_best_nodes(G, regions, 0, positions)

    crumbs = []
    swaps = []

    for i in range(1, n_frames):
        prev_nodes = nodes
        positions = get_positions(gt_data[i])
        nodes = get_best_nodes(G, regions, i, positions)

        check_node_consistency(G, i, prev_nodes, nodes, crumbs, swaps)

    print "CRUMBS: "

    nums = {}
    multiple = 0
    for c in crumbs:
        if c[0] in nums:
            nums[c[0]] += 1
            multiple += 1
        else:
            nums[c[0]] = 1

        print " ", c[0], c[1].id_, c[2].id_


    print "SWAPS: "
    for s in swaps:
        print " ", s[0], s[1].id_, s[2].id_

    print "CRUMBS NUM: ", len(crumbs), multiple, "SWAPS NUM: ", len(swaps)


def simplify_g_antlikeness(G, regions, ant_num):
    for f in regions:
        ant_s = []
        for r in regions[f]:
            if not r in G.node:
                continue

            prob = antlikeness.get_prob(r)
            ant_s.append(prob[1])

        ids = np.argsort(-np.array(ant_s))

        for i, id in zip(range(len(ids)), ids):
            if ant_s[id] < 0.2:
            # if i >= ant_num or ant_s[id] < 0.1:
                G.remove_node(regions[f][id])

def get_assignment_score(r1, r2, pred=0):
    d = np.linalg.norm(r1.centroid() + pred - r2.centroid()) / float(AVG_MAIN_A)
    ds = max(0, (2-d) / 2.0)

    # p1 = ant_num_svm.predict_proba([ant_number_svm.get_x(r1, AVG_AREA, AVG_MARGIN)])
    # p2 = ant_num_svm.predict_proba([ant_number_svm.get_x(r2, AVG_AREA, AVG_MARGIN)])

    q1 = antlikeness.get_prob(r1)
    q2 = antlikeness.get_prob(r2)

    # s = ds * min(p1[0][0], p2[0][0]) * min(q1[1], q2[1])
    antlikeness_diff = 1 - abs(q1[1]-q2[1])
    # s = ds * min(q1[1], q2[1])
    s = ds * antlikeness_diff

    # return s, ds, 0, min(q1[1], q2[1])
    return s, ds, 0, antlikeness_diff

def compute_edges(G):
    for n1, n2 in G.edges():
        s, ds, multi, antlike = get_assignment_score(n1, n2)

        if -s > ZERO_EPS_THRESHOLD:
            G.remove_edge(n1, n2)
        else:
            G[n1][n2]['type'] = 'd'
            G[n1][n2]['score'] = -s
            G[n1][n2]['d'] = ds
            G[n1][n2]['s'] = n2.area()-n1.area()
            G[n1][n2]['o'] = multi
            G[n1][n2]['m'] = antlike


def visualize_nodes(im, r):
    vis = draw_points_crop(im, r.pts(), square=True, color=(0, 255, 0, 0.35))
    cv2.putText(vis, str(r.id_), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.55, (255, 255, 255), 1, cv2.cv.CV_AA)

    return vis


def cc_optimization_2best(G, nodes1, nodes2):
    configurations = []
    conf_scores = []

    get_configurations(G, nodes1, nodes2, [], 0, configurations, conf_scores)

    if len(conf_scores) < 2:
        return conf_scores, configurations

    scores = [0, 0]
    confs = [None, None]
    ids = np.argsort(conf_scores)
    for i in range(2):
        scores[i] = conf_scores[ids[i]]
        confs[i] = configurations[ids[i]]

    return scores, confs

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


def cc_certainty(G, c1, c2):
    configurations = []
    scores = []

    get_configurations(G, c1, c2, [], 0, configurations, scores)
    ids = np.argsort(scores)
    configurations = np.array(configurations)
    scores = np.array(scores)
    configurations = configurations[ids]
    scores = scores[ids]

    n_ = float(len(c1))
    cert = abs(scores[0] / n_)
    if len(scores) > 1:
        cert = abs(scores[0] / n_) * abs(scores[0]-scores[1])

    return cert, configurations, scores

def update_costs(G, n):
    in_d = G.in_degree(n)
    out_d = G.out_degree(n)

    affected = []
    if in_d == 1 and out_d > 0:
        e_ = G.in_edges(n)
        prev_n = e_[0][0]
        pred = n.centroid() - prev_n.centroid()

        for _, n2 in G.out_edges(n):
            s = G[n][n2]['score']

            s2, _, _, _ = get_assignment_score(n, n2, pred)
            s2 = -s2

            if s2 < s:
                G[n][n2]['score'] = s2
                print "better score ", n.id_, n2.id_, pred, s, s2
                affected.append(n)

    elif in_d > 0 and out_d == 1:
        e_ = G.out_edges(n)
        next_n = e_[0][1]
        pred = n.centroid() - next_n.centroid()

        for n1, _ in G.in_edges(n):
            s = G[n1][n]['score']

            s2, _, _, _ = get_assignment_score(n1, n, pred)
            s2 = -s2

            if s2 < s:
                G[n1][n]['score'] = s2
                print "better score next ", n1.id_, n.id_, pred, s, s2
                affected.append(n1)


    return affected


def run_():
    global ant_num_svm, antlikeness
    ant_num_svm = ant_number_svm.get_svm_model()

    app = QtGui.QApplication(sys.argv)

    antlikeness = Antlikeness()
    f_regions = {}
    # imgs_gray = {}
    # vid = get_auto_video_manager(vid_path)
    for f in range(init_frames):
        # im = vid.move2_next()
        f_regions[f] = select_msers_cached(f, use_area_filter=False, use_sort=False)
        # imgs_gray[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    antlikeness.learn(f_regions, classes)

    i = 0
    areas = []
    major_axes = []
    margins = []

    for f in f_regions:
        for r in f_regions[f]:
            if classes[i]:
                areas.append(r.area())
                major_axes.append(r.a_ * 2)
                margins.append(r.margin_)
            i += 1

    areas = np.array(areas)
    major_axes = np.array(major_axes)
    margins = np.array(margins)

    AVG_AREA = np.median(areas)
    AVG_MAIN_A = np.median(major_axes)
    AVG_MARGIN = np.median(margins)
    print "AVG: ", AVG_AREA, AVG_MAIN_A, AVG_MARGIN

    try:
        if not CACHE:
            raise Exception

        start = time.time()
        with open(working_dir+'/g'+str(n_frames)+'.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            g = up.load()
            regions = up.load()

        end = time.time()
        print "LOADING, takes ", end - start, " seconds which is ", (end-start) / n_frames, " seconds per frame"

    except:
        bg_model = None
        if USE_BG_SUB:
            bg_model = MaxIntensity(vid_path)
            bg_model.compute_model()
            bg_model.bg_model = gaussian_filter(bg_model.bg_model, sigma=3)

        vid = get_auto_video_manager(vid_path)
        g, regions = create_g(n_frames, vid, bg_model)

        # with open(working_dir+'/original_g'+str(n_frames)+'.pkl', 'wb') as f:
        #     p = pickle.Pickler(f)
        #     p.dump(g)

    start = time.time()

    solver = Solver(project)
    for frame in range(n_frames):
        solver.add_regions_in_t(regions[frame], frame)

    # pool = mp.Pool()
    # vid = get_auto_video_manager(vid_path)

    # for frame in regions:
    #     im = vid.move2_next()
    #
    #     results = []
    #     for r in regions[frame]:
    #         results.append(pool.apply_async(visualize_nodes, args=(im, r)))
    #
    #     for p, r in zip(results, regions[frame]):
    #         g.node[r]['img'] = p.()

    # for frame in regions:
    #     im = vid.move2_next()
    #
    #     # results = []
    #     # for r in regions[frame]:
    #     #     results.append(pool.apply_async(visualize_nodes, args=(im, r)))
    #
    #     for r in regions[frame]:
    #         g.node[r]['img'] = visualize_nodes(im, r)

    end = time.time()
    print "DRAWING NODES, takes ", end - start, " seconds which is ", (end-start) / n_frames, " seconds per frame"

    start = time.time()
    compute_edges(g)

    end = time.time()
    print "COMPUTING EDGE PRICES, takes ", end - start, " seconds which is ", (end-start) / n_frames, " seconds per frame"
    # ngv_ = NodeGraphVisualizer(g, {}, regions)
    # w_ = ngv_.visualize()
    #
    # w_.showMaximized()

    start = time.time()
    # simplify(g, [merge_detector, split_detector, cc_solver, confirmed_rule])
    # simplify_g_antlikeness(g, regions, ant_num)

    solver.simplify()

    # compressed_g = nx.DiGraph()
    # for n, d in g.nodes(data=True):
    #     compressed_g.add_node(n, d)
    #
    # for n1, n2, d in g.edges(data=True):
    #     compressed_g.add_edge(n1, n2, d)
    #
    # # simplify(g, [cc_stable_marriage_solver, confirmed_rule])
    # # simplify(g, [cc_solver, confirmed_rule])
    # # simplify(g, [weak_overlap])
    # simplify(compressed_g, [adaptive_threshold, symetric_cc_solver, update_costs])
    # simplify(compressed_g, [symetric_cc_solver])
    end = time.time()
    print "SIMPLIFIED, takes ", end - start, " seconds which is ", (end-start) / n_frames, " seconds per frame"

    # simplify(compressed_g, [update_costs])
    # simplify(compressed_g, [adaptive_threshold])
    # simplify(compressed_g, [symetric_cc_solver])

    # simplify_g(compressed_g)

    # # with open('/Users/fnaiser/Documents/chunks/noplast_2262_results.arr', 'rb') as f:
    # with open('/Users/fnaiser/Documents/chunks/eight_1505_results.arr', 'rb') as f:
    #     gt_data = pickle.load(f)
    #
    # check_gt(g, regions, n_frames, gt_data)

    imgs = {}
    # ngv = NodeGraphVisualizer(g, imgs, regions)
    # ngv = NodeGraphVisualizer(compressed_g, imgs, regions)

    ccs = solver.get_ccs()
    print len(ccs)
    ccs = sorted(ccs, key=lambda k: k['certainty'])

    # with open(working_dir+'/certainty_visu.pkl', 'wb') as f:
    #     p = pickle.Pickler(f)
    #     p.dump(compressed_g)
    #     # p.dump(regions)
    #     p.dump(ccs)
    #     p.dump(vid_path)

    cv = CertaintyVisualizer(solver.g, get_auto_video_manager(vid_path))

    i = 0
    for c in ccs:
        if i > 10:
            break
        cv.add_configuration(c)
        i += 1


    cv.show()

    # w = ngv.visualize()
    # w.showMaximized()

    app.exec_()
    sys.exit()

if __name__ == '__main__':
    run_()
    # cProfile.run('print run_(); print')
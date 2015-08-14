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


# max speed of #px / frame
MAX_SPEED = 75
MIN_AREA = 50

AVG_AREA = 240
UNDEFINED_POS_SCORE = 0.1


USE_UNDEFINED = False

def display_images_on_nodes():
    img = cv2.imread('/Users/fnaiser/Documents/bg.jpg')
    img = get_safe_selection(img, 100, 100, 100, 100)

    # create an empty graph
    G = nx.Graph()

    # add three edges
    G.add_edge('A', 'B');
    G.add_edge('B', 'C');
    G.add_edge('C', 'A');
    G.add_edge('D', 'A');
    G.add_edge('E', 'A');

    G['A']['B']['Weight'] = 10
    G['B']['C']['Weight'] = 14
    G['C']['A']['Weight'] = 25
    G['D']['A']['Weight'] = 30
    G['E']['A']['Weight'] = 70


    # There are two plots here:
    # First one for the graph
    # Second one for the corresponding node images
    fig = plt.figure(1)
    ax = plt.subplot(111)
    plt.axis('off')

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    #height and width of the image
    h = 50.0
    w = 50.0
    for each_node in G:
        # figure coordinates
        xx, yy = trans(pos[each_node])
        # axes coordinates
        xa, ya = trans2((xx, yy))

        # this is the image size
        piesize_1 = (300.0 / (h * 80))
        piesize_2 = (300.0 / (w * 80))
        p2_2 = piesize_2 / 2
        p2_1 = piesize_1 / 2
        a = plt.axes([xa, ya, piesize_2, piesize_1])

        #insert image into the node
        G.node[each_node]['image'] = img
        #display it
        a.imshow(G.node[each_node]['image'])
        #turn off the axis from minor plot
        a.axis('off')

    #turn off the axis from major plot
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.show()


def disp_graph_with_images(G, pos, im_height, im_width):
    fig = plt.figure(1)
    ax = plt.subplot(111)
    plt.axis('off')

    nx.draw_networkx(G, pos=pos)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)

    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values()) + im_width
    ymax = cut * max(yy for xx, yy in pos.values()) + im_height

    print xmax, ymax
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    # height and width of the image
    h = 50.0
    w = 50.0
    # for each_node in G:
    #
    # # figure coordinates
    #     print pos[each_node]
    #     xx, yy = trans(pos[each_node])
    #     # axes coordinates
    #     xa, ya = trans2((xx, yy))
    #
    #     # this is the image size
    #     piesize_1 = (300.0 / (h*80))
    #     piesize_2 = (300.0 / (w*80))
    #     p2_2 = piesize_2 / 2
    #     p2_1 = piesize_1 / 2
    #     a = plt.axes([xa - p2_2, ya - p2_1, piesize_2, piesize_1])
    #
    #     #display it
    #     a.imshow(G.node[each_node]['image'])
    #     #turn off the axis from minor plot
    #     a.axis('off')

    #turn off the axis from major plot
    nx.draw_networkx_edges(G, pos)
    plt.axis('off')
    plt.savefig("/Users/fnaiser/Documents/graph.pdf")
    plt.show()


def select_msers(im):
    msers = get_msers_(im)
    groups = get_region_groups(msers)
    ids = margin_filter(msers, groups)

    ids = area_filter(msers, ids, MIN_AREA)
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
                from_x = to_x - x_margin
                from_y = top_offset + y * (height + y_margin) + height / 2

                pen = QtGui.QPen(QtCore.Qt.DashLine)
                pen.setColor(QtGui.QColor(0, 0, 0, 0x88))
                pen.setWidth(1.5)

                line_ = QtGui.QGraphicsLineItem(from_x, from_y, to_x, to_y)
                line_.setPen(pen)
                scene.addItem(line_)
                G.add_edge(region, r)

                if True:
                    x_ = (region.area() - r.area()) / float(AVG_AREA)
                    c_ = int(round(x_))
                    d_ = x_ - c_

                    std_ = 2 / 3.
                    n_ = normpdf(0, 0, std_)
                    p_ = normpdf(d_, 0, std_) / n_

                    G.add_edge(region, r, weight=p_, c=c_)

                    t_ = QtGui.QGraphicsTextItem('a = ' + str(r.area()) + ' p = ' + str(final_area_score(1, region)))
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


class Gt():
    def __init__(self, R1, R2, G):
        self.R1 = R1
        self.R2 = R2

        self.G = G

    def edge_set_score(self, from_r_id, es):
        s = 0
        for to_r_id, e in es.iteritems():
            s += e['weight'] * position_score(self.R1[from_r_id], self.R2[to_r_id])

        return s

    def capacity_restrictions(self):
        pass

    def possible_edges_set(self, it, R2_capacity_restrictions):
        pass

    def position_score(self, r1, r2):
        x = np.linalg.norm(r1.centroid() - r2.centroid())
        n_ = normpdf(0, 0, MAX_SPEED / 3)
        return normpdf(x, 0, MAX_SPEED / 3) / n_

    def configurations_position_scores(self, r1, configuraitons, available_regions):
        if USE_UNDEFINED:
            r_scores = np.zeros(len(available_regions) + 1)
        else:
            r_scores = np.zeros(len(available_regions))

        i = 0
        for r2 in available_regions:
            r_scores[i] = self.position_score(r1, r2)
            i += 1

        if USE_UNDEFINED:
            r_scores[i] = UNDEFINED_POS_SCORE

        scores = []
        for c in configuraitons:
            c = np.array(c)
            ids = np.nonzero(c)
            scores.append(np.prod(r_scores[ids] ** c[ids]))

        return scores

    def final_area_score(self, classes_num, reg):
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

    def node_configurations(self, val, available_regions, active_region, conf, configurations):
        a = 0
        if USE_UNDEFINED:
            a = 1

        if len(available_regions) + a == active_region:
            if val == 0:
                configurations.append(copy(conf))
        else:
            for i in range(val+1):
                conf[active_region] = i
                self.node_configurations(val-i, available_regions, active_region+1, conf, configurations)

    def get_combinations_step(self, configurations, regions, active_set_id, c, combinations):
        if active_set_id == len(configurations):
            combinations.append(copy(c))
        else:
            for i in range(len(configurations[regions[active_set_id]])):
                c[active_set_id] = i
                self.get_combinations_step(configurations, regions, active_set_id + 1, c, combinations)

    def get_combinations(self, r1_regions, configurations):
        c = [0 for i in range(len(configurations))]
        combinations = []

        self.get_combinations_step(configurations, r1_regions, 0, c, combinations)

        return combinations

    def get_hypotheses_scores(self, hyps, r1_regions, available_regions, configurations, configuration_position_scores):
        scores = np.ones(len(hyps))
        i = 0

        r1_regions_len = len(r1_regions)
        for h in hyps:
            assignments = {}
            for j in range(r1_regions_len):
                config_id = h[j]
                r1 = r1_regions[j]

                # print i, config_id, r1, len(configuration_position_scores[r1])

                scores[i] *= configuration_position_scores[r1][config_id]

                k = 0
                # everything except UNDEFINED

                if USE_UNDEFINED:
                    for n in configurations[r1][config_id][:-1]:
                        r2 = available_regions[r1][k]
                        # scores[i] += configuration_position_scores[r1][k]

                        assignments[r2] = assignments.get(r2, 0) + n
                        k += 1
                else:
                    for n in configurations[r1][config_id]:
                        r2 = available_regions[r1][k]
                        # scores[i] += configuration_position_scores[r1][k]

                        assignments[r2] = assignments.get(r2, 0) + n
                        k += 1

            for reg in assignments:
                if assignments[reg] > 0:
                    scores[i] *= self.final_area_score(assignments[reg], reg) ** assignments[reg]

            i += 1

        return scores

    def get_hypotheses(self, H, frame=0):
        """
        H - hypothesis, {region: [identities], ... }
        :param H:
        :return:
        """

        hyps = []

        configurations = {}
        available_regions = {}
        position_scores = {}

        r1_regions = []

        for r, ids in H.iteritems():
            test = G[r]
            available_regions[r] = []
            for r_, e_ in test.iteritems():
                try:
                    self.R2.index(r_)
                    available_regions[r].append(r_)
                except ValueError:
                    pass

            configurations[r] = []

            r1_regions.append(r)

            if USE_UNDEFINED:
                self.node_configurations(len(ids), available_regions[r], 0, [0 for i in range(len(available_regions[r]) + 1)], configurations[r])
            else:
                self.node_configurations(len(ids), available_regions[r], 0, [0 for i in range(len(available_regions[r]))], configurations[r])

            position_scores[r] = self.configurations_position_scores(r, configurations[r], available_regions[r])

        combinations = self.get_combinations(r1_regions, configurations)
        scores = self.get_hypotheses_scores(combinations, r1_regions, available_regions, configurations, position_scores)


        if len(scores) == 0:
            print "0 COMBINATIONS"
            return


        ids_ = np.argsort(-scores)

        self.get_hypotheses_scores([combinations[ids_[0]]], r1_regions, available_regions, configurations, position_scores)

        self.combinations = combinations
        self.r1_regions = r1_regions
        self.available_regions = available_regions
        self.configurations = configurations


        best_h = {}

        i = 0
        H_ = copy(H)
        for c in combinations[ids_[0]]:
            r1 = r1_regions[i]

            j = 0
            for a in configurations[r1][c]:
                if j < len(available_regions[r1]):
                    # print "undefined"
                    r2 = available_regions[r1][j]

                    if r2 not in best_h and a > 0:
                        best_h[r2] = []

                    for k in range(a):
                        best_h[r2].append(H_[r1].pop())

                j += 1

            i += 1

        max_ = len(scores)
        if max_ < 11:
            r_ = range(max_)
        else:
            r_ = range(5)
            r_ += range(len(combinations)-5, len(combinations))

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(QtGui.QLabel(str(frame)))
        for j in r_:
            id = ids_[j]

            b = QtGui.QPushButton(str(j) + ' ' +str(scores[id]))
            b.clicked.connect(partial(self.visualize_configuration, ids_[j]))

            vbox.addWidget(b)

        w_hyps.layout().addLayout(vbox)
        self.visualize_configuration(ids_[0])

        return best_h

    def visualize_configuration(self, c_id):
        i = 0
        pen = QtGui.QPen(QtCore.Qt.DashLine)
        pen.setColor(QtGui.QColor(0, 0, 0, 0x88))
        pen.setWidth(1)

        for r1 in self.r1_regions:
            for e, r2 in edges_[r1]:
                e.setPen(pen)

        pen.setWidth(3)
        for c in self.combinations[c_id]:

            r1 = self.r1_regions[i]
            for e, r2 in edges_[r1]:
                for j in range(len(self.configurations[r1][c])):
                    if self.configurations[r1][c][j] > 0:
                        pen.setWidth(3 * self.configurations[r1][c][j])
                        if j < len(self.available_regions[r1]) and r2 == self.available_regions[r1][j]:


                            e.setPen(pen)

            i += 1

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

if __name__ == '__main__':
    # vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
    # working_dir = '/Users/fnaiser/Documents/graphs'
    vid_path = '/Users/fnaiser/Documents/chunks/NoPlasterNoLid800.m4v'
    working_dir = '/Users/fnaiser/Documents/graphs2'
    AVG_AREA = 150
    MAX_SPEED = 50
    n_frames = 3


    vid = get_auto_video_manager(vid_path)

    selected = []

    visited = {}

    edges_ = {}
    texts_ = {}
    regions = {}
    app = QtGui.QApplication(sys.argv)

    w_hyps = QtGui.QWidget()
    w_hyps.setLayout(QtGui.QHBoxLayout())

    w = QtGui.QWidget()

    v = QtGui.QGraphicsView()

    w.setLayout(QtGui.QVBoxLayout())
    w.layout().addWidget(v)
    scene = MyScene()
    scene.clicked.connect(scene_clicked)
    v.setScene(scene)

    w.showMaximized()
    w_hyps.show()

    G = nx.Graph()
    y_margin = 10
    x_margin = 300
    height = 100
    width = 100

    top_offset = height + x_margin + 40

    pos = {}

    prev_msers = None

    frame_offset = 0
    vid.seek_frame(frame_offset)

    Rs = []
    H0 = [2, 3, 4, 6, 7, 9, 11, 13]


    for frame in range(1, n_frames+2):
        im = vid.next_frame()

        msers = select_msers_cached(frame + frame_offset)

        y = 0
        h_, w_, _ = im.shape
        if h_ <= w_:
            h_ = (width + x_margin) * (h_ / float(w_))
        w_ = width + x_margin
        top_offset = h_ + 40

        im_ = cvimg2qtpixmap(np.asarray(resize(im, (h_, w_)) * 255, dtype=np.uint8))
        it = scene.addPixmap(im_)
        it.setPos(frame * (width + x_margin), 20)

        t_ = QtGui.QGraphicsTextItem('frame = ' + str(frame + frame_offset))
        t_.setPos(frame * (width + x_margin), 0)
        scene.addItem(t_)

        Rs.append(msers)

        im_ = np.copy(im)
        for r in msers:
            node_name = str(frame) + '_' + str(y)
            G.add_node(r)


            # im_crop = draw_points(im_, r.pts())
            im_crop = draw_points_crop(im_, get_contour(r.pts()), color=(0, 0, 255, 0.7), square=True)

            im_crop = np.asarray(resize(im_crop, (height, width)) * 255, dtype=np.uint8)
            G.node[r]['image'] = im_crop

            pos[node_name] = (frame * (width + x_margin), y * (height + y_margin) + top_offset)
            it = scene.addPixmap(cvimg2qtpixmap(im_crop))
            it.setPos(pos[node_name][0], pos[node_name][1])

            regions[it] = r

            add_edges(r, y, frame, prev_msers, scene, G)

            y += 1

        prev_msers = msers

    # gt = Gt(Rs[0], Rs[1], G)
    #
    # R1 = Rs[0]
    # # H = {R1[2]: [0], R1[3]: [1], R1[4]: [2], R1[6]: [3], R1[7]: [4], R1[9]: [5], R1[11]: [6], R1[13]: [7]}
    # # H = {R1[1]: [0], R1[2]: [1], R1[3]: [2], R1[5]: [3], R1[6]: [4], R1[8]: [5], R1[10]: [6], R1[11]: [7]}
    #
    #
    # H = {R1[6]: [0], R1[8]: [1], R1[11]: [2], R1[13]: [3], R1[17]: [4], R1[21]: [5], R1[23]: [6], R1[24]: [7], R1[25]: [8], R1[28]: [9], R1[34]: [10], R1[37]: [11], R1[39]: [12], R1[40]: [13], R1[41]: [14]}
    #
    # hyps = gt.get_hypotheses(H, frame_offset)
    #
    # for i in range(1, n_frames):
    #     print i
    #     gt = Gt(Rs[i], Rs[i+1], G)
    #     hyps = gt.get_hypotheses(hyps, frame_offset+i)


    app.exec_()
    sys.exit()
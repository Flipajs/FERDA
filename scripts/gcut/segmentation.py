from __future__ import print_function
from __future__ import absolute_import
__author__ = 'flipajs'

from . import cyMaxflow
import operator
from .helpers import TWeights, Edges
import numpy as np


INT32_INFINITY = 2**31 - 1


class Segmentation:
    def __init__(self, img):
        self.img = img
        self.edges = None
        self.a_nodes_num = 0
        self.y_nodes_num = 0


        self.bg_median = np.median(img)
        self.bg_median = 128

        # initialization
        self.edges = Edges()
        self.node_tweights = TWeights()

        for p_i in range(self.img.shape[0] * self.img.shape[1]):
            self.node_tweights.add(p_i, 0, 0)


    def middle_dist(self, x, y):
        import math

        # TODDO: must be normalized to img size
        val = math.log(((x-self.img.shape[1]/2)**2 + (y-self.img.shape[0]/2)**2)**0.5 + 1, 3)
        return val

    def bg_dist(self, x, y):
        val = max(0, self.bg_median - self.img[y, x])
        return val / 2

    def segmentation(self):
        p_i = 0

        # ignore last row and col
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                # source_unary = self.middle_dist(j, i)
                source_unary = 0
                sink_unary = self.bg_dist(j, i)
                # sink_unary = 0

                self.node_tweights.plus_weights(p_i, source_unary, sink_unary)

                for a, b in [(0, 1), (1, 0)]:
                    q_i = p_i + b + self.img.shape[1] * a
                    if i + a >= self.img.shape[0] or j + b >= self.img.shape[1]:
                        continue

                    edge_cost, u11, u21 = self.reparametrization_((i, j), (i+a, j+b))

                    self.edges.add(p_i, q_i, edge_cost)
                    # self.edges.add(p_i, q_i, 1)

                    # increase unary terms due to reparametrization
                    self.node_tweights.plus_weights(p_i, u11, 0)
                    self.node_tweights.plus_weights(q_i, u21, 0)

                p_i += 1

        # ---- DO MAX FLOW ----
        new_alpha_ids, flow = self.do_maxflow_()

        import numpy as np

        mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.bool)
        for id in new_alpha_ids:
            i = id / self.img.shape[1]
            j = id % self.img.shape[1]

            mask[i, j] = True

        # import matplotlib.pyplot as plt
        # plt.imshow(mask, interpolation='nearest')
        # plt.ion()
        # plt.show()
        # plt.waitforbuttonpress()

        return mask


    def relabel(self, rel_pxs, new_alpha_ids, alpha, remember_labels=False):
        new_alpha_pxs = []
        old_labelling = {}

        id_px_mapping = {}
        for px, px_id in rel_pxs.iteritems():
            id_px_mapping[px_id] = px

        for id in new_alpha_ids:
            # it means that it is y_node (auxillary)
            if id not in id_px_mapping:
                continue

            px = id_px_mapping[id]

            if remember_labels:
                old_labelling[px] = px.label()

            px.set_label(alpha)
            new_alpha_pxs.append(px)

        if remember_labels:
            return new_alpha_pxs, old_labelling

        return new_alpha_pxs

    def add_y_nodes(self, rel_pxs, label_pxs_num):
        self.y_nodes_num = 0

        unique_labels = {}
        for px, p_i in rel_pxs.iteritems():
            unique_labels.setdefault(px.label(), []).append((px, p_i))

        for label, ul in unique_labels.iteritems():
            if self.label_cost[label] > 0:
                k = len(ul)

                # it is not possible to get rid of all pxs with this label
                if k < label_pxs_num[label]:
                    continue

                h = self.label_cost[label]
                h2 = h/2

                y_i = len(rel_pxs) + self.y_nodes_num
                self.y_nodes_num += 1

                self.node_tweights.add(y_i, (h*k)/2 - h, 0)

                for px, p_i in ul:
                    self.node_tweights.plus_weights(p_i, -h2, 0)
                    self.edges.add(p_i, y_i, h2)

    def add_y_node_alpha(self, rel_pxs, alpha):
        # speed up check...
        if self.label_cost[alpha] > 0:
            h = self.label_cost[alpha]

            y_i = len(rel_pxs) + self.y_nodes_num
            self.y_nodes_num += 1

            M2 = self.infinity_substitution/2.0
            self.node_tweights.add(y_i, h - M2 * len(rel_pxs), 0)

            for px, p_i in rel_pxs.iteritems():
                self.node_tweights.plus_weights(p_i, M2, 0)
                self.edges.add(p_i, y_i, M2)

    def do_maxflow_(self):
        nodes_num = self.img.shape[0] * self.img.shape[1]
        edges_num = self.edges.num()

        maxflow = cyMaxflow.PyMaxflow(nodes_num, edges_num, True)

        tw = self.node_tweights
        maxflow.add_multiple_tweights(tw.nodes, tw.capacities_source, tw.capacities_sink)

        maxflow.add_multiple_edges(self.edges.n1, self.edges.n2, self.edges.cost, self.edges.cost)

        flow = maxflow.maxflow()

        return maxflow.get_sink_nodes(), flow

    def compute_unary_cost_(self, px, label, rel_pxs):
        unary_val = 0
        for px_neigh in self.region.pixel_neighbours(px):
            if px_neigh not in rel_pxs:
                unary_val += self.edge_cost_function(label, px_neigh.label())

        return unary_val

    def get_relevant_pixels_(self, alpha):
        labels_px_nums = {}
        relevant_pxs = {}

        i = 0
        alpha_px_exist = False

        for p in self.pixels:
            labels_px_nums[p.label()] = labels_px_nums.setdefault(p.label(), 0) + 1

            if p.label() != alpha:
                if alpha in p.available_labels():
                    relevant_pxs[p] = i
                    i += 1
            else:
                alpha_px_exist = True

        return relevant_pxs, alpha_px_exist, labels_px_nums

    def reparametrization_(self, p, q):
        """
        Does reparametrization that at the end, the v11 = v00 = 0
        and v10 = v01 = new_edge_cost
        :param p:
        :param q:
        :param alpha:
        :return:
        """


        # v11 = self.edge_cost_function(alpha, alpha)
        # v10 = self.edge_cost_function(alpha, q.label())
        # v01 = self.edge_cost_function(p.label(), alpha)
        # v00 = self.edge_cost_function(p.label(), q.label())

        # i1 = int(self.img[p[0], p[1]])
        # i2 = int(self.img[q[0], q[1]])
        import math

        i1 = -math.log((self.img[p[0], p[1]]+1)/256.0)
        i2 = -math.log((self.img[q[0], q[1]]+1)/256.0)

        v11 = 0
        v10 = abs(i1 - i2)
        v01 = abs(i1 - i2)
        v00 = 0

        a = v10 - v00
        b = v01 - v00
        c = v11 - v00

        delta = b - a

        u11 = (c - delta)/2.0
        u21 = (c + delta)/2.0

        new_edge_cost = (v01 + v10 - v00 - v11)/2.0

        return new_edge_cost, u11, u21

    @staticmethod
    def edge_already_visited_(p, q):
        if p.x() > q.x():
            return True
        if p.y() > q.y():
            return True

        return False

    def prepare_edges_(self, rel_pxs, alpha):
        for p, p_i in rel_pxs.iteritems():
            for q in self.region.pixel_neighbours(p):
                if self.edge_already_visited_(p, q):
                    continue

                if q in rel_pxs:
                    edge_cost, u11, u21 = self.reparametrization_(p, q, alpha)

                    q_i = rel_pxs[q]
                    self.edges.add(p_i, q_i, edge_cost)

                    # increase unary terms due to reparametrization
                    self.node_tweights.plus_weights(p_i, u11, 0)
                    self.node_tweights.plus_weights(q_i, u21, 0)




if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    for i in range(1, 1000):
        print(i)

        img = cv2.imread('/Users/flipajs/Desktop/temp/rf/' + str(i) + '_i.png')
        proba_im = cv2.imread('/Users/flipajs/Desktop/temp/rf/' + str(i) + '.png')
        proba_im = proba_im[:, :, 1]
        proba_im.shape = (proba_im.shape[0], proba_im.shape[1])
        s = Segmentation(proba_im)
        mask = s.segmentation()
        # mask = np.asarray(np.logical_not(mask), dtype=np.uint8) * 255

        alpha = 0.5
        overlaycolour = [255, 0, 0]
        for c in range(3):
            img[:, :, c] = np.asarray((1 - alpha) * img[:, :, c] + alpha * mask[:, :] * overlaycolour[c], dtype=np.uint8)

        cv2.imwrite('/Users/flipajs/Desktop/temp/rf/'+str(i)+'_m.png', img)

        # plt.imshow(mask)
        # plt.show()
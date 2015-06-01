__author__ = 'fnaiser'

import networkx as nx
import numpy as np
from core.settings import Settings as S_
from core.graph.graph_utils import *
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_mser_by_id

CONFIRMED = 'c'


class Reduced():
    def __init__(self, region):
        self.centroid = region.centroid()
        self.t = region.frame_
        self.mser_id = region.id_


class Chunk():
    def __init__(self):
        self.reduced = []
        self.start_t = np.inf
        self.end_t = -1
        self.is_sorted = False

    def add_region(self, r):
        self.start_t = min(self.start_t, r.frame_)
        self.end_t = max(self.end_t, r.frame_)

        self.reduced.append(Reduced(r))
        self.is_sorted = False

    def merge(self, chunk):
        self.reduced += chunk.reduced
        self.start_t = min(self.start_t, chunk.start_t)
        self.end_t = max(self.end_t, chunk.end_t)

    def remove_from_beginning(self):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.t)
            self.is_sorted = True

        return self.reduced.pop(0)

    def remove_from_end(self):
        if not self.is_sorted:
            self.reduced = sorted(self.reduced, key=lambda k: k.t)
            self.is_sorted = True

        return self.reduced.pop()


class Configuration():
    def __init__(self, id, regions_t1, regions_t2, certainty, confs, scores):
        self.regions_t1 = regions_t1
        self.regions_t2 = regions_t2

        self.certainty = certainty
        self.configurations = confs
        self.scores = scores
        self.id = id
        self.t = regions_t1[0].frame_


class Solver():
    def __init__(self, project):
        self.g = nx.DiGraph()
        self.project = project
        self.frames = {}
        self.first_frame = np.inf
        self.last_frame = -1
        self.major_axis_median = project.stats.major_axis_median
        self.max_distance = S_.solver.max_edge_distance_in_ant_length * self.major_axis_median
        self.antlikeness = project.stats.antlikeness_svm

        self.antlike_filter = True
        self.rules = [self.adaptive_threshold, self.symmetric_cc_solver, self.update_costs]

        self.cc_id = 0

    def add_regions_in_t(self, regions, t):
        if t not in self.frames:
            self.frames[t] = []

            if len(regions) > 0:
                if self.last_frame < t:
                    self.last_frame = t
                if self.first_frame > t:
                    self.first_frame = t

        for r in regions:
            if self.antlike_filter:
                prob = self.antlikeness.get_prob(r)
                if prob[1] < S_.solver.antlikeness_threshold:
                    continue

            self.g.add_node(r)
            self.frames[t].append(r)

        self.add_edges_to_t(t)

    def add_edges_(self, regions_t1, regions_t2):
        for r_t1 in regions_t1:
            for r_t2 in regions_t2:
                d = np.linalg.norm(r_t1.centroid() - r_t2.centroid())

                if d < self.max_distance:
                    s, ds, multi, antlike = self.assignment_score(r_t1, r_t2)
                    self.g.add_edge(r_t1, r_t2, type='d', score=-s)

    def add_edges_to_t(self, t):
        if t-1 in self.frames:
            self.add_edges_(self.frames[t-1], self.frames[t])

    def simplify(self, queue=None, return_affected=False):
        if queue is None:
            queue = self.g.nodes()

        all_affected = set()

        while queue:
            n = queue.pop()

            #chunk test
            num_out, n_out = num_out_edges_of_type(self.g, n, CONFIRMED)
            if num_out == 1 and 'chunk_ref' in self.g[n][n_out]:
                continue

            for r in self.rules:
                affected = r(n)
                if return_affected:
                    (all_affected.add(x) for x in affected)

                queue.extend(affected)

        return all_affected

    def adaptive_threshold(self, n):
        vals_out, best_out = get_best_n_out_nodes(self.g, n, 2)
        if best_out[0]:
            if self.g[n][best_out[0]]['type'] == CONFIRMED:
                return []
        else:
            return []

        vals_in, best_in = get_best_n_in_nodes(self.g, best_out[0], 2)
        if best_in[0] == n and vals_out[0] < -S_.solver.certainty_threshold:
            cert = -vals_out[0]
            n1 = n
            n2 = best_out[0]
            affected = []
            self.g[n1][n2]['certainty'] = cert
            if best_out[1] or best_in[1]:
                s = self.g[n][best_out[0]]['score']

                s_out = 0
                if best_out[1]:
                    s_out = vals_out[1]

                s_in = 0
                if best_in[1]:
                    s_in = vals_in[1]

                cert = abs(s) * abs(s - (min(s_out, s_in)))
                self.g[n1][n2]['certainty'] = cert

            if cert > S_.solver.certainty_threshold:
                for _, n2_ in self.g.out_edges(n1):
                    if n2_ != n2:
                        self.g.remove_edge(n1, n2_)
                        affected.append(n2_)
                        for n1_, _ in self.g.in_edges(n2_):
                            if n1_ != n:
                                affected.append(n1_)

                for n1_, _ in self.g.in_edges(n2):
                    if n1_ != n1:
                        self.g.remove_edge(n1_, n2)
                        affected.append(n1_)

                self.g[n1][n2]['type'] = CONFIRMED

            return affected

        return []

    def symmetric_cc_solver(self, n):
        s1, s2 = get_cc(self.g, n)

        affected = []
        # TODO:
        # in this case, there might be to much combinations....
        if len(s1) == len(s2) and len(s1) > 1 and len(s1) + len(s2) < 12:
            scores, configs = best_2_cc_configs(self.g, s1, s2)
            if not scores:
                return []


            if len(scores) == 1:
                n_ = float(len(s1))
                # to power of 2 because we want to multiply it by difference to secend best, which is 0
                cert = abs(scores[0] / n_)**2
            else:
                sc1 = scores[0]
                sc2 = scores[1]
                n_ = float(len(s1))
                cert = abs(sc1 / n_) * (abs(sc1-sc2))

            if cert >= S_.solver.certainty_threshold:
                for n1, n2 in configs[0]:
                    if n1 and n2:
                        for _, n2_ in self.g.out_edges(n1):
                            if n2_ != n2:
                                self.g.remove_edge(n1, n2_)
                                affected.append(n2_)

                        for n1_, _ in self.g.in_edges(n2):
                            if n1_ != n1:
                                self.g.remove_edge(n1_, n2)
                                affected.append(n1_)

                        affected.append(n1)
                        affected.append(n2)

                        self.g[n1][n2]['type'] = CONFIRMED
                        self.g[n1][n2]['certainty'] = cert
            else:
                for n1, n2 in configs[0]:
                    if n1 and n2:
                        self.g[n1][n2]['certainty'] = cert

        return affected

    def update_costs(self, n):
        in_d = self.g.in_degree(n)
        out_d = self.g.out_degree(n)

        affected = []
        if in_d == 1 and out_d > 0:
            e_ = self.g.in_edges(n)
            prev_n = e_[0][0]
            pred = n.centroid() - prev_n.centroid()

            for _, n2 in self.g.out_edges(n):
                s = self.g[n][n2]['score']

                s2, _, _, _ = self.assignment_score(n, n2, pred)
                s2 = -s2

                if s2 < s:
                    self.g[n][n2]['score'] = s2
                    # print "better score ", n.id_, n2.id_, pred, s, s2
                    affected.append(n)

        elif in_d > 0 and out_d == 1:
            e_ = self.g.out_edges(n)
            next_n = e_[0][1]
            pred = n.centroid() - next_n.centroid()

            for n1, _ in self.g.in_edges(n):
                s = self.g[n1][n]['score']

                s2, _, _, _ = self.assignment_score(n1, n, pred)
                s2 = -s2

                if s2 < s:
                    self.g[n1][n]['score'] = s2
                    # print "better score next ", n1.id_, n.id_, pred, s, s2
                    affected.append(n1)

        return affected

    def assignment_score(self, r1, r2, pred=0):
        d = np.linalg.norm(r1.centroid() + pred - r2.centroid()) / float(self.major_axis_median)
        ds = max(0, (2-d) / 2.0)

        #TODO: get rid of this hack... also in antlikness test in solver.py
        # flag for virtual region
        if r1.min_intensity_ == -2:
            q1 = 1.0
        else:
            q1 = self.antlikeness.get_prob(r1)[1]

        if r2.min_intensity_ == -2:
            q2 = 1.0
        else:
            q2 = self.antlikeness.get_prob(r2)[1]

        antlikeness_diff = 1 - abs(q1-q2)
        s = ds * antlikeness_diff

        return s, ds, 0, antlikeness_diff

    def simplify_to_chunks(self):
        # for frame in self.frames:
        #     for n in self.frames[frame]:
        for n in self.g.nodes():
                in_num, in_n = num_in_edges_of_type(self.g, n, CONFIRMED)
                out_num, out_n = num_out_edges_of_type(self.g, n, CONFIRMED)

                if out_num == 1 and in_num == 1:
                    if 'chunk_ref' in self.g[in_n][n]:
                        chunk = self.g[in_n][n]['chunk_ref']
                    else:
                        chunk = Chunk()

                    # case when there are 2 chunks (due parallelization) -> MERGE
                    if 'chunk_ref' in self.g[n][out_n]:
                        second_chunk = self.g[n][out_n]['chunk_ref']

                        # chunk.add_region(in_n)
                        # self.g.remove_node(in_n)

                        chunk.merge(second_chunk)

                    chunk.add_region(n)

                    self.g.remove_node(n)
                    self.g.add_edge(in_n, out_n, type=CONFIRMED, chunk_ref=chunk)

    def get_ccs(self, queue=[]):
        if not queue:
            queue = self.g.nodes()

        touched = {}
        ccs = []
        for n in queue:
            if n not in touched:
                out_n, _ = num_out_edges_of_type(self.g, n, CONFIRMED)
                if out_n == 1:
                    touched[n] = True
                    continue

                # We don't want to show last nodes as problem
                if n.frame_ == self.last_frame:
                    continue

                c1, c2 = get_cc(self.g, n)

                for n_ in c1:
                    touched[n_] = True

                #TODO:
                if len(c1) + len(c2) > 12:
                    cert = 0
                    confs = [[]]
                    for r in c1:
                        best_n2 = None
                        best_score = -1
                        for _, n2 in self.g.out_edges(r):
                            if self.g[r][n2]['score'] > best_score:
                                best_n2 = n2
                                best_score = -1

                        confs[0].append((n, best_n2))

                    scores = [0]
                else:
                    cert, confs, scores = cc_certainty(self.g, c1, c2)

                conf = Configuration(self.cc_id, c1, c2, cert, confs, scores)
                self.cc_id += 1

                ccs.append(conf)

        return ccs

    def get_new_ccs(self, all_affected):
        new_ccs = []
        node_representative = []
        touched = {}
        while all_affected:
            n = all_affected.pop()
            if n in touched:
                continue

            node_representative.append(n)

            cc = self.get_ccs([n])
            for c_ in cc:
                for n_ in c_.regions_t1:
                    touched.setdefault(n_, True)

            if len(cc) == 1:
                new_ccs.append(cc[0])
            elif len(cc) > 1:
                print "WARINGN: confirm_edges MULTIPLE ccs"
                raise IndexError
            else:
                new_ccs.append(None)

        return new_ccs, node_representative

    def order_ccs_by_size(self, new_ccs, node_representative):
        cc_sizes = [0 if cc is None else len(cc.regions_t1) for cc in new_ccs]
        ids = np.argsort(-np.array(cc_sizes))
        new_ccs = [new_ccs[id] for id in ids]
        node_representative = [node_representative[id] for id in ids]

        return new_ccs, node_representative

    def confirm_edges(self, edge_pairs):
        affected = []
        for (n1, n2) in edge_pairs:
            affected.append(n1)
            affected.append(n2)

            for _, n2_ in self.g.out_edges(n1):
                if n2_ != n2:
                    self.g.remove_edge(n1, n2_)
                    affected.append(n2_)
                    for n1_, _ in self.g.in_edges(n2_):
                        if n1_ != n1:
                            affected.append(n1_)

            for n1_, _ in self.g.in_edges(n2):
                if n1_ != n1:
                    self.g.remove_edge(n1_, n2)
                    affected.append(n1_)

            self.g[n1][n2]['type'] = CONFIRMED

        all_affected = list(self.simplify(affected[:], return_affected=True))
        all_affected = list(set(all_affected + affected))
        # self.simplify_to_chunks()

        new_ccs, node_representative = self.get_new_ccs(all_affected)

        # order them by size, this will prevent widgets removing when we want update them...
        return self.order_ccs_by_size(new_ccs, node_representative)

    def disassemble_chunk(self, n, chunk, reversed_dir):
        if reversed_dir:
            reduced = chunk.remove_from_end()
        else:
            reduced = chunk.remove_from_beginning()

        vid = get_auto_video_manager(self.project.video_paths)
        img = vid.seek_frame(reduced.t)
        if self.project.bg_model:
            img = self.project.bg_model.bg_subtraction(img)

        if self.project.arena_model:
            img = self.project.arena_model.mask_image(img)

        region = get_mser_by_id(img, reduced.mser_id, reduced.t)

        self.g.add_node(region)

        if reversed_dir:
            self.g.add_edge(n, region, type=CONFIRMED, chunk_ref=chunk)
        else:
            self.g.add_edge(region, n, type=CONFIRMED, chunk_ref=chunk)

        return region

    def merged(self, new_regions, replace, to_fit, t_reversed):
        for n in new_regions:
            self.g.add_node(n)

        to_connect = set()
        if t_reversed:
            for n in replace:
                for n1, _, d in self.g.in_edges(n, data=True):
                    # because if the second node from chunk (n1) is in the consecutive frame, then you just take this region...
                    if 'chunk_ref' in d and n.frame_ - 1 > n1.frame_:
                        n1 = self.disassemble_chunk(n1, d['chunk_ref'], t_reversed)

                    to_connect.add(n1)
        else:
            for n in replace:
                for _, n2, d in self.g.out_edges(n, data=True):
                    if 'chunk_ref' in d and n.frame_ + 1 < n2.frame_:
                        n2 = self.disassemble_chunk(n2, d['chunk_ref'], t_reversed)

                    to_connect.add(n2)

        for n in replace:
            self.g.remove_node(n)



        regions_t1 = new_regions if t_reversed else to_fit
        regions_t2 = to_fit if t_reversed else new_regions
        self.add_edges_(regions_t1, regions_t2)

        affected = list(regions_t1)[:] + list(regions_t2)[:]

        print "MERGED, to connect", to_connect

        regions_t1 = to_connect if t_reversed else new_regions
        regions_t2 = new_regions if t_reversed else to_connect
        self.add_edges_(regions_t1, regions_t2)

        new_ccs, node_representative = self.get_new_ccs(list(affected) + list(regions_t2))
        new_ccs, node_representative = self.order_ccs_by_size(new_ccs, node_representative)

        return [], new_ccs, node_representative

    def add_virtual_region(self, r):
        self.g.add_node(r)
        t = r.frame_

        r_t_minus = []
        r_t_plus = []
        r_t = []

        for n in self.g.nodes():
            if n.frame_ == t-1:
                r_t_minus.append(n)
            elif n.frame_ == t+1:
                r_t_plus.append(n)
            elif n.frame_ == t:
                r_t.append(n)

        self.add_edges_(r_t_minus, [r])
        self.add_edges_([r], r_t_plus)

        new_ccs, node_representative = self.get_new_ccs(r_t_minus + r_t)
        new_ccs, node_representative = self.order_ccs_by_size(new_ccs, node_representative)

        return new_ccs, node_representative

    def is_chunk(self, n):
        # returns (bool, bool, ref) where first is true if node is in chunk and the second returns True if it is t_reversed, and ref is None or reference to chunk
        for n1, _, d in self.g.in_edges(n, data=True):
            if 'chunk_ref' in d:
                return True, True, d['chunk_ref']

        for _, n2, d in self.g.out_edges(n, data=True):
            if 'chunk_ref' in d:
                return True, False, d['chunk_ref']

        return False, False, None

    def remove_region(self, r):
        affected = set()
        for n, _ in self.g.in_edges(r):
            affected.add(n)
            for _, n_ in self.g.out_edges(n):
                affected.add(n_)

        affected = list(affected)
        affected.remove(r)

        is_ch, t_reversed, chunk_ref = self.is_chunk(r)
        if is_ch:
            n = self.disassemble_chunk(r, chunk_ref, t_reversed)

        self.g.remove_node(r)

        new_ccs, node_representative = self.get_new_ccs(affected)
        new_ccs, node_representative = self.order_ccs_by_size(new_ccs, node_representative)

        return new_ccs, node_representative

    def start_nodes(self):
        return self.frames[self.first_frame]

    def end_nodes(self):
        return self.frames[self.last_frame]
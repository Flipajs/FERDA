__author__ = 'fnaiser'

import networkx as nx
import numpy as np
from core.settings import Settings as S_
from core.graph.graph_utils import *
from core.region.mser import get_msers_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from core.settings import Settings as S_
from skimage.transform import rescale
import numpy as np
from chunk import Chunk
from configuration import Configuration
import scipy
from core.log import LogCategories, ActionNames
from utils.img import prepare_for_segmentation
from utils.constants import EDGE_CONFIRMED
import time
import cPickle as pickle
import sqlite3 as sql


class Solver:
    def __init__(self, project):
        self.g = nx.DiGraph()
        self.project = project

        self.start_t = np.inf
        self.end_t = -1

        self.major_axis_median = project.stats.major_axis_median
        self.max_distance = project.solver_parameters.max_edge_distance_in_ant_length * self.major_axis_median
        self.antlikeness = project.stats.antlikeness_svm

        # TODO: add to config
        self.antlike_filter = True
        self.rules = [self.adaptive_threshold, self.symmetric_cc_solver, self.update_costs]
        self.nodes_in_t = {}

        self.ignored_nodes = {}

        self.cc_id = 0

    def add_node(self, n):
        if S_.general.log_graph_edits:
            self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.ADD_NODE, n)
        self.start_t = min(self.start_t, n.frame_)
        self.end_t = max(self.end_t, n.frame_)

        self.g.add_node(n)
        self.nodes_in_t.setdefault(n.frame_, []).append(n)

    def remove_node(self, n, disassembly=True):
        n = self.match_if_reconstructed(n)
        if n is None:
            print "remove node n is None"
            return

        if disassembly:
            is_ch, t_reversed, ch = self.is_chunk(n)
            if is_ch:
                ch.pop_last(self) if t_reversed else ch.pop_first(self)

        # save all edges
        if S_.general.log_graph_edits:
            self.project.log.add_many(self.edges_iter(n))

        self.nodes_in_t[n.frame_].remove(n)
        if not self.nodes_in_t[n.frame_]:
            del self.nodes_in_t[n.frame_]

        self.g.remove_node(n)

        # maybe we need to shrink time boundaries...
        if self.end_t == n.frame_ or self.start_t == n.frame_:
            self.update_time_boundaries()

    def edges_iter(self, n):
        # save all edges
        for n1, n2, d in self.g.in_edges(n, data=True):
            #print LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, pickle.dumps({'n1': n1, 'n2': n2, 'data': d})
            yield (int(time.time()), LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, sql.Binary(pickle.dumps({'n1': n1, 'n2': n2, 'data': d})))

        for n1, n2, d in self.g.out_edges(n, data=True):
            yield (int(time.time()), LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, sql.Binary(pickle.dumps({'n1': n1, 'n2': n2, 'data': d})))

        yield (int(time.time()), LogCategories.GRAPH_EDIT, ActionNames.REMOVE_NODE, pickle.dumps(n))

    def match_if_reconstructed(self, n):
        if n not in self.g:
            return self.find_similar(n)

        return n

    def remove_edge(self, n1, n2):
        n1 = self.match_if_reconstructed(n1)
        n2 = self.match_if_reconstructed(n2)

        if n1 is None or n2 is None:
            if n1 is None:
                print "remove_edge n1 is None, n2: ", n2
            else:
                print "remvoe_edge n2 is None, n1: ", n1
            return

        d = self.g.get_edge_data(n1, n2)

        if S_.general.log_graph_edits:
            self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, {'n1': n1, 'n2': n2, 'data': d})
        self.g.remove_edge(n1, n2)

    def add_edge_fast(self, n1, n2, **data):
        if S_.general.log_graph_edits:
            self.project.log.add(LogCategories.GRAPH_EDIT,
                             ActionNames.ADD_EDGE,
                             {'n1': n1,
                              'n2': n2,
                              'data': data})
        self.g.add_edge(n1, n2, **data)

    def add_edge(self, n1, n2, **data):
        n1 = self.match_if_reconstructed(n1)
        n2 = self.match_if_reconstructed(n2)
        if n1 is None or n2 is None:
            if n1 is None:
                print "add_edge n1 is None, n2: ", n2
            else:
                print "add_edge n2 is None, n1: ", n1
            return

        # if n1 not in self.g.nodes():
        #     print "n1 not in g.nodes"
        #
        # if n2 not in self.g.nodes():
        #     print "n2 not in g.nodes"

        self.add_edge_fast(n1, n2, **data)

    def update_time_boundaries(self):
        self.start_t = np.inf
        self.end_t = -1

        for n in self.g:
            self.start_t = min(self.start_t, n.frame_)
            self.end_t = max(self.end_t, n.frame_)

    def update_nodes_in_t_refs(self):
        self.nodes_in_t = {}
        for n in self.g:
            self.nodes_in_t.setdefault(n.frame_, []).append(n)

        self.update_time_boundaries()

    def get_antlikeness(self, n):
        if n in self.g and 'antlikeness' in self.g.node[n]:
            prob = self.g.node[n]['antlikeness']
        else:
            prob = self.project.stats.antlikeness_svm.get_prob(n)[1]

        return prob

    def add_regions_in_t(self, regions, t, fast=False):
        for r in regions:
            if self.antlike_filter:
                if self.get_antlikeness(r) < self.project.solver_parameters.antlikeness_threshold:
                    continue

            self.add_node(r)

        self.add_edges_to_t(t, fast)

    def is_out_confirmed(self, n):
        # returns bool if there is outcoming confirmed edge from node n
        for _, _, d in self.g.out_edges(n, data=True):
            if 'type' in d and d['type'] == EDGE_CONFIRMED:
                return True

        return False

    def is_in_confirmed(self, n):
        for _, _, d in self.g.in_edges(n, data=True):
            if 'type' in d and d['type'] == EDGE_CONFIRMED:
                return True

        return False

    def add_edges_(self, regions_t1, regions_t2, fast=False):
        for r_t1 in regions_t1:
            for r_t2 in regions_t2:
                d = np.linalg.norm(r_t1.centroid() - r_t2.centroid())

                if d < self.max_distance:
                    if self.is_out_confirmed(r_t1):
                        continue

                    if self.is_in_confirmed(r_t2):
                        continue

                    s, ds, multi, _ = self.assignment_score(r_t1, r_t2)
                    # self.add_edge(r_t1, r_t2)
                    if fast:
                        self.add_edge_fast(r_t1, r_t2, type='d', score=-s)
                    else:
                        self.add_edge(r_t1, r_t2, type='d', score=-s)

    def add_edges_to_t(self, t, fast=False):
        if t-1 in self.nodes_in_t and t in self.nodes_in_t:
            self.add_edges_(self.nodes_in_t[t-1], self.nodes_in_t[t], fast=fast)

    def simplify(self, queue=None, return_affected=False, first_run=False):
        if queue is None:
            queue = self.g.nodes()

        all_affected = set()

        while queue:
            n = queue.pop()

            #chunk test
            num_out, n_out = num_out_edges_of_type(self.g, n, EDGE_CONFIRMED)
            if num_out == 1 and 'chunk_ref' in self.g[n][n_out]:
                continue

            for r in self.rules:
                start = time.time()
                affected = r(n)
                if return_affected:
                    for a in affected:
                        all_affected.add(a)
                        # (all_affected.add(x) for x in affected)
                if not first_run:
                    queue.extend(affected)

        return all_affected

    def adaptive_threshold(self, n):
        vals_out, best_out = get_best_n_out_nodes(self.g, n, 2)
        if best_out[0]:
            if self.g[n][best_out[0]]['type'] == EDGE_CONFIRMED:
                return []
        else:
            return []

        vals_in, best_in = get_best_n_in_nodes(self.g, best_out[0], 2)
        if best_in[0] == n and vals_out[0] < -self.project.solver_parameters.certainty_threshold:
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

            if cert > self.project.solver_parameters.certainty_threshold:
                for _, n2_ in self.g.out_edges(n1):
                    if n2_ != n2:
                        self.remove_edge(n1, n2_)
                        affected.append(n2_)
                        for n1_, _ in self.g.in_edges(n2_):
                            if n1_ != n:
                                affected.append(n1_)

                for n1_, _ in self.g.in_edges(n2):
                    if n1_ != n1:
                        self.remove_edge(n1_, n2)
                        affected.append(n1_)

                self.g[n1][n2]['type'] = EDGE_CONFIRMED

            return affected

        return []

    def get_cc_rec(self, n, depth, node_groups):
        if depth > 10:
            return

        if n.frame_ in node_groups and n in node_groups[n.frame_]:
            return

        node_groups.setdefault(n.frame_, []).append(n)

        for n1, _, d in self.g.in_edges(n, data=True):
            if 'chunk_ref' in d:
                continue

            self.get_cc_rec(n1, depth-1, node_groups)

        for _, n2, d in self.g.out_edges(n, data=True):
            if 'chunk_ref' in d:
                continue

            self.get_cc_rec(n2, depth+1, node_groups)

        # for n1, n2, d in self.g.edges(n, data=True):
        #     if 'chunk_ref' in d:
        #         continue
        #
        #     n_ = n1 if n2 == n else n2
        #
        #     self.get_cc_rec(n_, depth+1, node_groups)

    def get_cc_from_node(self, n):
        node_groups = {}
        self.get_cc_rec(n, 0, node_groups)

        keys = node_groups.keys()
        keys = sorted(keys)

        g = []
        for k in keys:
            g.append(node_groups[k])

        return g

    def find_similar(self, n):
        if n.frame_ in self.nodes_in_t:
            adepts = self.nodes_in_t[n.frame_]
        else:
            return None

        for a in adepts:
            # TODO some reasonable eps
            if np.linalg.norm(n.centroid() - a.centroid()) < 3:
                return a

        return None

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
                # to power of 2 because we want to multiply it by difference to second best, which is 0
                cert = abs(scores[0] / n_)**2
            else:
                sc1 = scores[0]
                sc2 = scores[1]
                n_ = float(len(s1))
                cert = abs(sc1 / n_) * (abs(sc1-sc2))

            if cert >= self.project.solver_parameters.certainty_threshold:
                for n1, n2 in configs[0]:
                    if n1 and n2:
                        for _, n2_ in self.g.out_edges(n1):
                            if n2_ != n2:
                                self.remove_edge(n1, n2_)
                                affected.append(n2_)

                        for n1_, _ in self.g.in_edges(n2):
                            if n1_ != n1:
                                self.remove_edge(n1_, n2)
                                affected.append(n1_)

                        affected.append(n1)
                        affected.append(n2)

                        self.g[n1][n2]['type'] = EDGE_CONFIRMED
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

        if r1.is_virtual:
            q1 = 1.0
        else:
            q1 = self.get_antlikeness(r1)

        if r2.is_virtual:
            q2 = 1.0
        else:
            q2 = self.get_antlikeness(r2)

        antlikeness_diff = 1 - abs(q1-q2)
        s = ds * antlikeness_diff

        return s, ds, 0, antlikeness_diff

    def simplify_to_chunks(self, nodes=None):
        """
        Goes through given nodes and check if outgoing edge is confirmed, if yes, the node is added into chunk.
        If there are no nodes given, it goes trough whole graph.
        """

        if not nodes:
            nodes = self.g.nodes()

        nodes = sorted(nodes, key=lambda k: k.frame_)

        for n in nodes:
            if n not in self.g:
                continue

            out_num, out_n = num_out_edges_of_type(self.g, n, EDGE_CONFIRMED)
            if out_num == 1:
                is_ch, _, chunk = self.is_chunk(n)

                if is_ch:
                    #check if it is the same chunk
                    if chunk.end_n == out_n:
                        continue

                    chunk.append_right(out_n, self)
                else:
                    is_ch_out, _, chunk_out = self.is_chunk(out_n)
                    if is_ch_out:
                        chunk_out.append_left(n, self)
                    else:
                        Chunk(n, out_n, self, store_area=self.project.other_parameters.store_area_info, id=self.project.solver_parameters.new_chunk_id())

    def get_ccs(self, queue=[]):
        if not queue:
            queue = self.g.nodes()

        # sort nodes so the order of ccs is always the same thus the ID of ccs make sense.
        queue = sorted(queue, key=lambda k: (k.frame_, k.id_))

        touched = {}
        ccs = []
        for n in queue:
            if n not in touched:
                out_n, _ = num_out_edges_of_type(self.g, n, EDGE_CONFIRMED)
                if out_n == 1:
                    touched[n] = True
                    continue

                # We don't want to show last nodes as problem
                if n.frame_ == self.end_t:
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

            if n not in self.g:
                new_ccs.append(None)
                continue

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
        affected = set()
        for (n1, n2) in edge_pairs:
            affected.add(n1)
            affected.add(n2)

            for _, n2_ in self.g.out_edges(n1):
                if n2_ != n2:
                    self.remove_edge(n1, n2_)
                    affected.add(n2_)
                    for n1_, _ in self.g.in_edges(n2_):
                        if n1_ != n1:
                            affected.add(n1_)

            for n1_, _ in self.g.in_edges(n2):
                if n1_ != n1:
                    self.remove_edge(n1_, n2)
                    affected.add(n1_)

            # This will happen when there is edge missing (action connect_with_and_confirm)
            # in this case add the edge
            if n2 not in self.g[n1]:
                self.add_edge(n1, n2, score=-1)

            self.g[n1][n2]['type'] = EDGE_CONFIRMED

        # affected = list(affected)
        # all_affected = list(self.simplify(affected[:], return_affected=True))
        # all_affected = list(set(all_affected + affected))

        self.simplify_to_chunks(affected)

    def get_chunk_node_partner(self, n):
        for n_, _, d in self.g.in_edges(n, data=True):
            if 'chunk_ref' in d:
                return n_

        for _, n_, d in self.g.out_edges(n, data=True):
            if 'chunk_ref' in d:
                return n_

        return None

    def split_chunks(self, n, chunk):
        raise Exception("split_chunks in solver.py not implemented yet!!!")
        # _, _, chunk = self.is_chunk(n)

    def merged(self, new_regions, replace, t_reversed):
        for n in new_regions:
            self.add_node(n)

        self.remove_node(replace)

        r_t_minus, r_t, r_t_plus = self.get_regions_around(new_regions[0].frame_)

        # TEST
        for n in new_regions:
            found = False
            for r in r_t:
                if r == n:
                    found = True
                    break

            if not found:
                raise Exception('new regions not found')

        self.add_edges_(r_t_minus, r_t)
        self.add_edges_(r_t, r_t_plus)

    def get_regions_around(self, t):
        # returns (list, list, list) of nodes in t_minus, t, t+plus
        r_t_minus = [] if t-1 not in self.nodes_in_t else self.nodes_in_t[t-1]
        r_t_plus = [] if t+1 not in self.nodes_in_t else self.nodes_in_t[t+1]
        r_t = [] if t not in self.nodes_in_t else self.nodes_in_t[t]

        return r_t_minus, r_t, r_t_plus

    def add_virtual_region(self, r):
        self.add_node(r)
        t = r.frame_

        r_t_minus, r_t, r_t_plus = self.get_regions_around(t)

        self.add_edges_(r_t_minus, [r])
        self.add_edges_([r], r_t_plus)

        new_ccs, node_representative = self.get_new_ccs(r_t_minus + r_t)
        new_ccs, node_representative = self.order_ccs_by_size(new_ccs, node_representative)

        return new_ccs, node_representative

    def is_chunk(self, n):
        # returns (bool, bool, ref) where first is true if node is in chunk and the second returns True if it is t_reversed, and ref is None or reference to chunk
        for n1, _, d in self.g.in_edges(n, data=True):
            if 'chunk_ref' in d:
                if d['chunk_ref'] is None:
                    raise Exception("CHUNK REF IS NONE!")
                return True, True, d['chunk_ref']

        for _, n2, d in self.g.out_edges(n, data=True):
            if 'chunk_ref' in d:
                return True, False, d['chunk_ref']

        return False, False, None

    def remove_region(self, r, strong=False):
        affected = set()
        for n, _ in self.g.in_edges(r):
            affected.add(n)
            for _, n_ in self.g.out_edges(n):
                affected.add(n_)

        for _, n in self.g.out_edges(r):
            affected.add(n)
            for n_, _ in self.g.in_edges(n):
                affected.add(n_)

        affected = list(affected)
        if r in affected[:]:
            affected.remove(r)

        self.remove_node(r)

    def strong_remove(self, r):
        is_ch, t_reversed, ch = self.is_chunk(r)

        if is_ch:
            # TODO: save to log somehow...
            self.remove_node(ch.start_n, False)
            self.remove_node(ch.end_n, False)
        else:
            return self.remove_region(r)

    def start_nodes(self):
        return self.nodes_in_t[self.start_t]

    def end_nodes(self):
        return self.nodes_in_t[self.end_t]

    def chunks_in_frame(self, frame):
        chunks = self.chunk_list()

        in_frame = []
        for ch in chunks:
            if ch.start_t() <= frame <= ch.end_t():
                in_frame.append(ch)

        return in_frame

    def chunk_list(self):
        chunks = []
        for n in self.g:
            for _, _, d in self.g.out_edges(n, data=True):
                if 'chunk_ref' in d:
                    chunks.append(d['chunk_ref'])

        if len(chunks) != len(set(chunks)):
            raise Exception("ERROR in solver... len(chunks) != len(set(chunks))")

        return chunks

    def save(self, autosave=False):
        print "SAVING PROGRESS... Wait please"

        wd = self.project.working_directory

        name = '/progress_save.pkl'
        if autosave:
            name = '/temp/__autosave.pkl'

        with open(wd+name, 'wb') as f:
            pc = pickle.Pickler(f, -1)
            pc.dump(self.g)
            pc.dump(self.project.log)
            pc.dump(self.ignored_nodes)

        print "PROGRESS SAVED"

    def save_progress_only_chunks(self):
        wd = self.project.working_directory

        name = '/progress_save.pkl'

        to_remove = []
        for n in self.g:
            is_ch, t_reversed, ch = self.is_chunk(n)
            if not is_ch or ch.length() < self.project.solver_parameters.global_view_min_chunk_len:
                to_remove.append(n)

        print "NODES", len(self.g)
        S_.general.log_graph_edits = False
        for n in to_remove:
            if n not in self.g:
                continue

            try:
                self.strong_remove(n)
            except:
                pass

        print "NODES", len(self.g)
        S_.general.log_graph_edits = True

        with open(wd+name, 'wb') as f:
            pc = pickle.Pickler(f, -1)
            pc.dump(self.g)
            pc.dump(self.project.log)
            pc.dump(self.ignored_nodes)

        print "ONLY CHUNKS PROGRESS SAVED"
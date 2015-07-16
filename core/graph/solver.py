__author__ = 'fnaiser'

import networkx as nx
import numpy as np
from core.settings import Settings as S_
from core.graph.graph_utils import *
from utils.video_manager import get_auto_video_manager
from core.region.mser import get_mser_by_id, get_msers_
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


class Solver:
    def __init__(self, project):
        self.g = nx.DiGraph()
        self.project = project

        self.start_t = np.inf
        self.end_t = -1

        self.major_axis_median = project.stats.major_axis_median
        self.max_distance = S_.solver.max_edge_distance_in_ant_length * self.major_axis_median
        self.antlikeness = project.stats.antlikeness_svm

        # TODO: add to config
        self.antlike_filter = True
        self.rules = [self.adaptive_threshold, self.symmetric_cc_solver, self.update_costs]
        self.nodes_in_t = {}

        self.cc_id = 0

    def add_node(self, n):
        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.ADD_NODE, n)
        self.start_t = min(self.start_t, n.frame_)
        self.end_t = max(self.end_t, n.frame_)

        self.g.add_node(n)
        self.nodes_in_t.setdefault(n.frame_, []).append(n)

    def remove_node(self, n):
        is_ch, t_reversed, ch = self.is_chunk(n)
        if is_ch:
            ch.pop_last() if t_reversed else ch.pop_first()

        # save all edges
        for n1, n2, d in self.g.in_edges(n, data=True):
            self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, {'n1': n1, 'n2': n2, 'data': d})

        for n1, n2, d in self.g.out_edges(n, data=True):
            self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, {'n1': n1, 'n2': n2, 'data': d})

        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_NODE, n)

        self.nodes_in_t[n.frame_].remove(n)
        if not self.nodes_in_t[n.frame_]:
            del self.nodes_in_t[n.frame_]

        self.g.remove_node(n)

        # maybe we need to shrink time boundaries...
        if self.end_t == n.frame_ or self.start_t == n.frame_:
            self.update_time_boundaries()

    def remove_edge(self, n1, n2):
        d = self.g.get_edge_data(n1, n2)
        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.REMOVE_EDGE, {'n1': n1, 'n2': n2, 'data': d})
        self.g.remove_edge(n1, n2)

    def add_edge(self, n1, n2, **data):
        self.project.log.add(LogCategories.GRAPH_EDIT,
                             ActionNames.ADD_EDGE,
                             {'n1': n1,
                              'n2': n2,
                              'data': data})
        self.g.add_edge(n1, n2, data)

    def update_time_boundaries(self):
        self.start_t = np.inf
        self.end_t = -1

        for n in self.g.nodes():
            self.start_t = min(self.start_t, n.frame_)
            self.end_t = max(self.end_t, n.frame_)

    def update_nodes_in_t_refs(self):
        self.nodes_in_t = {}
        for n in self.g.nodes():
            self.nodes_in_t.setdefault(n.frame_, []).append(n)

        self.update_time_boundaries()

    def add_regions_in_t(self, regions, t):
        for r in regions:
            if self.antlike_filter:
                prob = self.antlikeness.get_prob(r)
                if prob[1] < S_.solver.antlikeness_threshold:
                    continue

            self.add_node(r)

        self.add_edges_to_t(t)

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

    def add_edges_(self, regions_t1, regions_t2):
        for r_t1 in regions_t1:
            if self.is_out_confirmed(r_t1):
                continue

            for r_t2 in regions_t2:
                if self.is_in_confirmed(r_t2):
                    continue

                d = np.linalg.norm(r_t1.centroid() - r_t2.centroid())

                if d < self.max_distance:
                    s, ds, multi, antlike = self.assignment_score(r_t1, r_t2)
                    self.add_edge(r_t1, r_t2, type='d', score=-s)

    def add_edges_to_t(self, t):
        if t-1 in self.nodes_in_t:
            self.add_edges_(self.nodes_in_t[t-1], self.nodes_in_t[t])

    def simplify(self, queue=None, return_affected=False):
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
                affected = r(n)
                if return_affected:
                    for a in affected:
                        all_affected.add(a)
                    # (all_affected.add(x) for x in affected)

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

    def get_cc_from_node(self, n):
        node_groups = []
        out_n, _ = num_out_edges_of_type(self.g, n, EDGE_CONFIRMED)
        if out_n == 1:
            return node_groups

        s1, s2 = get_cc_without_confirmed(self.g, n)
        node_groups.append(s1)
        node_groups.append(s2)

        for i in range(10):
            new_s1 = set()
            new_s2 = set()
            for n2 in s2:
                a1, a2 = get_cc_without_confirmed(self.g, n2)

                for n_ in a2:
                    new_s2.add(n_)

                for n_ in a1:
                    new_s1.add(n_)

            if len(new_s2) <= 0:
                break

            node_groups[len(node_groups)-1] = list(new_s1)
            # node_groups[len(node_groups)-1] = list(set(node_groups[len(node_groups)-1]))
            s2 = list(new_s2)
            node_groups.append(s2)

        for i in range(len(node_groups)):
            node_groups[i] = list(set(node_groups[i]))

        return node_groups

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

            if cert >= S_.solver.certainty_threshold:
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

        #TODO: get rid of this hack... also in antlikness test in solver.py
        # flag for virtual region
        q1 = self.antlikeness.get_prob(r1)[1]
        try:
            if r1.is_virtual:
                q1 = 1.0
        except:
            pass

        q2 = self.antlikeness.get_prob(r2)[1]
        try:
            if r2.is_virtual:
                q2 = 1.0
        except:
            pass

        antlikeness_diff = 1 - abs(q1-q2)
        s = ds * antlikeness_diff

        return s, ds, 0, antlikeness_diff

    def simplify_to_chunks(self, nodes=None):
        if not nodes:
            nodes = self.g.nodes()

        for n in nodes:
            in_num, in_n = num_in_edges_of_type(self.g, n, EDGE_CONFIRMED)
            out_num, out_n = num_out_edges_of_type(self.g, n, EDGE_CONFIRMED)

            if out_num == 1 and in_num == 1:

                self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.ASSEMBLE_CHUNK, {'n': n})

                if 'chunk_ref' in self.g[in_n][n]:
                    chunk = self.g[in_n][n]['chunk_ref']
                else:
                    chunk = Chunk(in_n, n, out_n)

                # case when there are 2 chunks (due parallelization) -> MERGE
                if 'chunk_ref' in self.g[n][out_n]:
                    second_chunk = self.g[n][out_n]['chunk_ref']

                    chunk.merge(second_chunk, self)
                    self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.MERGE_CHUNKS, {'n': n})

                chunk.append_left(n, self.g, self.project.log)

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

            if n not in self.g.nodes():
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

            self.g[n1][n2]['type'] = EDGE_CONFIRMED

        affected = list(affected)
        all_affected = list(self.simplify(affected[:], return_affected=True))
        all_affected = list(set(all_affected + affected))
        self.simplify_to_chunks()

        new_ccs, node_representative = self.get_new_ccs(all_affected)

        # order them by size, this will prevent widgets removing when we want update them...
        return self.order_ccs_by_size(new_ccs, node_representative)

    def get_chunk_node_partner(self, n):
        for n_, _, d in self.g.in_edges(n, data=True):
            if 'chunk_ref' in d:
                return n_

        for _, n_, d in self.g.out_edges(n, data=True):
            if 'chunk_ref' in d:
                return n_

        return None

    def disassemble_chunk(self, n, chunk=None, reversed_dir=None):
        if not chunk:
            _, reversed_dir, chunk = self.is_chunk(n)

        self.project.log.add(LogCategories.GRAPH_EDIT, ActionNames.DISASSEMBLE_CHUNK, {'n': n})

        if reversed_dir:
            reduced = chunk.remove_from_end()
            if not reduced:
                region = self.get_chunk_node_partner(n)
                self.remove_edge(n, region)
        else:
            reduced = chunk.remove_from_beginning()
            if not reduced:
                region = self.get_chunk_node_partner(n)
                self.remove_edge(region, n)

        # if the chunk is only n1, n2  and nothing between in reduced form, the reduced is None and the disassembled region is in region var
        if reduced:
            region = reduced.reconstruct(self.project)

            self.add_node(region)

            if reversed_dir:
                self.add_edge(n, region, type=EDGE_CONFIRMED, chunk_ref=chunk, score=1.0)
                _, _, t_plus = self.get_regions_around(region.frame_)
                self.add_edges_([region], t_plus)
            else:
                self.add_edge(region, n, type=EDGE_CONFIRMED, chunk_ref=chunk, score=1.0)
                t_minus, _, _ = self.get_regions_around(region.frame_)
                self.add_edges_(t_minus, [region])

        return region

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
        if r in affected:
            affected.remove(r)

        is_ch, t_reversed, chunk_ref = self.is_chunk(r)
        if is_ch:
            # get the other end of chunk
            if t_reversed:
                for n1, _ in self.g.in_edges(r):
                    n_ = n1
            else:
                for _, n2 in self.g.out_edges(r):
                    n_ = n2

            if not strong:
                self.disassemble_chunk(n_, chunk_ref, t_reversed)

            affected.append(n_)

        self.remove_node(r)

        new_ccs, node_representative = self.get_new_ccs(affected)
        new_ccs, node_representative = self.order_ccs_by_size(new_ccs, node_representative)

        return new_ccs, node_representative

    def strong_remove(self, r):
        is_ch, t_reversed, ch = self.is_chunk(r)

        if is_ch:
            ch_end_n = self.get_chunk_node_partner(r)

            new_ccs, node_representative = self.remove_region(r, strong=True)
            new_ccs2, node_representative2 = self.remove_region(ch_end_n, strong=True)
            new_ccs += new_ccs2
            node_representative += node_representative2

            new_ccs_ = []
            node_representatives_ = []

            # remove everything from first remove region which doesn't make sense after second removing...
            for cc, n in zip(new_ccs, node_representative):
                if not cc:
                    continue

                ok = True
                for n in cc.regions_t1:
                    if n not in self.g.nodes():
                        ok = False
                        break
                if ok:
                    for n in cc.regions_t2:
                        if n not in self.g.nodes():
                            ok = False
                            break
                if ok:
                    new_ccs_.append(cc)
                    node_representatives_.append(n)

            return new_ccs_, node_representatives_
        else:
            return self.remove_region(r)

    def start_nodes(self):
        return self.nodes_in_t[self.start_t]

    def end_nodes(self):
        return self.nodes_in_t[self.end_t]
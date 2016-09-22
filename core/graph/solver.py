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
import graph_tool
from graph_manager import GraphManager

class Solver:
    def __init__(self, project):
        """
        We are maximizing.
        Score is in the <0,1> range.

        :param project:
        :return:
        """

        self.project = project

        self.major_axis_median = project.stats.major_axis_median

        # TODO: add to config
        self.antlike_filter = True
        self.rules = [self.adaptive_threshold, self.symmetric_cc_solver, self.update_costs]

        self.ignored_nodes = {}

        self.cc_id = 0

    def simplify(self, queue=None, rules=None):
        num_changed = 0

        if queue is None:
            queue = self.project.gm.get_all_relevant_vertices()

        if rules is None:
            rules = self.rules

        # TODO: does it still make sense?
        queue = sorted(queue, key=lambda x: self.project.gm.region(x).area()+self.project.gm.region(x).centroid()[0]+self.project.gm.region(x).frame()+self.project.gm.region(x).centroid()[1])

        while queue:
            vertex = self.project.gm.g.vertex(queue.pop())

            r = self.project.gm.region(vertex)

            # # skip the ...
            # if self.project.gm.chunk_start(vertex):
            #     continue

            for ru in rules:
                # it is necessary to check this here, because it is possible that the vertex will be removed applying one of the rules
                if r.frame() not in self.project.gm.vertices_in_t:
                    continue
                if vertex not in self.project.gm.vertices_in_t[r.frame()]:
                    continue

                affected = ru(vertex)

                if len(affected) > 0:
                    num_changed += 1

                queue.extend(affected)

        return num_changed

    def get_antlikeness(self, n):
        if n.is_virtual:
            return 1.0

        return self.project.stats.antlikeness_svm.get_prob(n)[1]

    def adaptive_threshold(self, vertex):
        if self.project.gm.ch_start_longer(vertex):
            return []

        best_out_scores, best_out_vertices = self.project.gm.get_2_best_out_vertices(vertex)

        if not best_out_vertices[0]:
            return []

        r = self.project.gm.region(vertex)
        if r.frame() == 2191 and (r.area() == 750 or r.area() == 2191):
            print r

        best_in_scores, best_in_vertices = self.project.gm.get_2_best_in_vertices(best_out_vertices[0])
        if best_in_vertices[0] == vertex and best_in_scores[0] >= self.project.solver_parameters.certainty_threshold:
            cert = best_out_scores[0]
            v1 = vertex
            v2 = best_out_vertices[0]
            affected = []

            if (best_out_vertices[1] and best_out_vertices[0] != best_out_vertices[1]) or \
                    (best_in_vertices[1] and best_in_vertices[0] != best_in_vertices[1]):
                s = best_out_scores[0]

                s_out = 0
                if best_out_vertices[1]:
                    s_out = best_out_scores[1]

                s_in = 0
                if best_in_vertices[1]:
                    s_in = best_in_scores[1]

                # r1 = self.project.gm.region(best_in_vertices[0])
                # r2 = self.project.gm.region(best_out_vertices[0])
                #
                # area_coef = abs(r1.area()-r2.area()) / min(r1.area(), r2.area())
                # # hard area rule...
                # if area_coef > 0.5:
                #     return []

                # desc1 = self.zernike_desc.describe(r1)
                # desc2 = self.zernike_desc.describe(r2)
                #
                # desc_correction = self.project.solver_parameters.zernike_plus
                # if np.linalg.norm(desc1-desc2) > self.project.solver_parameters.zernike_thresh:
                #     desc_correction = self.project.solver_parameters.zernike_minus

                desc_correction = 0

                # cert = abs(s) * abs(s - (min(s_out, s_in))) + desc_correction
                cert = abs(s) * abs(s - (min(s_out, s_in))) + desc_correction

            self.project.gm.g.ep['certainty'][self.project.gm.g.edge(v1, v2)] = cert

            if cert > self.project.solver_parameters.certainty_threshold:
                affected = self.confirm_edges([(v1, v2)])

            return affected

        return []



    def match_if_reconstructed(self, n):
        if n not in self.g:
            return self.find_similar(n)

        return n

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

    def symmetric_cc_solver(self, vertex):
        s1, s2 = self.project.gm.get_cc(vertex)

        affected = []
        # TODO:
        # in this case, there might be to much combinations....
        if len(s1) == len(s2) and 1 < len(s1) < 5:
            scores, matchings = self.project.gm.get_2_best_matchings(s1, s2)
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
                for n1, n2 in matchings[0]:
                    if n1 and n2:
                        # for n2_ in n1.out_neighbours():
                        #     if n2_ != n2:
                        #         self.project.gm.remove_edge(n1, n2_)
                        #         affected.append(n2_)
                        #
                        # for n1_ in n2.in_neighbours():
                        #     if n1_ != n1:
                        #         self.remove_edge(n1_, n2)
                        #         affected.append(n1_)
                        #
                        # affected.append(n1)
                        # affected.append(n2)

                        e = self.project.gm.g.edge(n1, n2)
                        self.project.gm.g.ep['certainty'][e] = cert
                        affected += self.confirm_edges([(n1, n2)])
            else:
                for n1, n2 in matchings[0]:
                    if n1 and n2:
                        e = self.project.gm.g.edge(n1, n2)
                        self.project.gm.g.ep['certainty'][e] = cert

        return affected

    def update_costs(self, vertex):
        in_d = vertex.in_degree()
        out_d = vertex.out_degree()

        region = self.project.gm.region(vertex)

        affected = []
        if in_d == 1 and out_d > 0:
            ch, ch_is_end = self.project.gm.is_chunk(vertex)

            pred = None
            # if there is a chunk, we have to obtain region differently
            if ch:
                if ch_is_end and ch.length() > 1:
                    prev_region = self.project.gm.region(ch.nodes_[-2])
                    pred = region.centroid() - prev_region.centroid()
            # else:
            #     # there is just one edge...
            #     pred = 0
            #     for e_ in vertex.in_edges():
            #         prev_region = self.project.gm.region(e_.source())
            #         pred = region.centroid() - prev_region.centroid()

            if pred is not None:
                for e in vertex.out_edges():
                    s = self.project.gm.g.ep['score'][e]

                    out_region = self.project.gm.region(e.target())
                    s2, _, _, _ = self.assignment_score(region, out_region, pred)

                    if s2 > s:
                        self.project.gm.g.ep['score'][e] = s2
                        # affected.append(vertex)

                        # make EMD test
                        s1, s2 = self.project.gm.get_cc(self.project.gm.g.vertex(e.source()))
                        self.dsmc_process_cc_(s1, s2, self.project.stats.area_median)

                        if self.project.gm.g.ep['score'][e] > 0:
                           affected.append(e.source())
                           # affected.append(e.target())

        if in_d > 0 and out_d == 1:
            ch, ch_is_end = self.project.gm.is_chunk(vertex)

            pred = None

            # if there is a chunk, we have to obtain region differently
            if ch:
                if ch.length() > 1 and not ch_is_end:
                    next_region = self.project.gm.region(ch.nodes_[1])
                    pred = region.centroid() - next_region.centroid()
            # else:
            #     # there is only one edge
            #     for e_ in vertex.out_edges():
            #         next_region = self.project.gm.region(e_.target())
            #         pred = region.centroid() - next_region.centroid()

            if pred is not None:
                for e in vertex.in_edges():
                    s = self.project.gm.g.ep['score'][e]

                    in_region = self.project.gm.region(e.source())
                    s2, _, _, _ = self.assignment_score(in_region, region, pred)

                    if s2 > s:
                        self.project.gm.g.ep['score'][e] = s2
                        # affected.append(e.source())

                        # make EMD test
                        s1, s2 = self.project.gm.get_cc(self.project.gm.g.vertex(e.source()))
                        self.dsmc_process_cc_(s1, s2, self.project.stats.area_median)

                        if self.project.gm.g.ep['score'][e] > 0:
                            affected.append(e.source())
                            # affected.append(e.target())

        return affected

    def assignment_score(self, r1, r2, pred=0):
        d = np.linalg.norm(r1.centroid() + pred - r2.centroid()) / float(self.major_axis_median)
        max_d = self.project.solver_parameters.max_edge_distance_in_ant_length
        ds = max(0, (max_d-d) / max_d)

        q1 = self.get_antlikeness(r1)
        q2 = self.get_antlikeness(r2)

        antlikeness_diff = 1 - abs(q1-q2)
        # antlikeness_diff = 1
        s = ds * antlikeness_diff

        if self.project.solver_parameters.use_colony_split_merge_relaxation():
            a1 = r1.area()
            a2 = r2.area()

            if a1 < a2:
                a1, a2 = a2, a1

            area_diff = (a1 - a2)

            # simple split / merge test... Quite strict.
            if area_diff > self.project.stats.area_median * 0.5:
                # when regions are big...
                if a1 < self.project.stats.area_median * 5:
                    if area_diff/float(a1) > 0.15:
                        s = 0
                else:
                    s = 0

        return s, ds, 0, antlikeness_diff

    def assignment_score_pos_orient(self, r1, r2):
        """


        Args:
            r1:
            r2:
            orient_weight:

        Returns:

        """

        d = np.linalg.norm(r1.centroid() - r2.centroid()) / float(self.major_axis_median)
        max_d = self.project.solver_parameters.max_edge_distance_in_ant_length
        ds = max(0, (max_d-d) / max_d)

        dt = (r1.theta_ - r2.theta_) % np.pi
        dt = max(0, (np.pi/2-dt) / (np.pi/2))

        return ds*dt, ds, dt

    def get_ccs(self, queue=[]):
        if not queue:
            queue = self.project.gm.g.nodes()

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

    def confirm_edges(self, edge_pairs):
        """
        for each pair of edges (v1, v2), removes all the edges going from v1.t -> v2.t except for edge v1 -> v2
        if there is a chunk, append (right/left) else create new one
        """

        affected = set()
        for v1, v2 in edge_pairs:
            affected.add(v1)
            affected.add(v2)

            for e in v1.out_edges():
                affected.add(e.target())

                for aff_neigh in e.target().in_neighbours():
                    affected.add(aff_neigh)

                self.project.gm.remove_edge_(e)

            for e in v2.in_edges():
                affected.add(e.source())

                for aff_neigh in e.source().out_neighbours():
                    affected.add(aff_neigh)

                self.project.gm.remove_edge_(e)

            # This will happen when there is an edge missing (action connect_with_and_confirm)
            if not self.project.gm.g.edge(v1, v2):
                self.project.gm.add_edge(v1, v2, 1)

            # test chunk existence, if there is none, create new one.
            v1_ch = self.project.gm.chunk_end(v1)
            v2_ch = self.project.gm.chunk_start(v2)
            if v1_ch:
                v1_ch.append_right(v2, self.project.gm)
            elif v2_ch:
                v2_ch.append_left(v1, self.project.gm)
            else:
                self.project.chm.new_chunk(map(int, [v1, v2]), self.project.gm)

        affected = list(affected)
        # all_affected = list(self.simplify(affected[:], return_affected=True))
        # all_affected = list(set(all_affected + affected))
        return affected

    def merged(self, new_regions, replace, t_reversed=False):
        """
        is called when fitting is finished...
        """
        if not isinstance(replace, list):
            replace = [replace]

        new_vertices = []
        for r in new_regions:
            new_vertices.append(self.project.gm.add_vertex(r))

        r_t_minus = []
        r_t_plus = []

        for r in replace:
            r = self.project.gm.g.vertex(r)
            r_t_minus.extend([v for v in r.in_neighbours()])
            r_t_plus.extend([v for v in r.out_neighbours()])

            self.project.gm.remove_vertex(r)

        r_t_minus = self.project.gm.get_vertices_in_t(new_regions[0].frame_ - 1)
        r_t_plus = self.project.gm.get_vertices_in_t(new_regions[0].frame_ + 1)

        self.project.gm.add_edges_(r_t_minus, new_vertices)
        self.project.gm.add_edges_(new_vertices, r_t_plus)

        self.project.gm.fitting_logger.add(new_vertices, r_t_minus, replace)

        return new_vertices

    def merged_chunk(self, model_vertices, new_regions, replace, t_reversed, chunk):
        """
        if t_reversed = False
        model_regions t-1
        new_regions t
        replace t

        is called when fitting is finished...
        """

        new_vertices = []
        for r in new_regions:
            r.pts_ = np.asarray(np.round(r.pts_), dtype=np.uint32)
            new_vertices.append(self.project.gm.add_vertex(r))

        r_t_minus = []
        r_t_plus = []

        if t_reversed:
            r_t_minus = []
            if chunk.length() == 0:
                end_vertex = self.project.gm.g.vertex(replace)
                for v in end_vertex.in_neighbours():
                    r_t_minus.apend(v)
        else:
            r_t_plus = []
            if chunk.length() == 0:
                start_vertex = self.project.gm.g.vertex(replace)
                for v in start_vertex.out_neighbours():
                    r_t_plus.append(v)

        self.project.gm.remove_vertex(replace)

        if t_reversed:
            r_t_plus = model_vertices
        else:
            r_t_minus = model_vertices

        r_t = new_vertices

        # remove all

        # TEST
        for n in new_regions:
            found = False
            for r in r_t:
                if self.project.gm.region(r) == n:
                    found = True
                    break

            if not found:
                raise Exception('new regions not found')

        self.project.gm.add_edges_(r_t_minus, r_t)
        self.project.gm.add_edges_(r_t, r_t_plus)

        return new_vertices

    def get_vertices_around_t(self, t):
        # returns (list, list, list) of nodes in t_minus, t, t+plus
        v_t_minus = self.project.gm.get_vertices_in_t(t-1)
        v_t = self.project.gm.get_vertices_in_t(t)
        v_t_plus = self.project.gm.get_vertices_in_t(t+1)

        return v_t_minus, v_t, v_t_plus

    def add_virtual_region(self, region):
        self.project.rm.add(region)
        vertex = self.project.gm.add_vertex(region)

        r_t_minus, r_t, r_t_plus = self.get_vertices_around_t(region.frame_)

        self.project.gm.add_edges_(r_t_minus, [vertex])
        self.project.gm.add_edges_([vertex], r_t_plus)

    # def remove_vertex(self, vertex):
    #     affected = []
    #
    #     for v in vertex.all_edges():
    #         affected.append(v)
    #
    #     self.project.gm.remove_vertex(vertex)
    #
    #     return affected

    def strong_remove(self, vertex):
        ch, _ = self.project.gm.is_chunk(vertex)

        if ch:
            affected = []

            affected += self.project.gm.remove_vertex(ch.start_vertex_id(), disassembly=False)
            affected += self.project.gm.remove_vertex(ch.end_vertex_id(), disassembly=False)

            return affected
        else:
            return self.project.gm.remove_vertex(vertex)

    def save(self, autosave=False):
        print "SAVING PROGRESS... Wait please"

        wd = self.project.working_directory

        name = '/progress_save.pkl'
        if autosave:
            name = '/temp/__autosave.pkl'

        self.project.save()

        # with open(wd+name, 'wb') as f:
        #     pc = pickle.Pickler(f, -1)
        #     pc.dump(self.project.gm.g)
        #     pc.dump(self.project.chm)
        #     pc.dump(self.ignored_nodes)

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
            pc.dump(self.ignored_nodes)

        print "ONLY CHUNKS PROGRESS SAVED"

    def dsmc_process_cc_(self, s1, s2, area_med):
        from scripts.EMD import get_unstable_num, detect_stable
        if len(s1) > 1 or len(s2) > 1:
            # edges = set()

            regions_P = []
            for s in s1:
                r = self.project.gm.region(s)
                regions_P.append((r.area(), r.centroid(), s))
                #
                # for e in s.out_edges():
                #     edges.add(e)

            # edges = list(edges)

            regions_Q = []
            for s in s2:
                r = self.project.gm.region(s)
                regions_Q.append((r.area(), r.centroid(), s))

            unstable_num, stability_P, stability_Q, preferences = detect_stable(regions_P, regions_Q, thresh=0.7, area_med=area_med)
            for i, v in enumerate(s1):
                r = regions_P[i]
                for e in v.out_edges():
                    edge_prohibited = True

                    if stability_P[i]:
                        for r2_i, r2 in enumerate(regions_Q):
                            if e.target() == r2[2] and stability_Q[r2_i]:
                                if preferences[r[2]] == r2[2] and preferences[r2[2]] == r[2]:
                                    edge_prohibited = False

                    if edge_prohibited:
                        self.project.gm.g.ep['score'][e] = 0

            # for v, i in zip(s1, range(len(s1))):
            #     if not stability_P[i]:
            #         for e in v.out_edges():
            #             self.project.gm.g.ep['score'][e] = 0
            #
            # for v, i in zip(s2, range(len(s2))):
            #     if not stability_Q[i]:
            #         for e in v.in_edges():
            #             self.project.gm.g.ep['score'][e] = 0

    def detect_split_merge_cases(self, frames=None):
        if frames is None:
            frames = self.project.gm.vertices_in_t
        else:
            frames_ = []
            for t in frames:
                if t in self.project.gm.vertices_in_t:
                    frames_.append(t)

            frames = frames_

        for t in frames:
            if t == 2191:
                print 2191
            vs = [v for v in self.project.gm.vertices_in_t[t]]

            while vs:
                v = vs[0]

                s1, s2 = self.project.gm.get_cc(self.project.gm.g.vertex(v))
                self.dsmc_process_cc_(s1, s2, self.project.stats.area_median)

                for v in s1:
                    vs.remove(v)

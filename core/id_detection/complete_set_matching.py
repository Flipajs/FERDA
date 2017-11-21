import numpy as np


class CompleteSetMatching:
    def __init__(self, project, get_tracklet_probs_callback, get_tracklet_p1s_callback):
        self.p = project
        self.get_probs = get_tracklet_probs_callback
        self.get_p1s = get_tracklet_p1s_callback

    def classify_cs(self):
        # matching to IDs, classification but more robust - we want to use each class once
        pass

    def cs2cs_matching_ids_unknown(self, cs1, cs2):
        # TODO: probability is better than cost, easier to interpret
        # get distance costs
        # get ID assignments costs
        # solve matching
        # register matched tracklets to have the same virtual ID

        C_a = self.appearance_cost(cs1, cs2)
        C_s = self.spatial_cost(cs1, cs2)
        pass

    def spatial_cost(self, cs1, cs2):
        # should be neutral if temporal distance is too big
        # should be restrictive when spatial distance is big
        max_d = self.p.solver_parameters.max_edge_distance_in_ant_length * self.p.stats.major_axis_median
        C = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, t1 in enumerate(cs1):
            t1_ef = t1.end_frame(self.p.gm)
            for j, t2 in enumerate(cs2):
                temporal_d = t2.start_frame(self.p.gm) - t1_ef

                t1_end_r = self.p.gm.region(t1.end_node())
                t2_start_r = self.p.gm.region(t2.start_node())
                spatial_d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid())

                # / float(md)

                # should be there any weight?
                spatial_d = spatial_d / float(max_d * temporal_d)

                # sp_d > 1... 0.5
                # sp_d ~0 ... 1.0

                # 1 frame... d = 0.2 md... sp_d ~ 0.2, prob ~ 0.8
                # 5 frame... d = 2 md... sp_d ~ 0.4, prob ~ 0.6
                # 5 frame... d = 1 md... sp_d ~ 0.2, prob ~ 0.8

                # TODO: ? is it right?
                # TODO,,, frame too big, decrease prob if it is good...
                prob = max(0.5, 1 - spatial_d)

                # # TODO: raise uncertainty with frame_d
                # prob -= (frame_d - 1) * 0.05

                C[i, j] = prob
                C[j, i] = prob

        return C

    def appearance_cost(self, cs1, cs2):
        # ...thoughts...
        # get probabilities for each tracklet
        # ? just probabilities? Or "race conditions term" included ?
        # in my opinion, race condition is already treated by matching
        # thus I suggest using only get_p1, including "homogenity" score

        C = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, t1 in enumerate(cs1):
            p1 = self.get_probs(t1)
            k1 = np.argmax(p1)
            val1 = p1[k1]

            for j, t2 in enumerate(cs2):
                p2 = self.get_probs(t2)
                k2 = np.argmax(p2)
                val2 = p2[k2]

                cost1 = val1 * p2[k1]
                cost2 = p1[k2] * val2

                cost = max(cost1, cost2)

                C[i, j] = cost
                C[j, i] = cost

        return C

    def brajgl(self):

        md = self.p.solver_parameters.max_edge_distance_in_ant_length * self.p.stats.major_axis_median
        C = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, g in enumerate(cs1.values()):
            X = []
            for t_id in g:
                x = self._get_tracklet_proba(self.p.chm[t_id])

                if len(X) == 0:
                    X = np.array(x)
                else:
                    X = np.vstack([X, np.array(x)])

            # TODO: idTracker like probs
            # TODO: or at least compute certainty as first best and second best matching...
            # for each ID, compute ID tracker probs

            # res = self._get_tracklet_proba()
            C[i, :] = np.mean(X, axis=0)

        # TODO: nearest tracklet (position) based on min over all t_distances
        for i, repr_id1 in enumerate(cs1.keys()):
            for j, repr_id2 in enumerate(cs2.keys()):
                # find best pair
                # TODO: can be optimized...
                best_ti = None
                best_tj = None
                best_d = np.inf

                for t_id_i in cs1[repr_id1]:
                    for t_id_j in cs2[repr_id2]:
                        ti = self.p.chm[t_id_i]
                        tj = self.p.chm[t_id_j]

                        ef = ti.end_frame(self.p.gm)
                        sf = tj.start_frame(self.p.gm)

                        # something smaller than np.inf to guarantee at least one best_ti, best_tj
                        d = 10000000
                        if ef < sf:
                            d = sf - ef

                        if d < best_d:
                            best_d = d
                            best_ti = ti
                            best_tj = tj

                t1 = best_ti
                t2 = best_tj

                # same track
                # let, probability is 1
                prob = 1
                if t1 != t2:
                    t1_end_f = t1.end_frame(self.p.gm)
                    t2_start_f = t2.start_frame(self.p.gm)

                    # not allowed, probability is zero
                    if t1_end_f >= t2_start_f:
                        prob = 0
                    else:
                        t1_end_r = self.p.gm.region(t1.end_node())
                        t2_start_r = self.p.gm.region(t2.start_node())

                        frame_d = t2_start_f - t1_end_f
                        # d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid()) / float(frame_d * md)
                        d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid()) / float(md)
                        prob = max(0, 1 - d)

                        # TODO: raise uncertainty with frame_d
                        prob -= (frame_d - 1) * 0.05

                # probability complement... something like cost
                # TODO: -log(P) ?
                C[i, j] = 1 - C[i, j] * prob

        # TODO: what to do with too short CS, this will stop aglomerattive clustering

        from scipy.optimize import linear_sum_assignment

        # use hungarian (nonnegative matrix)
        row_ind, col_ind = linear_sum_assignment(C)
        price = C[row_ind, col_ind].sum()
        price_norm = price / float(len(cs1))

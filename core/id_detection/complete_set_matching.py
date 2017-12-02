import numpy as np
from utils.video_manager import get_auto_video_manager
from tqdm import tqdm
from lazyme.string import color_print


class CompleteSetMatching:
    def __init__(self, project, get_tracklet_probs_callback, get_tracklet_p1s_callback):
        self.p = project
        self.get_probs = get_tracklet_probs_callback
        self.get_p1s = get_tracklet_p1s_callback

    def process(self):
        CSs = self.find_cs()

        isolated_cs_groups = []

        qualities = []
        for i in range(len(CSs)-1):
            print "CS {}, CS {}".format(i, i+1)
            perm, quality = self.cs2cs_matching_ids_unknown(CSs[i], CSs[i+1])

            cs1_max_frame = 0
            cs2_min_frame = np.inf
            for (t1, t2) in perm:
                if t1 == t2:
                    break

                cs1_max_frame = max(cs1_max_frame, t1.start_frame(self.p.gm))
                cs2_min_frame = min(cs2_min_frame, t2.end_frame(self.p.gm))

            print "cs1 max frame: {}, cs2 min frame: {}".format(cs1_max_frame, cs2_min_frame)

            # TODO: threshold 1
            QUALITY_THRESHOLD = 0.4

            # propagate IDS if quality is good enough:
            if quality[1] > QUALITY_THRESHOLD:
                # TODO: if t1.P is empty... do virtual ID init
                if len(perm[0][0].P) == 0:
                    isolated_cs_groups.append(set())

                    ics_g_i = len(isolated_cs_groups) - 1

                    num_animals = len(self.p.animals)
                    full_set = set(range(ics_g_i * num_animals, (ics_g_i + 1)*num_animals))
                    for i, t in enumerate(perm[0]):
                        t.P = set([i + ics_g_i*num_animals])
                        t.N = full_set.difference(t.P)
            else:
                color_print('QUALITY BELOW', color='red')

            # TODO: transitivity? when t1 -> t2 assignment uncertain, look on ID probs for t2->t3 and validate wtih t1->t3
            
            for (t1, t2) in perm:
                print "[{} |{}| (te: {})] -> {} |{}| (ts: {})".format(t1.id(), len(t1), t1.end_frame(self.p.gm), t2.id(), len(t2), t2.start_frame(self.p.gm))
                t2.P = set(t1.P)
                t2.N = set(t2.N)
                isolated_cs_groups[-1].add(t1)
                isolated_cs_groups[-1].add(t2)

            print quality
            print

            qualities.append(quality)

        print

        print "ISOLATED CS GROUPS: {}".format(len(isolated_cs_groups))
        for cs in isolated_cs_groups:
            total_len = 0
            s = ""
            for t in cs:
                s += ", "+str(t.id())
                total_len += len(t)

            print "total len: {}, IDs: {}".format(total_len, s)
            print

        p.save()
        import matplotlib.pyplot as plt
        qualities = np.array(qualities)
        plt.plot(qualities[:, 0])
        plt.grid()
        plt.figure()
        plt.plot(qualities[:, 1])
        plt.grid()
        plt.show()

        for i in range(50, 60):
            print "CS {}, CS {}".format(0, i)
            perm, quality = self.cs2cs_matching_ids_unknown(CSs[0], CSs[i])
            for (t1, t2) in perm:
                print t1.id(), " -> ", t2.id()

            print quality

    def find_cs(self):
        unique_tracklets = set()
        CSs = []
        vm = get_auto_video_manager(self.p)
        total_frame_count = vm.total_frame_count()

        frame = 0
        i = 0
        old_frame = 0
        print "analysing project, searching Complete Sets"
        print
        with tqdm(total=total_frame_count) as pbar:
            while True:
                group = self.p.chm.chunks_in_frame(frame)
                if len(group) == 0:
                    break

                singles_group = filter(lambda x: x.is_single(), group)

                if len(singles_group) == len(self.p.animals) and min([len(t) for t in singles_group]) >= 1:
                    CSs.append(singles_group)

                    for t in singles_group:
                        unique_tracklets.add(t)

                    frame = min([t.end_frame(self.p.gm) for t in singles_group]) + 1
                else:
                    frame = min([t.end_frame(self.p.gm) for t in group]) + 1

                i += 1
                pbar.update(frame - old_frame)
                old_frame = frame

        return CSs

    def classify_cs(self):
        # matching to IDs, classification but more robust - we want to use each class once
        
        # ? how to deal with 
        
        pass

    def cs2cs_matching_ids_unknown(self, cs1, cs2):
        # TODO: probability is better than cost, easier to interpret
        # get distance costs
        # get ID assignments costs
        # solve matching
        # register matched tracklets to have the same virtual ID
        perm = []

        cs1, cs2, cs_shared = self.remove_straightforward_tracklets(cs1, cs2)
        if len(cs1) == 1:
            perm.append((cs1[0], cs2[0]))
            quality = [1.0, 1.0]
        else:
            P_a = self.appearance_probabilities(cs1, cs2)
            P_s = self.spatial_probabilities(cs1, cs2, lower_bound=0.5)

            # 1 - ... it is minimum weight matching
            P = 1 - np.multiply(P_a, P_s)

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(P)

            for rid, cid in zip(row_ind, col_ind):
                perm.append((cs1[rid], cs2[cid]))

            x_ = 1 - P[row_ind, col_ind]
            quality = (x_.min(), x_.sum() / float(len(x_)))

        for t in cs_shared:
            perm.append((t, t))

        return perm, quality

    def spatial_probabilities(self, cs1, cs2, lower_bound=0.5):
        # should be neutral if temporal distance is too big
        # should be restrictive when spatial distance is big
        max_d = self.p.solver_parameters.max_edge_distance_in_ant_length * self.p.stats.major_axis_median
        P = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, t1 in enumerate(cs1):
            t1_ef = t1.end_frame(self.p.gm)
            for j, t2 in enumerate(cs2):
                if t1 == t2:
                    prob = 1.0
                else:
                    temporal_d = t2.start_frame(self.p.gm) - t1_ef

                    if temporal_d < 0:
                        prob = -np.inf
                    else:
                        t1_end_r = self.p.gm.region(t1.end_node())
                        t2_start_r = self.p.gm.region(t2.start_node())
                        spatial_d = np.linalg.norm(t1_end_r.centroid() - t2_start_r.centroid())

                        # should be there any weight?
                        spatial_d = spatial_d / float(max_d)

                        # TODO: what if it just makes something strange out of luck? E.G. Two distant CS with one tracklet which has perfect distance thus p~1.0and all others have ~0.5
                        if (1 - spatial_d) < 0:
                            val = 0
                        else:
                            val = (1 - spatial_d) ** temporal_d

                        prob = max(0.0, val)

                P[i, j] = prob


        # it might occur when t1 ends after t2 starts
        invalid = P < 0

        # minimize P_s impact when distance is too big
        P[P<lower_bound] = lower_bound
        P[invalid] = 0

        return P

    def remove_straightforward_tracklets(self, cs1, cs2):
        cs1 = set(cs1)
        cs2 = set(cs2)
        shared = cs1.intersection(cs2)

        return list(cs1.difference(shared)), list(cs2.difference(shared)), list(shared)

    def appearance_probabilities(self, cs1, cs2):
        # ...thoughts...
        # get probabilities for each tracklet
        # ? just probabilities? Or "race conditions term" included ?
        # in my opinion, race condition is already treated by matching
        # thus I suggest using only get_p1, including "homogenity" score

        C = np.zeros((len(cs1), len(cs2)), dtype=np.float)

        for i, t1 in enumerate(cs1):
            p1 = np.mean(self.get_probs(t1), axis=0)
            k1 = np.argmax(p1)
            val1 = p1[k1]

            for j, t2 in enumerate(cs2):
                p2 = np.mean(self.get_probs(t2), axis=0)
                k2 = np.argmax(p2)
                val2 = p2[k2]

                cost1 = val1 * p2[k1]
                cost2 = p1[k2] * val2

                likelihood = max(cost1, cost2)

                C[i, j] = likelihood

        return C

if __name__ == '__main__':
    from core.project.project import Project
    from core.id_detection.learning_process import LearningProcess

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1')

    lp = LearningProcess(p)
    lp.reset_learning()

    csm = CompleteSetMatching(p, lp._get_tracklet_proba, lp.get_tracklet_p1s)
    csm.process()



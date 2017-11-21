import numpy as np
from utils.video_manager import get_auto_video_manager
from tqdm import tqdm


class CompleteSetMatching:
    def __init__(self, project, get_tracklet_probs_callback, get_tracklet_p1s_callback):
        self.p = project
        self.get_probs = get_tracklet_probs_callback
        self.get_p1s = get_tracklet_p1s_callback

    def process(self):
        CSs = self.find_cs()

        for i in range(10):
            perm, quality = self.cs2cs_matching_ids_unknown(CSs[i], CSs[i+1])
            print perm
            print quality

        for i in range(50, 60):
            perm, quality = self.cs2cs_matching_ids_unknown(CSs[0], CSs[i])
            print perm
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

        P_a = self.appearance_probabilities(cs1, cs2)
        P_s = self.spatial_probabilities(cs1, cs2)

        # minimize P_s impact when distance is too big
        P_s[P_s<0.5] = 0.5

        # 1 - ... it is minimum weight matching
        P = 1 - np.multiply(P_a, P_s)

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(P)

        perm = []
        for rid, cid in zip(row_ind, col_ind):
            perm.append((rid, cid))

        x_ = P[row_ind, col_ind]
        quality = (x_.min(), x_.sum() / float(len(x_)))

        return perm, quality

    def spatial_probabilities(self, cs1, cs2):
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

                # should be there any weight?
                spatial_d = spatial_d / float(max_d)

                prob = max(0.0, (1 - spatial_d)**temporal_d)

                C[i, j] = prob
                C[j, i] = prob

        return C

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

                cost = max(cost1, cost2)

                C[i, j] = cost
                C[j, i] = cost

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



import numpy as np
from utils.video_manager import get_auto_video_manager
from tqdm import tqdm
from lazyme.string import color_print
import matplotlib.pyplot as plt
from scipy.misc import imread


class CompleteSetMatching:
    def __init__(self, project, get_tracklet_probs_callback, get_tracklet_p1s_callback, descriptors):
        self.p = project
        self.get_probs = get_tracklet_probs_callback
        self.get_p1s = get_tracklet_p1s_callback
        self.descriptors = descriptors

    def process(self):
        CSs = self.find_cs()

        #####
        for t in CSs[0]:
            self.get_representants(t, n=10)

        id_ = 0

        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        from matplotlib.widgets import Cursor

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cur = Cursor(ax, horizOn=False)
        plt.hold(True)

        qualities = []
        for i in range(len(CSs)-1):
            print "CS {}, CS {}".format(i, i+1)
            perm, quality = self.cs2cs_matching_descriptors_and_spatial(CSs[i], CSs[i+1])

            cs1_max_frame = 0
            cs2_min_frame = np.inf
            dividing_frame = 0
            for (t1, t2) in perm:
                if t1 == t2:
                    break

                cs1_max_frame = max(cs1_max_frame, t1.start_frame(self.p.gm))
                cs2_min_frame = min(cs2_min_frame, t2.end_frame(self.p.gm))

                dividing_frame = max(dividing_frame, t2.start_frame(self.p.gm))

            print "cs1 max frame: {}, cs2 min frame: {}".format(cs1_max_frame, cs2_min_frame)

            # TODO: threshold 1
            QUALITY_THRESHOLD = 0.4

            for pair in perm:
                t = pair[0]
                if len(t.P) == 0:
                    t.P = set([id_])
                    # TODO: what should we do with P set when virtual IDs are introduced?
                    # t.N = full_set.difference(t.P)
                    id_ += 1

            not_same = 0
            c = [0. + 1-quality[1], quality[1],0., 0.2]
            # propagate IDS if quality is good enough:
            if quality[1] > QUALITY_THRESHOLD:
                # TODO: transitivity? when t1 -> t2 assignment uncertain, look on ID probs for t2->t3 and validate wtih t1->t3

                for (t1, t2) in perm:
                    print "[{} |{}| (te: {})] -> {} |{}| (ts: {})".format(t1.id(), len(t1), t1.end_frame(self.p.gm),
                                                                          t2.id(), len(t2), t2.start_frame(self.p.gm))
                    t2.P = set(t1.P)
                    t2.N = set(t2.N)
            else:
                color_print('QUALITY BELOW', color='red')
                # c = [1., 0.,0.,0.7]

                for pair in perm:
                    if pair[0] != pair[1]:
                        not_same += 1

            plt.plot([dividing_frame, dividing_frame], [-5, -5 + 4.7*quality[1]], c=c)
            plt.plot([dividing_frame, dividing_frame], [0, id_-1 + not_same], c=c)

            print quality
            print

            qualities.append(quality)

        print

        #### visualize and stats
        from utils.rand_cmap import rand_cmap

        new_cmap = rand_cmap(id_+1, type='bright', first_color_black=True, last_color_black=False)
        print "#IDs: {}".format(id_+1)
        support = {}
        tracks = {}
        tracks_mean_desc = {}
        for t in self.p.chm.chunk_gen():
            if len(t.P):
                t_identity = list(t.P)[0]
                support[t_identity] = support.get(t_identity, 0) + len(t)
                if t_identity not in tracks:
                    tracks[t_identity] = []

                tracks[t_identity].append(t.id())

                t_desc_w = self.get_mean_descriptor(t) * len(t)
                if t_identity not in tracks_mean_desc:
                    tracks_mean_desc[t_identity] = t_desc_w
                else:
                    tracks_mean_desc[t_identity] += t_desc_w

                plt.scatter(t.start_frame(self.p.gm), t_identity, c=new_cmap[t_identity], edgecolor=[0.,0.,0.,.3])
                plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm)+0.1], [t_identity, t_identity],
                         c=new_cmap[t_identity],
                         path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
            else:
                if t.is_noise() or len(t) < 5:
                    continue
                if t.is_single():
                    c = [0, 1, 0, .3]
                else:
                    c = [0, 0, 1, .3]

                y = t.id() % id_
                plt.scatter(t.start_frame(self.p.gm), y, c=c, marker='s', edgecolor=[0., 0., 0., .1])
                plt.plot([t.start_frame(self.p.gm), t.end_frame(self.p.gm) + 0.1], [y, y],
                         c=c,
                         linestyle='-')


        plt.grid()

        print "SUPPORT"
        for id in sorted(support.keys()):
            print "{}: {}, #{} ({})".format(id, support[id], len(tracks[id]), tracks[id])


        # print "ISOLATED CS GROUPS: {}".format(len(isolated_cs_groups))
        # for cs in isolated_cs_groups:
        #     total_len = 0
        #     s = ""
        #     for t in cs:
        #         s += ", "+str(t.id())
        #         total_len += len(t)
        #
        #     print "total len: {}, IDs: {}".format(total_len, s)
        #     print

        p.save()
        qualities = np.array(qualities)
        plt.figure()
        plt.plot(qualities[:, 0])
        plt.grid()
        plt.figure()
        plt.plot(qualities[:, 1])
        plt.grid()

        plt.figure()

        mean_ds = []
        for id_, mean in tracks_mean_desc.iteritems():
            mean_ds.append(mean/float(support[id]))

        print("track ids order: {}".format(list(tracks_mean_desc.iterkeys())))
        from scipy.spatial.distance import pdist, squareform
        plt.imshow(squareform(pdist(mean_ds)), interpolation='nearest')
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

    def cs2cs_matching_descriptors_and_spatial(self, cs1, cs2):
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
            P_a = self.appearance_distance_probabilities(cs1, cs2)
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

    def get_mean_descriptor(self, tracklet):
        descriptors = []
        for r_id in tracklet.rid_gen(self.p.gm):
            if r_id in self.descriptors:
                descriptors.append(self.descriptors[r_id])

        if len(descriptors) == 0:
            import warnings
            warnings.warn("descriptors missing for t_id: {}, creating zero vector".format(tracklet.id()))

            descriptors.append(np.zeros(32, ))


        descriptors = np.array(descriptors)

        res = np.mean(descriptors, axis=0)

        assert len(res) == 32

        return res

    def appearance_distance_probabilities(self, cs1, cs2):
        # returns distances to mean descriptors
        from scipy.spatial.distance import cdist

        cs1_descriptors = []
        for i, t1 in enumerate(cs1):
            cs1_descriptors.append(self.get_mean_descriptor(t1))

        cs2_descriptors = []
        for i, t2 in enumerate(cs2):
            cs2_descriptors.append(self.get_mean_descriptor(t2))

        C = cdist(cs1_descriptors, cs2_descriptors)

        max_d = 3.0
        C = C / max_d
        C = 1 - C

        return C

    def desc_clustering_analysis(self):
        from sklearn.cluster import KMeans
        import numpy as np

        Y = []
        X = []
        for y, x in tqdm(self.descriptors.iteritems()):
            Y.append(y)
            X.append(x)

        Y = np.array(Y)

        nbins = 10
        kmeans = KMeans(n_clusters=nbins, random_state=0).fit(X)

        labels = kmeans.labels_


        plt.figure()
        plt.hist(labels, bins=nbins)

        from scipy.spatial.distance import pdist, squareform
        plt.figure()
        plt.imshow(squareform(pdist(kmeans.cluster_centers_)), interpolation='nearest')

        for i in range(nbins):
            xx, yy = 5, 5
            fig, axarr = plt.subplots(xx, yy)
            axarr = axarr.flatten()

            for j, r_id in enumerate(np.random.choice(Y[labels == i], xx*yy)):
                for k in range(6):
                    img = np.random.rand(50, 50, 3)
                    try:
                        img = imread('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/'+str(k)+'/'+str(r_id)+'.jpg')
                        break
                    except:
                        pass

                axarr[j].imshow(img)
                axarr[j].set_title(str(k))
                axarr[j].axis('off')

            plt.suptitle(str(i))
            plt.show()


        kmeans.cluster_centers_

    def get_representants(self, tracklet, n=5):
        linkages = ['average', 'complete', 'ward']
        linkage = linkages[0]
        connectivity = None

        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(linkage=linkage,
                                        connectivity=connectivity,
                                        n_clusters=n)

        # for given GT ID
        for id_ in range(6):
            X = []
            r_ids = []

            r_ids_arr = tracklet.rid_gen(self.p.gm)

            r_ids_arr = []
            import os
            for r_id in os.listdir('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/'+str(id_)+'/'):
                r_id = int(r_id[:-4])
                r_ids_arr.append(r_id)

            for r_id in r_ids_arr:
                if r_id in self.descriptors:
                    X.append(self.descriptors[r_id])
                    r_ids.append(r_id)
                else:
                    print r_id

            r_ids = np.array(r_ids)
            y = model.fit_predict(X)

            print np.histogram(y, bins=n)

            num_examples = 5
            fig, axarr = plt.subplots(num_examples, n)
            axarr = axarr.flatten()

            for i in range(n):
                for j, r_id in enumerate(np.random.choice(r_ids[y == i], min(num_examples, np.sum(y == i)))):
                    for k in range(6):
                        img = np.random.rand(50, 50, 3)
                        try:
                            img = imread('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/' + str(k) + '/' + str(
                                r_id) + '.jpg')
                            break
                        except:
                            pass

                    axarr[j * n + i].imshow(img)
                    if j == 0:
                        axarr[j * n + i].set_title(str(np.sum(y == i)))

            for i in range(n*num_examples):
                axarr[i].axis('off')

            plt.suptitle(len(y))
            plt.show()

if __name__ == '__main__':
    from core.project.project import Project
    from core.id_detection.learning_process import LearningProcess

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1')
    # p.load('/Users/flipajs/Documents/wd/FERDA/Camera3_new')

    lp = LearningProcess(p)
    lp.reset_learning()

    import pickle
    with open('/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_Cam1/descriptors.pkl') as f:
        descriptors = pickle.load(f)

    csm = CompleteSetMatching(p, lp._get_tracklet_proba, lp.get_tracklet_p1s, descriptors)
    # csm.desc_clustering_analysis()
    csm.process()



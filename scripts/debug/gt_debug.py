from utils.gt.gt import GT
import cPickle as pickle

if __name__ == '__main__':
    from core.project.project import Project
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')
    p.load('/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground')

    with open(p.working_directory+'/temp/isolation_score.pkl', 'rb') as f:
    # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/isolation_score.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        p.gm.g = up.load()
        up.load()
        chm = up.load()
        p.chm = chm

    from core.region.region_manager import RegionManager
    p.rm = RegionManager(p.working_directory+'/temp', db_name='part0_rm.sqlite3')
    p.gm.rm = p.rm

    p.chm.add_single_vertices_chunks(p, frames=range(4500))
    p.gm.update_nodes_in_t_refs()


    # p.load('/Users/flipajs/Documents/wd/zebrafish')
    # p.GT_file = '/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl'
    # p.save()

    gt = GT()
    gt.load(p.GT_file)

    epsilons = []
    edges = []
    variant = []
    symmetric = []

    theta = 0.5

    for v in p.gm.active_v_gen():
        if int(v) == 11043:
            print "stop"
        else:
            continue

        e, es = p.gm.get_2_best_out_edges_appearance_motion_mix(v)

        if e[1] is not None:
            # e_, es_ = p.gm.get_2_best_in_edges_appearance_motion_mix(e[0].source())
            # if e_[0].target() == e[0].target() or (e_[1] is not None and e_[1].target() == e[0].target()):
            #     symmetric.append(1)
            # else:
            #     symmetric.append(0)

            A = es[0]
            B = es[1]
            if gt.test_edge(p.gm.get_chunk(e[0].source()), p.gm.get_chunk(e[0].target()), p):
                eps = (A / theta) - (A + B)
                variant.append(0)
            else:
                eps = (A + B) / ((1/theta) - 1)
                variant.append(1)


            epsilons.append(eps)
            edges.append((int(e[0].source()), int(e[0].target())))

    print min(epsilons), max(epsilons)

    with open(p.working_directory+'/temp/epsilons', 'wb') as f:
        pickle.dump((epsilons, edges, variant), f)

    # gt.project_stats(p)

    # t1 = p.gm.get_chunk(762)
    # print gt.tracklet_id_set(t1, p)
    #
    # t2 = p.gm.get_chunk(784)
    # print gt.tracklet_id_set(t2, p)
    #
    # print gt.tracklet_id_set(p.gm.get_chunk(891), p)

    # gt.build_from_PN(p)
    # gt.save('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')

    # gt = GT()
    # gt.build_from_PN(p)
    #
    # # gt.load('/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl')
    # gt.save('/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl')
    #
    # print gt.get_clear_positions(100)
    # print gt.get_clear_rois(100)
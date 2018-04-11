import cPickle as pickle
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
from core.graph.solver import Solver
import warnings
from itertools import izip
from tqdm import tqdm
import os
import datetime


def check_if_already_assembled(project):
    wd = project.working_directory
    rm_path = wd+'/rm.sqlite3'
    if os.path.exists(rm_path) and len(project.rm):
        warnings.warn("Region Manager already exists. It was renamed to rm_CURRENT_DATETIME.sqlite3")

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(rm_path, rm_path[:-8] + '_' + dt + '.sqlite3')
        project.rm = RegionManager(wd)

    chm_path = wd + '/chunk_manager.pkl'
    if os.path.exists(chm_path):
        warnings.warn("Chunk Manager already exists. It was renamed to chm_CURRENT_DATETIME.sqlite")

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(chm_path, chm_path[:-4] + '_' + dt + '.pkl')

    gm_path = wd + '/graph_manager.pkl'
    if os.path.exists(gm_path):
        warnings.warn("Graph Manager already exists. It was renamed to gm_CURRENT_DATETIME.sqlite")

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(gm_path, gm_path[:-4] + '_' + dt + '.pkl')


def assembly_after_parallelization(bgcomp):
    check_if_already_assembled(bgcomp.project)

    print "Starting assembly..."
    from core.graph.graph_manager import GraphManager
    # TODO: add to settings

    bgcomp.project.chm = ChunkManager()
    bgcomp.solver = Solver(bgcomp.project)
    bgcomp.project.gm = GraphManager(bgcomp.project, bgcomp.solver.assignment_score)

    if not bgcomp.project.is_cluster():
        bgcomp.update_callback(0, 're-indexing...')

    # if not bgcomp.project.is_cluster():
    #     from core.settings import Settings as S_
    #     # switching off... We don't want to log following...
    #     S_.general.log_graph_edits = False

    part_num = bgcomp.part_num

    from utils.misc import is_flipajs_pc
    if is_flipajs_pc():
        # TODO: remove this line
        part_num = 5
        # bgcomp.first_part = 100
        pass

    bgcomp.project.color_manager = None

    import time
    merging_t = time.time()

    print("\nLOADING PARTS AND MERGING...")
    # for i in range(part_num):
    for i in tqdm(range(bgcomp.first_part, bgcomp.first_part + part_num), leave=False):
        rm_old = RegionManager(db_wd=bgcomp.project.working_directory + '/temp',
                               db_name='part' + str(i) + '_rm.sqlite3', cache_size_limit=1, supress_init_print=True)

        with open(bgcomp.project.working_directory + '/temp/part' + str(i) + '.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            g_ = up.load()
            relevant_vertices = up.load()
            chm_ = up.load()

            merge_parts(bgcomp.project.gm, g_, relevant_vertices, bgcomp.project, rm_old, chm_)

        if not bgcomp.project.is_cluster():
            bgcomp.update_callback((i + 1) / float(part_num))

    fir = bgcomp.project.solver_parameters.frames_in_row

    if not bgcomp.project.is_cluster():
        bgcomp.update_callback(-1, 'joining parts...')

    bgcomp.project.gm.rm = bgcomp.project.rm

    print
    print "RECONNECTING GRAPHS"
    print
    vs_todo = []

    start_ = (bgcomp.first_part + 1) * fir
    for part_end_t in range(start_, start_ + fir*part_num, fir):
        t_v = bgcomp.project.gm.get_vertices_in_t(part_end_t - 1)
        t1_v = bgcomp.project.gm.get_vertices_in_t(part_end_t)

        vs_todo.extend(t_v)

        connect_graphs(bgcomp, t_v, t1_v, bgcomp.project.gm, bgcomp.project.rm)
        # self.solver.simplify(t_v, rules=[self.solver.adaptive_threshold])

    print "merge t: {:.2f}".format(time.time() - merging_t)
    print "#CHUNKS: ", len(bgcomp.project.chm)
    print "simplifying "

    p = bgcomp.project
    one2one_t = time.time()
    try:
        # TODO:
        if bgcomp.project.type == 'colony':
            rules = [bgcomp.solver.adaptive_threshold]

            while True:
                if bgcomp.solver.simplify(rules=rules) == 0:
                    break
        else:
            bgcomp.solver.simplify(rules=[bgcomp.solver.one2one])
    except:
        bgcomp.solver.one2one()

    print
    print "\tfirst one2one t: {:.2f}s".format(time.time() - one2one_t)

    learn_assignment_t = time.time()

    bgcomp.project.save_semistate('first_tracklets')
    from scripts.regions_stats import learn_assignments, add_score_to_edges, tracklet_stats
    learn_assignments(bgcomp.project, max_examples=50000, display=False)

    print
    print "\tlearn assignment t: {:.2f}s".format(time.time() - learn_assignment_t)
    learn_assignment_t = time.time()

    p.gm.g.ep['movement_score'] = p.gm.g.new_edge_property("float")
    add_score_to_edges(p)

    print "\tscore edges t: {:.2f}s".format(time.time() - learn_assignment_t)

    p.save_semistate('edge_cost_updated')

    update_t = time.time()
    p.gm.update_nodes_in_t_refs()
    p.chm.reset_itree(p.gm)
    print "\tupdate t: {:.2f}s".format(time.time() - update_t)

    tracklet_stats(p)

    # TODO: max num of iterations...
    for i in range(10):
        score_type = 'appearance_motion_mix'
        eps = 0.3

        strongly_better_t = time.time()
        strongly_better_e = p.gm.strongly_better_eps2(eps=eps, score_type=score_type)
        if len(strongly_better_e) == 0:
            print
            print "BREAK"
            break

        strongly_better_e = sorted(strongly_better_e, key=lambda x: -x[0])

        print
        print "ITERATION: {}, #decisions: {}, t: {:.2f}s".format(i, len(strongly_better_e), time.time()-strongly_better_t)
        confirm_t = time.time()
        for _, e in tqdm(strongly_better_e, leave=False):
            if p.gm.g.edge(e.source(), e.target()) is not None:
                bgcomp.solver.confirm_edges([(e.source(), e.target())])

        print "\tconfirm_t: {:.2f}".format(time.time() - confirm_t)

        # tracklet_stats(p)

        p.gm.update_nodes_in_t_refs()
        p.chm.reset_itree(p.gm)

    p.save_semistate('eps_edge_filter')

    # if not bgcomp.project.is_cluster():
    #     from core.settings import Settings as S_
    #     S_.general.log_graph_edits = True

    p.solver = bgcomp.solver

    p.gm.project = bgcomp.project
    p.chm.add_single_vertices_chunks(p)

    if not bgcomp.project.is_cluster():
        bgcomp.update_callback(-1, 'saving...')

    p.gm.update_nodes_in_t_refs()

    if not bgcomp.do_semi_merge:
        p.save()

    print "SANITY CHECK..."
    sanity_check = True
    for ch in tqdm(p.chm.chunk_gen(), total=len(p.chm), leave=False):
        if len(ch) > 1:
            # test start
            v = ch.start_node()
            if not p.gm.g.vp['chunk_start_id'][v] or p.gm.g.vp['chunk_end_id'][v]:
                print v, ch, p.gm.g.vp['chunk_start_id'][v], p.gm.g.vp['chunk_end_id'][v]
                sanity_check = False

            v = ch.end_node()
            if p.gm.g.vp['chunk_start_id'][v] or not p.gm.g.vp['chunk_end_id'][v]:
                print v, ch, p.gm.g.vp['chunk_start_id'][v], p.gm.g.vp['chunk_end_id'][v]
                sanity_check = False

    print
    print "SANITY CHECK SUCCEEDED: {}".format(sanity_check)

    print
    print "ONE 2 ONE optimization"
    bgcomp.project.solver.one2one(check_tclass=True)
    print "DONE"
    print
    tracklet_stats(bgcomp.project)

    if not bgcomp.project.is_cluster():
        bgcomp.finished_callback(bgcomp.solver)


def connect_graphs(bgcomp, vertices1, vertices2, gm, rm):
    if vertices1:
        #r1 = gm.region(vertices1[0])

        bgcomp.project.gm.add_edges_(vertices1, vertices2)

    # for v1 in vertices1:
    #     r1 = gm.region(v1)
    #     for v2 in vertices2:
    #         r2 = gm.region(v2)
    #
    #         d = np.linalg.norm(r1.centroid() - r2.centroid())
    #
    #         if d < gm.max_distance:
    #             s, ds, multi, antlike = self.solver.assignment_score(r1, r2)
    #             gm.add_edge_fast(v1, v2, 0)


def merge_parts(new_gm, old_g, old_g_relevant_vertices, project, old_rm, old_chm):
    """
    merges all parts (from parallelisation)
    we want to merge all these structures (graph, region and chunk managers) into one

    in the beginning there were separate graphs(for given time bounds) with ids starting from 0
    ids in region manager also starts with 0, the same for chunk manager
    -> reindexing is needed

    :param new_gm:
    :param old_g:
    :param old_g_relevant_vertices:
    :param project:
    :param old_rm:
    :param old_chm:
    :return:
    """

    single_vertices = []

    new_chm = project.chm
    new_rm = project.rm

    vertex_map = {}
    used_chunks_ids = set()

    old_vs = []
    old_rids = []
    # reindex vertices
    for v_id in old_g_relevant_vertices:
        if not old_g.vp['active'][v_id]:
            continue

        old_v = old_g.vertex(v_id)
        old_vs.append(old_v)
        old_rids.append(old_g.vp['region_id'][old_v])

    # becaused old_regions will be sorted by region_id and v_id increments in the same time as region_id...
    old_vs = sorted(old_vs)

    old_regions = old_rm[old_rids]
    new_rm.add(old_regions)

    for old_v, old_reg in izip(old_vs, old_regions):
        new_v = new_gm.add_vertex(old_reg)
        vertex_map[old_v] = new_v

        used_chunks_ids.add(old_g.vp['chunk_start_id'][old_v])
        used_chunks_ids.add(old_g.vp['chunk_end_id'][old_v])

        if old_g.vp['chunk_start_id'][old_v] == 0 and old_g.vp['chunk_end_id'][old_v] == 0:
            single_vertices.append(new_v)

    # because 0 id means - no chunk assigned to this node!
    if 0 in used_chunks_ids:
        used_chunks_ids.remove(0)
    elif len(used_chunks_ids) == 0:
        warnings.warn("There is 0 chunks in old graph", Warning)

    # go through all edges and copy them with all edge properties...
    for old_e in old_g.edges():
        v1_old = old_e.source()
        v2_old = old_e.target()
        old_score = old_g.ep['score'][old_e]

        if v1_old in vertex_map and v2_old in vertex_map:
            v1_new = vertex_map[v1_old]
            v2_new = vertex_map[v2_old]
        else:
            # this means there was some outdated edge, it is fine to ignore it...
            continue

        # # add edges only in one direction
        # if int(v1_new) > int(v2_new):
        #     continue

        # ep['score'] is assigned in add_edge call
        new_e = new_gm.add_edge(v1_new, v2_new, old_score)
        new_gm.g.ep['movement_score'][new_e] = old_g.ep['movement_score'][old_e]

    # chunk id = 0 means no chunk assigned
    chunks_map = {0: 0}
    # update chunks
    for old_id_ in used_chunks_ids:
        ch = old_chm[old_id_]

        new_list = []
        for old_v in ch.nodes_:
            if old_v in vertex_map:
                new_list.append(int(vertex_map[old_v]))
            else:
                # old_id__ = old_rm[old_g.vp['region_id'][old_g.vertex(old_v)]].id()

                id_ = new_rm.add(old_rm[old_g.vp['region_id'][old_g.vertex(old_v)]])
                # list of ids is returned [id] ...
                id_ = id_[0]

                # this happens in case when the vertex will not be in new graph, but we wan't to keep the region in
                # RM (e. g. for inner points of chunks)
                new_list.append(-id_)

        _, new_id_ = new_chm.new_chunk(new_list, new_gm)

        chunks_map[old_id_] = new_id_

    for old_v, new_v in vertex_map.iteritems():
        new_gm.g.vp['chunk_start_id'][new_v] = chunks_map[old_g.vp['chunk_start_id'][old_v]]
        new_gm.g.vp['chunk_end_id'][new_v] = chunks_map[old_g.vp['chunk_end_id'][old_v]]

        # # create chunk for each single vertex
        # for v_id in single_vertices:
        #     ch, id = project.chm.new_chunk([int(v_id)], project.gm)
        #     new_gm.g.vp['chunk_start_id'][v_id] = id
        #     new_gm.g.vp['chunk_end_id'][v_id] = id
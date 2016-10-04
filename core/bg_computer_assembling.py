
import sys
import cPickle as pickle
from core.project.project import Project
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
from core.graph.solver import Solver


def assembly_after_parallelization(bgcomp):
    print "Starting assembly..."
    from core.graph.graph_manager import GraphManager
    # TODO: add to settings

    cache_size_limit = 5

    # Settings won't work on cluster, need PyQt libraries...
    if not bgcomp.project.is_cluster():
        from core.settings import Settings as S_
        cache_size_limit = S_.cache.region_manager_num_of_instances

    db_wd = bgcomp.project.working_directory
    if bgcomp.do_semi_merge:
        # means - do not use database, use memory only
        cache_size_limit = -1
        db_wd = None

    bgcomp.project.rm = RegionManager(db_wd=db_wd, cache_size_limit=cache_size_limit)

    bgcomp.project.chm = ChunkManager()
    bgcomp.solver = Solver(bgcomp.project)
    bgcomp.project.gm = GraphManager(bgcomp.project, bgcomp.solver.assignment_score)

    if not bgcomp.project.is_cluster():
        bgcomp.update_callback(0, 're-indexing...')

    if not bgcomp.project.is_cluster():
        from core.settings import Settings as S_
        # switching off... We don't want to log following...
        S_.general.log_graph_edits = False

    part_num = bgcomp.part_num

    from utils.misc import is_flipajs_pc
    if is_flipajs_pc():
        # TODO: remove this line
        part_num = 2
        pass

    bgcomp.project.color_manager = None

    print "merging..."
    # for i in range(part_num):
    for i in range(bgcomp.first_part, bgcomp.first_part + part_num):
        rm_old = RegionManager(db_wd=bgcomp.project.working_directory + '/temp',
                               db_name='part' + str(i) + '_rm.sqlite3', cache_size_limit=1)

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

    if bgcomp.project.solver_parameters.use_emd_for_split_merge_detection():
        bgcomp.project.solver.detect_split_merge_cases()

    bgcomp.project.gm.rm = bgcomp.project.rm

    print "reconnecting graphs"

    vs_todo = []

    start_ = (bgcomp.first_part + 1) * fir
    for part_end_t in range(start_, start_ + fir*part_num, fir):
        t_v = bgcomp.project.gm.get_vertices_in_t(part_end_t - 1)
        t1_v = bgcomp.project.gm.get_vertices_in_t(part_end_t)

        vs_todo.extend(t_v)

        connect_graphs(bgcomp, t_v, t1_v, bgcomp.project.gm, bgcomp.project.rm)
        # self.solver.simplify(t_v, rules=[self.solver.adaptive_threshold])

    if bgcomp.project.solver_parameters.use_emd_for_split_merge_detection():
        bgcomp.project.solver.detect_split_merge_cases()

    print "#CHUNKS: ", len(bgcomp.project.chm)
    bgcomp.solver.simplify(vs_todo, rules=[bgcomp.solver.adaptive_threshold])

    print "simplifying "

    if not bgcomp.project.is_cluster():
        from core.settings import Settings as S_
        S_.general.log_graph_edits = True

    bgcomp.project.solver = bgcomp.solver

    bgcomp.project.gm.project = bgcomp.project

    # from utils.color_manager import colorize_project
    # import time
    # s = time.time()
    # # colorize_project(bgcomp.project)
    # print "color manager takes %f seconds" % (time.time() - s)

    if not bgcomp.project.is_cluster():
        bgcomp.update_callback(-1, 'saving...')

    if not bgcomp.do_semi_merge:
        bgcomp.project.save()

    print ("#CHUNKS: %d") % (len(bgcomp.project.chm.chunk_list()))

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
    #             gm.add_edge_fast(v1, v2, 0)9


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
    # reindex vertices
    for v_id in old_g_relevant_vertices:
        if not old_g.vp['active'][v_id]:
            continue

        old_v = old_g.vertex(v_id)
        old_reg = old_rm[old_g.vp['region_id'][old_v]]
        new_rm.add(old_reg)

        new_v = new_gm.add_vertex(old_reg)
        vertex_map[old_v] = new_v

        used_chunks_ids.add(old_g.vp['chunk_start_id'][old_v])
        used_chunks_ids.add(old_g.vp['chunk_end_id'][old_v])

        if old_g.vp['chunk_start_id'][old_v] == 0 and old_g.vp['chunk_end_id'][old_v] == 0:
            single_vertices.append(new_v)

    # because 0 id means - no chunk assigned!
    used_chunks_ids.remove(0)

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

        # add edges only in one direction
        if int(v1_new) > int(v2_new):
            continue

        # ep['score'] is assigned in add_edge call
        new_e = new_gm.add_edge(v1_new, v2_new, old_score)
        new_gm.g.ep['certainty'][new_e] = old_g.ep['certainty'][old_e]

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
                old_id__ = old_rm[old_g.vp['region_id'][old_g.vertex(old_v)]].id()

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

    # create chunk for each single vertex
    for v_id in single_vertices:
        ch, id = project.chm.new_chunk([int(v_id)], project.gm)
        new_gm.g.vp['chunk_start_id'][v_id] = id
        new_gm.g.vp['chunk_end_id'][v_id] = id
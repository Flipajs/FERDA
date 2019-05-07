import cPickle as pickle
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
from core.graph.graph_manager import GraphManager
from core.segmentation import load_segmentation_info
from core.config import config
from scripts.regions_stats import learn_assignments, add_score_to_edges, print_tracklet_stats
import warnings
from itertools import izip
from tqdm import tqdm
import os
from os.path import join
import datetime
import glob
import shutil


def is_assembly_completed(project):
    """
    Check whether the assembly step is completed.

    Fails on project with zero regions or zero tracklets.

    :param project: core.Project
    :return: bool
    """
    return len(project.rm) > 0 and project.gm.g.num_vertices() > 0 and len(project.chm) > 0


def backup(project):
    rm_path = join(project.working_directory, 'rm.sqlite3')
    chm_path = join(project.working_directory, 'chunk_manager.pkl')
    gm_path = join(project.working_directory, 'graph_manager.pkl')
    if os.path.exists(rm_path) and len(project.rm):
        warnings.warn("Region Manager already exists. It was renamed to rm_CURRENT_DATETIME.sqlite3")

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(rm_path, rm_path[:-8] + '_' + dt + '.sqlite3')
        project.rm = RegionManager(project.working_directory)

    if os.path.exists(chm_path):
        warnings.warn("Chunk Manager already exists. It was renamed to chm_CURRENT_DATETIME.sqlite")

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(chm_path, chm_path[:-4] + '_' + dt + '.pkl')

    if os.path.exists(gm_path):
        warnings.warn("Graph Manager already exists. It was renamed to gm_CURRENT_DATETIME.sqlite")

        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(gm_path, gm_path[:-4] + '_' + dt + '.pkl')


def get_parts_num(parts_path):
    return len(glob.glob(join(parts_path, '*/')))


def graph_assembly(project, do_semi_merge=False):
    # backup(project)

    print("Starting assembly...")
    # TODO: add to settings

    # project.chm = ChunkManager()
    # project.gm = GraphManager(project.gm.assignment_score)

    import time
    merging_t = time.time()

    print("\nLOADING PARTS AND MERGING...")
    parts_path = join(project.working_directory, 'temp')
    n_parts = get_parts_num(parts_path)
    parts_info = []
    # rms = [RegionManager.from_dir(join(parts_path, str(i))) for i in range(n_parts)]
    merged_rm = RegionManager()
    for rm in [RegionManager.from_dir(join(parts_path, str(i))) for i in range(n_parts)]:
        merged_rm.extend(rm)
        rm.close()  # otherwise the shutil.rmtree(parts_path) fails on NFS while the regions.h5 maybe still open
        # TODO: possibly close / open to unload already written data from memory
    project.set_rm(merged_rm)
    for frame, df_frame in tqdm(project.rm.regions_df.set_index('frame_').groupby(level=0), desc='creating graph'):
        regions = [project.rm[i] for i in df_frame['id_'].values]
        project.gm.add_regions_in_t(regions, frame)
    # for frame, rids in project.gm.vertices_in_t.iteritems():
    #     for rid in rids:
    #         assert project.gm.region(rid).frame() == frame
    project.gm.project = project  # workaround, to be refactored
    project.solver.create_tracklets()
    del project.gm.project

    # for i in tqdm(range(n_parts), leave=False):
    #     # parts_info.append(load_segmentation_info(parts_path, i))
    #     part_dir = join(parts_path, str(i))
    #     from core.project.project import Project, set_managers
    #     project_part = Project.from_dir(part_dir)
    #
    #     # with open(join(parts_path, 'part{}.pkl'.format(i)), 'rb') as f:
    #     #     up = pickle.Unpickler(f)
    #     #     g_ = up.load()
    #     #     relevant_vertices = up.load()
    #     #     chm_ = up.load()
    #
    #     merge_parts(project, project_part)  # relevant_vertices
    #
    # project.gm.rm = project.rm
    #
    # print("\nRECONNECTING GRAPHS\n")
    #
    # for info1, info2 in zip(parts_info[:-1], parts_info[1:]):
    # # for part_end_t in range(start_, start_ + frames_in_row * n_parts, frames_in_row):
    #     vertices1 = project.gm.get_vertices_in_t(info1['frame_end'])
    #     vertices2 = project.gm.get_vertices_in_t(info2['frame_start'])
    #
    #     connect_graphs(project, vertices1, vertices2, project.gm, project.rm)
    #     # self.solver.simplify(vertices1, rules=[self.solver.adaptive_threshold])
    #
    # print "merge t: {:.2f}".format(time.time() - merging_t)
    # print "#CHUNKS: ", len(project.chm)
    # print "simplifying "
    #
    p = project
    # one2one_t = time.time()
    # # try:
    # #     graph_solver.simplify(rules=[graph_solver.one2one])
    # # except:
    # graph_solver.create_tracklets()
    #
    # print "\n\tfirst one2one t: {:.2f}s".format(time.time() - one2one_t)

    # learn and add edge weights

    learn_assignment_t = time.time()

    # project.load_semistate(project.project_file, 'first_tracklets')
    # learn isolation forests for appearance and movement of consecutive regions in all tracklets
    movement, appearance = learn_assignments(project, max_examples=50000, display=False)

    print("\n\tlearn assignment t: {:.2f}s".format(time.time() - learn_assignment_t))
    learn_assignment_t = time.time()

    p.gm.g.ep['movement_score'] = p.gm.g.new_edge_property("float")
    add_score_to_edges(p.gm, movement, appearance)

    print("\tscore edges t: {:.2f}s".format(time.time() - learn_assignment_t))

    # p.save_semistate('edge_cost_updated')

    update_t = time.time()
    # TODO: is this needed?
    p.gm.update_nodes_in_t_refs()
    p.chm.reset_itree()
    print("\tupdate t: {:.2f}s".format(time.time() - update_t))

    print_tracklet_stats(p)



    # TODO: max num of iterations...
    for i in range(10):
        score_type = 'appearance_motion_mix'
        eps = 0.3

        strongly_better_t = time.time()
        strongly_better_e = p.gm.strongly_better_eps2(eps=eps, score_type=score_type)
        if len(strongly_better_e) == 0:
            print "\nBREAK"
            break

        strongly_better_e = sorted(strongly_better_e, key=lambda x: -x[0])

        print "\nITERATION: {}, #decisions: {}, t: {:.2f}s".format(i, len(strongly_better_e), time.time()-strongly_better_t)
        confirm_t = time.time()
        for _, e in tqdm(strongly_better_e, leave=False):
            if p.gm.g.edge(e.source(), e.target()) is not None:
                p.solver.confirm_edges([(e.source(), e.target())])

        print "\tconfirm_t: {:.2f}".format(time.time() - confirm_t)

        # tracklet_stats(p)

        p.gm.update_nodes_in_t_refs()
        p.chm.reset_itree()

    # p.save_semistate('eps_edge_filter')
    # p.solver = graph_solver

    # p.gm.project = project
    p.chm.add_single_vertices_tracklets(p.gm)

    p.gm.update_nodes_in_t_refs()

    # if not do_semi_merge:
    #     p.save()

    print("SANITY CHECK...")
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
    project.gm.project = project  # workaround, to be refactored
    project.solver.create_tracklets()
    del project.gm.project
    print "DONE"
    print
    print_tracklet_stats(project)
    shutil.rmtree(parts_path)  # remove segmentation parts


def connect_graphs(project, vertices1, vertices2, gm, rm):
    if vertices1:
        #r1 = gm.region(vertices1[0])

        project.gm.add_edges_(vertices1, vertices2)

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


def merge_parts(project, project_part):
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
    new_gm = project.gm
    old_g = project_part.gm.g
    old_rm = project_part.rm
    old_chm = project_part.chm

    vertex_map = {}
    used_chunks_ids = set()

    old_vs = []
    old_rids = []
    # reindex vertices
    for v_id in project_part.gm.get_all_relevant_vertices():
        if not project_part.gm.g.vp['active'][v_id]:
            continue

        old_v = old_g.vertex(v_id)
        old_vs.append(old_v)
        old_rids.append(old_g.vp['region_id'][old_v])

    # because old_regions will be sorted by region_id and v_id increments in the same time as region_id...
    old_vs = sorted(old_vs)

    old_regions = [old_rm[rid] for rid in old_rids]
    new_rm.extend(old_regions)

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

                new_rm.append(old_rm[old_g.vp['region_id'][old_g.vertex(old_v)]])
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
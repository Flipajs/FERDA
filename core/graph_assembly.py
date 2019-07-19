from core.region.region_manager import RegionManager
from core.config import config
from scripts.regions_stats import learn_assignments, add_score_to_edges, print_tracklet_stats
from tqdm import tqdm
from os.path import join
import glob
import shutil
import logging
import time

logger = logging.getLogger(__name__)


def get_parts_num(parts_path):
    return len(glob.glob(join(parts_path, '*/')))


def graph_assembly(project):
    logger.info("start")
    # TODO: add to settings

    project.reset_managers()  # clean-up possible previously computed data
    parts_path = join(project.working_directory, 'temp')
    n_parts = get_parts_num(parts_path)
    for rm in [RegionManager.from_dir(join(parts_path, str(i))) for i in range(n_parts)]:
        project.rm.extend(rm)
        rm.close()  # otherwise the shutil.rmtree(parts_path) fails on NFS while the regions.h5 maybe still open
        # TODO: possibly close / open to unload already written data from memory
    logger.debug(project.rm.regions_df.describe())
    for frame, df_frame in tqdm(project.rm.regions_df.set_index('frame_').groupby(level=0), desc='creating graph'):
        regions = [project.rm[i] for i in df_frame['id_'].values]
        project.gm.add_regions_in_t(regions, frame)
    # for frame, rids in project.gm.vertices_in_t.iteritems():
    #     for rid in rids:
    #         assert project.gm.region(rid).frame() == frame
    project.gm.project = project  # workaround, to be refactored
    project.solver.create_tracklets()
    del project.gm.project

    p = project

    learn_assignment_t = time.time()

    # learn isolation forests for appearance and movement of consecutive regions in all tracklets
    movement, appearance = learn_assignments(project, max_examples=50000, display=False)

    print("\n\tlearn assignment t: {:.2f}s".format(time.time() - learn_assignment_t))
    learn_assignment_t = time.time()

    p.gm.g.ep['movement_score'] = p.gm.g.new_edge_property("float")
    add_score_to_edges(p.gm, movement, appearance)

    print("\tscore edges t: {:.2f}s".format(time.time() - learn_assignment_t))

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

    p.chm.add_single_vertices_tracklets(p.gm)

    p.gm.update_nodes_in_t_refs()

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

    print "SANITY CHECK SUCCEEDED: {}".format(sanity_check)
    print "ONE 2 ONE optimization"

    project.gm.project = project  # workaround, to be refactored
    project.solver.create_tracklets()
    del project.gm.project

    print_tracklet_stats(project)
    try:
        shutil.rmtree(parts_path)  # remove segmentation parts
    except OSError as e:
        logger.warning('Failed to remove temporary project(s): {}. The problem could be related with NFS.'.format(e.message))
    logger.debug(project.gm)
    logger.info("DONE")

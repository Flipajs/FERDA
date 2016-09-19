__author__ = 'fnaiser'
import os
import sys
import inspect
import multiprocessing
pool=multiprocessing.Pool(processes=4)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import sys
from utils.video_manager import get_auto_video_manager
import cPickle as pickle
from core.region.mser import ferda_filtered_msers
from core.graph.solver import Solver
from core.project.project import Project
from core.settings import Settings as S_
from utils.img import prepare_for_segmentation
from core.region.region_manager import RegionManager
from core.graph.chunk_manager import ChunkManager
import numpy as np

import time

if __name__ == '__main__':
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')

    if not os.path.exists(proj.working_directory+'/temp'):
        os.mkdir(proj.working_directory+'/temp')

    solver = Solver(proj)
    from core.graph.graph_manager import GraphManager
    proj.gm = GraphManager(proj, proj.solver.assignment_score)
    proj.rm = RegionManager(db_wd=proj.working_directory+'/temp', db_name='part'+str(id)+'_rm.sqlite3', cache_size_limit=S_.cache.region_manager_num_of_instances)
    proj.chm = ChunkManager()
    proj.color_manager = None

    S_.general.log_graph_edits = False

    vid = get_auto_video_manager(proj)
    if id*frames_in_row > 0:
        img = vid.seek_frame(id*frames_in_row)
    else:
        img = vid.next_frame()

    if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
        proj.segmentation_model.set_image(img)
        seg = proj.segmentation_model.predict()
        img = np.asarray((-seg*255)+255, dtype=np.uint8)
    else:
        img = prepare_for_segmentation(img, proj)

    msers_t = 0
    solver_t = 0
    vid_t = 0
    file_t = 0

    for i in range(frames_in_row + last_n_frames):
        frame = id*frames_in_row + i

        s = time.time()
        msers = ferda_filtered_msers(img, proj, frame)

        if proj.colormarks_model:
            proj.colormarks_model.assign_colormarks(proj, msers)

        proj.rm.add(msers)
        msers_t += time.time()-s

        s = time.time()
        img = vid.next_frame()
        if img is None:
            break

        if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
            proj.segmentation_model.set_image(img)
            seg = proj.segmentation_model.predict()
            img = np.asarray((-seg*255)+255, dtype=np.uint8)
        else:
            img = prepare_for_segmentation(img, proj)

        vid_t += time.time() - s

        s = time.time()

        # TODO: test antlikeness before add regions_in_t
        # if self.antlike_filter:
        #     if self.get_antlikeness(r) < self.project.solver_parameters.antlikeness_threshold:
        #         continue
        #
        proj.gm.add_regions_in_t(msers, frame, fast=True)
        solver_t += time.time() - s

        # if i % 20 == 0:
        #     print
        #     print i
        #     sys.stdout.flush()

    solver.detect_split_merge_cases()

    s = time.time()
    print "#Edges BEFORE: ", proj.gm.g.num_edges()
    while True:
        num_changed1 = solver.simplify(rules=[solver.update_costs])
        num_changed2 = solver.simplify(rules=[solver.adaptive_threshold])

        if num_changed1+num_changed2 == 0:
            break

    # solver.simplify(rules=[solver.adaptive_threshold, solver.update_costs])
    # solver.simplify(rules=[solver.adaptive_threshold])
    # solver.simplify(rules=[solver.adaptive_threshold, solver.symmetric_cc_solver, solver.update_costs])
    # solver.simplify(rules=[solver.adaptive_threshold])
    # solver.simplify(rules=[solver.symmetric_cc_solver])

    print "#Edges AFTER: ", proj.gm.g.num_edges()
    solver_t += time.time() - s

    s = time.time()

    # # TODO: remove this...
    #
    # proj.solver = solver
    # proj.gm = proj.gm
    # proj.save()

    with open(proj.working_directory+'/temp/part'+str(id)+'.pkl', 'wb') as f:
        p = pickle.Pickler(f, -1)
        p.dump(proj.gm.g)
        p.dump(proj.gm.get_all_relevant_vertices())
        p.dump(proj.chm)

        # proj.gm.g.save(f)
        # p.dump(proj.gm.start_nodes())
        # p.dump(proj.gm.end_nodes())

    file_t = time.time() - s

    print "MSERS t:", round(msers_t, 2), "SOLVER t: ", round(solver_t, 2), "VIDEO t:", round(vid_t, 2), "FILE t: ", round(file_t, 2), "SUM t / frames_in_row:", round((msers_t + solver_t+vid_t+file_t)/float(frames_in_row), 2)

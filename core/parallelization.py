__author__ = 'fnaiser'
import os
import sys
import inspect

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

import time

if __name__ == '__main__':
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')
    # proj.arena_model = None
    proj.rm = RegionManager(db_name=proj.working_directory+'/temp/regions_part_'+str(id)+'.sqlite3')
    proj.chm = ChunkManager()

    S_.general.log_graph_edits = False

    vid = get_auto_video_manager(proj)
    if id*frames_in_row > 0:
        img = vid.seek_frame(id*frames_in_row)
    else:
        img = vid.next_frame()

    img = prepare_for_segmentation(img, proj)

    msers_t = 0
    solver_t = 0
    vid_t = 0
    file_t = 0

    solver = Solver(proj)
    for i in range(frames_in_row + last_n_frames):
        frame = id*frames_in_row + i

        s = time.time()
        msers = ferda_filtered_msers(img, proj, frame)
        proj.rm.add(msers)
        msers_t += time.time()-s

        s = time.time()
        img = vid.next_frame()
        if img is None:
            break

        img = prepare_for_segmentation(img, proj)
        vid_t += time.time() - s

        s = time.time()

        # TODO: test antlikeness before add regions_in_t
        # if self.antlike_filter:
        #     if self.get_antlikeness(r) < self.project.solver_parameters.antlikeness_threshold:
        #         continue
        #
        solver.gm.add_regions_in_t(msers, frame, fast=True)
        solver_t += time.time() - s

        # if i % 20 == 0:
        #     print
        #     print i
        #     sys.stdout.flush()

    s = time.time()
    print "#Edges BEFORE: ", solver.gm.g.num_edges()
    solver.simplify(rules=[solver.adaptive_threshold, solver.update_costs])
    # solver.simplify(rules=[solver.adaptive_threshold, solver.symmetric_cc_solver])
    # solver.simplify(rules=[solver.symmetric_cc_solver])

    print "#Edges AFTER: ", solver.gm.g.num_edges()
    # solver.simplify_to_chunks()
    solver_t += time.time() - s

    s = time.time()
    if not os.path.exists(proj.working_directory+'/temp'):
        os.mkdir(proj.working_directory+'/temp')

    with open(proj.working_directory+'/temp/g_simplified'+str(id)+'.gt', 'wb') as f:
        p = pickle.Pickler(f, -1)
        # solver.gm.g.save(f)
        # p.dump(solver.gm.start_nodes())
        # p.dump(solver.gm.end_nodes())

    file_t = time.time() - s

    print "MSERS t:", round(msers_t, 2), "SOLVER t: ", round(solver_t, 2), "VIDEO t:", round(vid_t, 2), "FILE t: ", round(file_t, 2), "SUM t / frames_in_row:", round((msers_t + solver_t+vid_t+file_t)/float(frames_in_row), 2)

    # for e in proj.gm.g.edges():
    #     r1 = proj.gm.region(e.source())
    #     r2 = proj.gm.region(e.target())
    #
    #     print r1, " -> ", r2
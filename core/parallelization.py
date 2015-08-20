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



if __name__ == '__main__':
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')

    S_.general.log_graph_edits = False

    vid = get_auto_video_manager(proj)
    if id*frames_in_row > 0:
        img = vid.seek_frame(id*frames_in_row)
    else:
        img = vid.next_frame()

    img = prepare_for_segmentation(img, proj)

    solver = Solver(proj)
    for i in range(frames_in_row + last_n_frames):
        frame = id*frames_in_row + i

        msers = ferda_filtered_msers(img, proj, frame)

        img = vid.next_frame()
        if img is None:
            break

        img = prepare_for_segmentation(img, proj)

        solver.add_regions_in_t(msers, frame, fast=True)

        # if i % 20 == 0:
        #     print
        #     print i
        #     sys.stdout.flush()

    # solver.simplify(first_run=True)
    # solver.simplify_to_chunks()

    solver.simplify()
    solver.simplify_to_chunks()

    if not os.path.exists(proj.working_directory+'/temp'):
        os.mkdir(proj.working_directory+'/temp')

    with open(proj.working_directory+'/temp/g_simplified'+str(id)+'.pkl', 'wb') as f:
        p = pickle.Pickler(f, -1)
        p.dump(solver.g)
        p.dump(solver.start_nodes())
        p.dump(solver.end_nodes())
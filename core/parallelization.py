__author__ = 'fnaiser'
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1, currentdir)

import sys
import math
from utils.video_manager import get_auto_video_manager
import cPickle as pickle
# import pickle
from core.graph.solver import Solver
from core.project import Project
from PyQt4 import QtGui
from gui.correction.certainty import CertaintyVisualizer


if __name__ == '__main__':
    print sys.path
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')

    from core.region.mser import get_msers_
    from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter


    #TODO: REMOVE
    min_area = proj.stats.area_median * 0.2

    vid = get_auto_video_manager(proj.video_paths)
    img = vid.seek_frame(id*frames_in_row)

    sum_ = 0

    solver = Solver(proj)
    for i in range(frames_in_row + last_n_frames):
        frame = id*frames_in_row + i

        m = get_msers_(img, frame)
        groups = get_region_groups(m)
        ids = margin_filter(m, groups)
        ids = area_filter(m, ids, min_area)
        ids = children_filter(m, ids)

        img = vid.move2_next()
        if img is None:
            break

        sum_ += len(m)

        solver.add_regions_in_t([m[id_] for id_ in ids], frame)

        print i


    solver.simplify()
    solver.simplify_to_chunks()

    if not os.path.exists(proj.working_directory+'/temp'):
        os.mkdir(proj.working_directory+'/temp')

    with open(proj.working_directory+'/temp/g_simplified'+str(id)+'.pkl', 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(solver.g)
        p.dump(solver.start_nodes())
        p.dump(solver.end_nodes())
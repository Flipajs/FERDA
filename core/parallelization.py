__author__ = 'fnaiser'
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import sys
from sklearn import svm
from utils.video_manager import get_auto_video_manager
import cPickle as pickle
from core.region.mser import get_msers_
from core.region.mser_operations import get_region_groups, margin_filter, area_filter, children_filter
from core.graph.solver import Solver
from core.project import Project
from core.settings import Settings as S_
from skimage.transform import rescale
import numpy as np

if __name__ == '__main__':
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')

    #TODO: REMOVE
    min_area = proj.stats.area_median * 0.2

    vid = get_auto_video_manager(proj.video_paths)
    img = vid.seek_frame(id*frames_in_row)
    if proj.bg_model:
        img = proj.bg_model.bg_subtraction(img)

    if proj.arena_model:
        img = proj.arena_model.mask_image(img)

    if S_.mser.img_subsample_factor > 1.0:
        img = np.asarray(rescale(img, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

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

        if proj.bg_model:
            img = proj.bg_model.bg_subtraction(img)

        if proj.arena_model:
            img = proj.arena_model.mask_image(img)

        if S_.mser.img_subsample_factor > 1.0:
            img = np.asarray(rescale(img, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

        sum_ += len(m)

        solver.add_regions_in_t([m[id_] for id_ in ids], frame)

        print
        print i
        sys.stdout.flush()

    solver.simplify()
    solver.simplify_to_chunks()

    if not os.path.exists(proj.working_directory+'/temp'):
        os.mkdir(proj.working_directory+'/temp')

    with open(proj.working_directory+'/temp/g_simplified'+str(id)+'.pkl', 'wb') as f:
        p = pickle.Pickler(f)
        p.dump(solver.g)
        p.dump(solver.start_nodes())
        p.dump(solver.end_nodes())
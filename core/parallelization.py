__author__ = 'fnaiser'
import os
import sys
import inspect
import multiprocessing
pool=multiprocessing.Pool(processes=4)
from subprocess import call
call(["hostname"])
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
import cv2
from core.region.mser_operations import children_filter

def prepare_img(proj, img):
    if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
        crop = prepare_for_segmentation(img, proj, False)
        proj.segmentation_model.set_image(crop)
        seg = proj.segmentation_model.predict()

        # make hard threshold
        if False:
            result = seg < 0.5
            result = np.asarray(result, dtype=np.uint8) * 255
        else:
            result = np.asarray((-seg * 255) + 255, dtype=np.uint8)
    else:
        result = prepare_for_segmentation(img, proj)

    return result


if __name__ == '__main__':
    print sys.argv
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')

    proj.solver_parameters.max_edge_distance_in_ant_length = 100

    if not os.path.exists(proj.working_directory+'/temp'):
        try:
            os.mkdir(proj.working_directory+'/temp')
        except OSError:
            pass

    if not proj.is_cluster():
        temp_local_path = proj.working_directory+'/temp'
    else:
        temp_local_path='/localhome/casillas/'

        if not os.path.exists(temp_local_path+proj_name):
            try:
                os.mkdir(temp_local_path+proj_name)
            except:
                print(temp_local_path+proj_name + "   was created between check and mkdir");

        temp_local_path=temp_local_path + proj_name

        if not os.path.exists(temp_local_path+'/temp'):
            try:
                os.mkdir(temp_local_path+'/temp')
            except:
                print(temp_local_path+'/temp' + "   was created between check and mkdir");

        temp_local_path=temp_local_path+'/temp'

    # check if part was computed before
    if os.path.isfile(working_dir+'/temp/part'+str(id)+'_rm.sqlite3'):
        if os.stat(working_dir+'/temp/part'+str(id)+'_rm.sqlite3')!=0:
            if os.path.isfile(working_dir+'/temp/part'+str(id)+'.pkl'):
                if os.stat(working_dir+'/temp/part'+str(id)+'.pkl')!=0:
                    import sys
                    sys.exit("Part already processed")
            
    solver = Solver(proj)
    from core.graph.graph_manager import GraphManager
    proj.gm = GraphManager(proj, proj.solver.assignment_score)
    # TODO: add global params
    proj.rm = RegionManager(db_wd=temp_local_path, db_name='part'+str(id)+'_rm.sqlite3', cache_size_limit=5)
    proj.chm = ChunkManager()
    proj.color_manager = None

    S_.general.log_graph_edits = False

    vid = get_auto_video_manager(proj)
    if id*frames_in_row > 0:
        img = vid.seek_frame(id*frames_in_row)
    else:
        img = vid.next_frame()

    if img is None:
        raise Exception("img is None, there is something wrong with frame: "+str(id*frames_in_row))

    img_gray = prepare_img(proj, img)

    msers_t = 0
    solver_t = 0
    vid_t = 0
    file_t = 0

    jj = 0
    for i in range(frames_in_row + last_n_frames):
        frame = id*frames_in_row + i

        s = time.time()
        msers = ferda_filtered_msers(img_gray, proj, frame)

        if proj.colormarks_model:
            proj.colormarks_model.assign_colormarks(proj, msers)

        proj.rm.add(msers)
        msers_t += time.time()-s

        s = time.time()

        # Check for last frame...
        if i+1 < frames_in_row + last_n_frames:
            img = vid.next_frame()
            if img is None:
                raise Exception("img is None, there is something wrong with frame: " + str(frame))

        img_gray = prepare_img(proj, img)

        vid_t += time.time() - s

        s = time.time()

        proj.gm.add_regions_in_t(msers, frame, fast=True)
        solver_t += time.time() - s

    # if proj.solver_parameters.use_emd_for_split_merge_detection():
    #     solver.detect_split_merge_cases()

    s = time.time()
    print "#Edges BEFORE: ", proj.gm.g.num_edges()
    while True:
        num_changed1 = solver.simplify(rules=[solver.update_costs])
        num_changed2 = solver.simplify(rules=[solver.adaptive_threshold])

        if num_changed1+num_changed2 == 0:
            break

    print "#Edges AFTER: ", proj.gm.g.num_edges()

    solver_t += time.time() - s

    s = time.time()

    with open(proj.working_directory+'/temp/part'+str(id)+'.pkl', 'wb') as f:
        p = pickle.Pickler(f, -1)
        p.dump(proj.gm.g)
        p.dump(proj.gm.get_all_relevant_vertices())
        p.dump(proj.chm)

    file_t = time.time() - s

    print "#Vertices: {}, #Edges: {}".format(proj.gm.g.num_vertices(), proj.gm.g.num_edges())
    print "MSERS t:", round(msers_t, 2), "SOLVER t: ", round(solver_t, 2), "VIDEO t:", round(vid_t, 2), "FILE t: ", round(file_t, 2), "SUM t / frames_in_row:", round((msers_t + solver_t+vid_t+file_t)/float(frames_in_row), 2)

    if proj.is_cluster():
        import shutil
        import glob
        for file in glob.glob(temp_local_path+'/part'+str(id)+'_rm.sqlite3'):
            shutil.move(file,working_dir+'/temp')

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


if __name__ == '__main__':
    working_dir = sys.argv[1]
    proj_name = sys.argv[2]
    id = int(sys.argv[3])
    frames_in_row = int(sys.argv[4])
    last_n_frames = int(sys.argv[5])

    f_log_name = 'id'+str(id)+'.log'
    # with open(f_log_name, 'wb') as f:
    #     f.write('init...')

    proj = Project()
    proj.load(working_dir+'/'+proj_name+'.fproj')

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
    # proj.rm = RegionManager(db_wd=temp_local_path, db_name='part'+str(id)+'_rm.sqlite3', cache_size_limit=S_.cache.region_manager_num_of_instances)
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

    # if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
    #     proj.segmentation_model.set_image(img)
    #     seg = proj.segmentation_model.predict()
    #     # img = np.asarray((-seg*255)+255, dtype=np.uint8)
    #     img = seg < 0.5
    #     img = np.asarray(img, dtype=np.uint8)*255
    # else:
    img_gray = prepare_for_segmentation(img, proj)

    msers_t = 0
    solver_t = 0
    vid_t = 0
    file_t = 0

    for i in range(frames_in_row + last_n_frames):
        frame = id*frames_in_row + i

        print frame
        s = time.time()
        msers = ferda_filtered_msers(img_gray, proj, frame)

        if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
            new_msers = []
            border = 10
            for m in msers:
                roi = m.roi()
                tl = roi.top_left_corner()
                br = roi.bottom_right_corner()

                h1 = max(0, tl[0]-border)
                h2 = min(img.shape[0]-1, br[0] + border)

                w1 = max(0, tl[1]-border)
                w2 = min(img.shape[1]-1, br[1] + border)

                crop = img[h1:h2, w1:w2, :].copy()

                proj.segmentation_model.set_image(crop)
                seg = proj.segmentation_model.predict()
                seg_img = np.asarray((-seg*255)+255, dtype=np.uint8)

                msers_ = ferda_filtered_msers(seg_img, proj, frame)
                for m in msers_:
                    # update offsets
                    offset = np.array([h1, w1])
                    for it in m.pts_rle_:
                        it['line'] += h1
                        it['col1'] += w1
                        it['col2'] += w1

                    m.pts_ += offset
                    m.pts_rle_
                    m.contour_ += offset
                    m.centroid_ += offset
                    if hasattr(m, 'roi_') and m.roi_ is not None:
                        m.roi_.y_ += h1
                        m.roi_.x_ += w1
                        m.roi_.y_max_ += h1
                        m.roi_.x_max_ += w1
                    new_msers.append(m)

            msers = new_msers

        if proj.colormarks_model:
            proj.colormarks_model.assign_colormarks(proj, msers)

        proj.rm.add(msers)
        msers_t += time.time()-s

        s = time.time()
        print frame
        img = vid.next_frame()
        if img is None:
            raise Exception("img is None, there is something wrong with frame: " + str(frame))

        # if hasattr(proj, 'segmentation_model') and proj.segmentation_model is not None:
        #     proj.segmentation_model.set_image(img)
        #     seg = proj.segmentation_model.predict()
        #     img = seg < 0.5
        #     img = np.asarray(img, dtype=np.uint8)*255
        # else:
        img_gray = prepare_for_segmentation(img, proj)

        vid_t += time.time() - s

        s = time.time()

        proj.gm.add_regions_in_t(msers, frame, fast=True)
        solver_t += time.time() - s

    if proj.solver_parameters.use_emd_for_split_merge_detection():
        solver.detect_split_merge_cases()

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

    print "MSERS t:", round(msers_t, 2), "SOLVER t: ", round(solver_t, 2), "VIDEO t:", round(vid_t, 2), "FILE t: ", round(file_t, 2), "SUM t / frames_in_row:", round((msers_t + solver_t+vid_t+file_t)/float(frames_in_row), 2)

    if proj.is_cluster():
        import shutil
        import glob
        for file in glob.glob(temp_local_path+'/part'+str(id)+'_rm.sqlite3'):
            shutil.move(file,working_dir+'/temp')

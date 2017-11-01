import os
import cv2
from core.project.project import Project
from utils.gt.gt import GT
from utils.img import get_safe_selection
from core.graph.region_chunk import RegionChunk
from utils.video_manager import get_auto_video_manager
import tqdm
import h5py

if __name__ == '__main__':
    # OUT_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_zebrafish'
    OUT_DIR = '/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_sowbug'
    MARGIN = 1.25

    try:
        os.mkdir(OUT_DIR)
    except:
        pass

    p = Project()
    # p.load('/Users/flipajs/Documents/wd/FERDA/Cam1')
    # p.load('/Users/flipajs/Documents/wd/FERDA/Camera3_new')
    # p.load('/Users/flipajs/Documents/wd/FERDA/zebrafish_new')
    p.load('/Users/flipajs/Documents/wd/FERDA/Sowbug3_new')


    # p.video_crop_model = {}
    # p.video_crop_model['y1'] = 0
    # p.video_crop_model['x1'] = 0
    # p.video_crop_model['frames'] = 0
    # p.save()

    GT = GT()
    # path = '/Users/flipajs/Documents/dev/ferda/data/GT/Cam1_.pkl'
    # path = '/Users/flipajs/Documents/dev/ferda/data/GT/5Zebrafish_nocover_22min.pkl'
    path = '/Users/flipajs/Documents/dev/ferda/data/GT/Sowbug.pkl'
    GT.load(path)
    try:
        GT.set_offset(y=p.video_crop_model['y1'],
                           x=p.video_crop_model['x1'],
                           frames=p.video_start_t)
    except:
        pass

    major_axis = p.stats.major_axis_median
    print major_axis
    # probably wrong in Cam1_... comment when using different dataset
    major_axis = 36

    offset = major_axis * MARGIN
    vm = get_auto_video_manager(p)

    examples = []
    for i in range(len(p.animals)):
        try:
            os.mkdir(OUT_DIR+'/'+str(i))
        except:
            pass

        examples.append(0)

    for t in tqdm.tqdm(p.chm.chunk_gen(), total=len(p.chm)):
        if t.is_single():
            id = GT.tracklet_id_set_without_checks(t, p)
            if len(id):
                id = id[0]

                examples[id] += len(t)

                rch = RegionChunk(t, p.gm, p.rm)
                for r in rch.regions_gen():
                    img = vm.get_frame(r.frame())
                    y, x = r.centroid()
                    crop = get_safe_selection(img, y-offset, x-offset, 2*offset, 2*offset)
                    cv2.imwrite(OUT_DIR+'/'+str(id)+'/'+str(r.id())+'.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


    for i, num in enumerate(examples):
        print "ID: {} #{}".format(i, num)


from __future__ import print_function
from core.project.project import Project
from utils.gt.gt import GT
import tqdm


def eval_cardinality_classification(p, gt):
    num_correct_tracklets = 0
    num_wrong_tracklets = 0

    num_correct_frames = 0
    num_wrong_frames = 0

    wrong_tracklets = []

    print("starting... ")
    for tracklet in tqdm.tqdm(p.chm.tracklet_gen(), total=len(p.chm)):
        cardinality_class, _ = gt.get_class_and_id(tracklet, p)

        if tracklet.segmentation_class == cardinality_class:
            num_correct_tracklets += 1
            num_correct_frames += len(tracklet)
        else:
            num_wrong_tracklets += 1
            num_wrong_frames += len(tracklet)
            wrong_tracklets.append((tracklet, cardinality_class))

    print("#wrong tracklets: {}({:.2%}), #wrong frames: {}({:.2%})".format(
        num_wrong_tracklets,
        (num_wrong_tracklets / float(num_wrong_tracklets + num_correct_tracklets)),
        num_wrong_frames,
          (num_wrong_frames) / float(num_correct_frames + num_wrong_frames)
    ))

    for t, c in wrong_tracklets:
        print("{} {} {}".format(t.id(), t.segmentation_class, c))


if __name__ == '__main__':
    path = '/Users/flipajs/Documents/wd/FERDA/new/180713_1633_Cam1_clip_initial'
    path = '/Users/flipajs/Documents/wd/FERDA/VAIB-conference-2018/Cam1_clip_arena_fixed'

    # path = '/Users/flipajs/Documents/wd/FERDA/VAIB-conference-2018/Sowbug3-fixed-segmentation'
    # path = '/datagrid/ferda/projects/tmp/180713_1631_Sowbug3_cut_initial'

    gt_path = 'data/GT/Cam1_clip.avi.pkl'
    # gt_path = 'data/GT/Sowbug3.pkl'

    p = Project()
    p.load(path)

    gt = GT()
    gt.load(gt_path)
    gt.set_offset(y=p.video_crop_model['y1'],
                  x=p.video_crop_model['x1'],
                  frames=p.video_start_t)

    eval_cardinality_classification(p, gt)
from core.project.project import Project
from utils.gt.gt import GT

def eval_cardinality_classification(p, gt):
    num_correct_tracklets = 0
    num_wrong_tracklets = 0

    num_correct_frames = 0
    num_wrong_frames = 0

    wrong_tracklets = []

    for tracklet in p.chm.tracklet_gen():
        cardinality_class, _ = gt.get_class_and_id(tracklet, p)

        if tracklet.get_cardinality() == cardinality_class:
            num_correct_tracklets += 1
            num_correct_frames += len(tracklet)
        else:
            num_wrong_tracklets += 1
            num_wrong_frames += len(tracklet)
            wrong_tracklets.append(tracklet)


    print("#wrong tracklets: {}({:.2%}), #wrong frames: {}({:.2%})".format(
        num_wrong_tracklets,
        (num_wrong_tracklets / float(num_wrong_tracklets + num_correct_tracklets)),
        num_wrong_frames,
          (num_wrong_frames) / float(num_correct_frames + num_wrong_frames)
    ))

if __name__ == '__main__':
    path = '/Users/flipajs/Documents/wd/FERDA/VAIB-conference-2018/Cam1_clip_arena_fixed'
    gt_path = 'data/GT/Cam1_clip.avi'

    p = Project()
    p.load(path)

    gt = GT()
    gt.load(gt_path)

    eval_cardinality_classification(p, gt_path)
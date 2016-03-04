import sys
sys.path.append('/Users/flipajs/Documents/dev/fastext/toolbox')
from ft import FASTex
import math
import cv2
from utils.video_manager import get_auto_video_manager
from core.project.project import Project

import matplotlib.pyplot as plt
import numpy

vm = None
ft = None
fig = None

def process_key_press(event):
    if event.key == 'n':
        show_next()


def show_next():
    global vm
    global ft

    plt.close()
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', process_key_press)

    img_c = vm.next_frame()
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)

    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REPLICATE )
    ft.getCharSegmentations(img, '', 'base')


    keypoints = ft.getLastDetectionKeypoints()
    strokes = []

    sizes_ = [10, 17, 25]

    scales = ft.getImageScales()
    scs_ = []
    for kp, i in zip(keypoints, range(len(keypoints))):
        s_ = ft.getKeypointStrokes(i) * (1.0 / scales[kp[2]])
        scs_.append(kp[2])
        strokes.append(s_)

    plt.imshow(img_c)

    for s_, sc_ in zip(strokes, scs_):
        if sc_ != 2:
            continue

        plt.scatter(s_[:, 0] - border, s_[:, 1] - border, s=int(sizes_[int(sc_)]), c=numpy.random.rand(3,))

    for s_, sc_ in zip(strokes, scs_):
        if sc_ != 1:
            continue

        plt.scatter(s_[:, 0] - border, s_[:, 1] - border, s=int(sizes_[int(sc_)]), c=numpy.random.rand(3,))

    for s_, sc_ in zip(strokes, scs_):
        if sc_ != 0:
            continue

        plt.scatter(s_[:, 0] - border, s_[:, 1] - border, s=int(sizes_[int(sc_)]), c=numpy.random.rand(3,))

    mng = plt.get_current_fig_manager()
    plt.show()



if __name__ == "__main__":
    p = Project()
    p.video_paths = ['/Users/flipajs/Documents/wd/Cam1_clip.avi']
    vm = get_auto_video_manager(p)

    scaleFactor = 1.4
    nlevels = 3
    edgeThreshold = 13
    keypointTypes = 2
    kMin = 9
    kMax = 11
    erode = 0
    segmentGrad = 0
    minCompSize = 4
    process_color = 0
    segmDeltaInt = 1
    min_tupple_top_bottom_angle = math.pi / 2
    maxSpaceHeightRatio = -1
    createKeypointSegmenter = 1

    params_dict = {
        'scaleFactor': scaleFactor,
        'nlevels': nlevels,
        'edgeThreshold': edgeThreshold,
        'keypointTypes': keypointTypes,
        'kMin': kMin,
        'kMax': kMax,
        'erode': erode,
        'segmentGrad': segmentGrad,
        'minCompSize': minCompSize,
        'process_color': process_color,
        'segmDeltaInt': segmDeltaInt,
        'min_tupple_top_bottom_angle': min_tupple_top_bottom_angle,
        'maxSpaceHeightRatio': maxSpaceHeightRatio,
        'createKeypointSegmenter': createKeypointSegmenter
    }

    process_color = 0

    border = 5

    ft = FASTex(edgeThreshold=edgeThreshold,
                createKeypointSegmenter=createKeypointSegmenter,
                nlevels=nlevels,
                minCompSize=minCompSize,
                keypointTypes=keypointTypes)

    show_next()
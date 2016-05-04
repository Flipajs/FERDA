from features import get_curvature_kp
from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from matplotlib import pyplot as plt
import cv2
import numpy as np
from core.region.mser import ferda_filtered_msers
import scipy.ndimage as ndimage


def data_cam2():
    #Cam2
    collisions = [
        {'s': [6, 7], 'm': 1, 'e': [18, 23]},
        {'s': [64, 65], 'm': 50, 'e': [48, 62]},
        {'s': [111, 112, 120], 'm': 132, 'e': [123, 124, 116]},
                  ]

    return collisions


if __name__ == '__main__':
    p = Project()
    data = data_cam2()
    name = 'Cam2/cam2.fproj'
    wd = '/Users/flipajs/Documents/wd/gt/'
    p.load(wd+name)
    vm = get_auto_video_manager(p)

    d = data[2]

    rs1 = p.gm.region(p.chm[d['s'][0]].end_vertex_id())
    rs2 = p.gm.region(p.chm[d['s'][1]].end_vertex_id())
    rs3 = p.gm.region(p.chm[d['s'][2]].end_vertex_id())

    plt.ion()

    get_curvature_kp(rs1.contour_without_holes(), True)
    get_curvature_kp(rs2.contour_without_holes(), True)
    get_curvature_kp(rs3.contour_without_holes(), True)

    r = p.gm.region(p.chm[d['m']].start_vertex_id())
    get_curvature_kp(r.contour_without_holes(), True)

    plt.waitforbuttonpress()
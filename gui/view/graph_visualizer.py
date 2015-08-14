__author__ = 'flipajs'


# for some reason on MAC machines it is necessary to import it, else there is problem with loading MSER DLLs

import sys

from PyQt4 import QtGui
import numpy as np
from skimage.transform import rescale

from scripts.region_graph3 import NodeGraphVisualizer, visualize_nodes
from core.settings import Settings as S_
from utils.video_manager import get_auto_video_manager, optimize_frame_access
from core.project.project import Project
from utils.misc import is_flipajs_pc


def call_visualizer(t_start, t_end, project, solver, min_chunk_len, update_callback=None, node_size=30, node_margin=0.1):
    solver = solver
    if t_start == t_end == -1:
        sub_g = solver.g
    else:
        nodes = []
        for n in solver.g.nodes():
            if t_start <= n.frame_ < t_end:
                nodes.append(n)

        sub_g = solver.g.subgraph(nodes)

    vid = get_auto_video_manager(project)
    regions = {}

    nodes = []
    chunks = set()
    for n in sub_g.nodes():
        is_ch, t_reversed, ch = solver.is_chunk(n)
        if is_ch:
            if ch.length() >= min_chunk_len:
                nodes.append(n)
                chunks.add(ch)

    optimized = optimize_frame_access(nodes)

    i = 0
    num_parts = 100
    part_ = len(optimized) / num_parts + 1

    # TODO: optimize for same frames...
    cache = {}

    for n, seq, _ in optimized:
        if n.frame_ in regions:
            regions[n.frame_].append(n)
        else:
            regions[n.frame_] = [n]

        # TODO: use cache again and invalidate it only in case when node margin has changed...
        # if 'img' not in solver.g.node[n]:
        if True:
            im = vid.get_frame(n.frame_, sequence_access=seq)

            sf = project.other_parameters.img_subsample_factor
            if sf > 1.0:
                if n.frame_ not in cache:
                    im = np.asarray(rescale(im, 1/sf) * 255, dtype=np.uint8)
                    cache[n.frame_] = im
                else:
                    im = cache[n.frame_]

            solver.g.node[n]['img'] = visualize_nodes(im, n, margin=node_margin)

        i += 1

        if update_callback is not None and i % part_ == 0:
            update_callback(i / float(len(optimized)))

    ngv = NodeGraphVisualizer(solver, solver.g, regions, list(chunks), node_size=node_size)
    ngv.visualize()

    return ngv


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    if is_flipajs_pc():
        project = Project()
        project.load('/Users/flipajs/Documents/wd/eight_test/test.fproj')
        # project.load('/Users/flipajs/Documents/wd/colonies_crop1/colonies.fproj')
    else:
        # EDIT HERE....

        project = Project()
        project.load('/home/simon/Documents/res/c3_1h30/c3_1h30.fproj')

    # from utils.video_manager import get_auto_video_manager
    # vid = get_auto_video_manager(project)
    # img = vid.next_frame()
    #
    # from pylab import ogrid, gca, show
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.cbook import get_sample_data
    # from matplotlib._png import read_png
    #
    # x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
    # ax = gca(projection='3d')
    # ax.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=np.asarray(img, dtype=np.float)/255.)
    #
    # show()

    # ex = call_visualizer(-1, -1, project)
    ex = call_visualizer(0, 700, project, project.saved_progress['solver'], 10)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()

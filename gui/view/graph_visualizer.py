__author__ = 'flipajs'


# for some reason on MAC machines it is necessary to import it, else there is problem with loading MSER DLLs

import sys

from PyQt4 import QtGui
import numpy as np
from skimage.transform import rescale

from scripts.region_graph3 import NodeGraphVisualizer, visualize_nodes
from core.settings import Settings as S_
from utils.video_manager import get_auto_video_manager, optimize_frame_access, optimize_frame_access_vertices
from core.project.project import Project
from utils.misc import is_flipajs_pc


def call_visualizer(t_start, t_end, project, solver, min_chunk_len, update_callback=None, node_size=30, node_margin=0.1, show_in_visualizer_callback=None, reset_cache=True, show_vertically=False):
    if t_start == t_end == -1:
        sub_g = solver.g
    else:
        g_nodes = []
        for vertex in project.gm.get_all_relevant_vertices():
            r = project.gm.region(vertex)
            if t_start <= r.frame_ < t_end:
                g_nodes.append((r, vertex))

    vid = get_auto_video_manager(project)
    frames_d = {}

    regions = []
    chunks = set()
    vertices = []
    region_vertices_mapping = {}
    for r, vertex in g_nodes:
        ch, _ = project.gm.is_chunk(vertex)
        if ch:
            if ch.length() >= min_chunk_len:
                regions.append(r)
                vertices.append(vertex)
                region_vertices_mapping[r] = vertex
                chunks.add(ch)

    optimized = optimize_frame_access_vertices(vertices, project)

    i = 0
    num_parts = 100
    part_ = len(optimized) / num_parts + 1

    # TODO: optimize for same frames...
    cache = {}


    images = {}

    for v, seq, _ in optimized:
        r = project.gm.region(v)
        if r.frame_ in frames_d:
            frames_d[r.frame_].append(r)
        else:
            frames_d[r.frame_] = [r]

        if reset_cache or 'img' not in solver.g.node[r]:
            im = vid.get_frame(r.frame_, sequence_access=seq)

            sf = project.other_parameters.img_subsample_factor
            if sf > 1.0:
                if r.frame_ not in cache:
                    im = np.asarray(rescale(im, 1/sf) * 255, dtype=np.uint8)
                    cache[r.frame_] = im
                else:
                    im = cache[r.frame_]

            # TODO: optimize... and add opacity parameter
            ch, _ = project.gm.is_chunk(v)
            c = (ch.color.blue(), ch.color.green(), ch.color.red(), 0.9)

            images[r] = visualize_nodes(im, r, margin=node_margin, color=c)

        i += 1

        if update_callback is not None and i % part_ == 0:
            update_callback(i / float(len(optimized)))

    ngv = NodeGraphVisualizer(solver, project.gm.g, frames_d, list(chunks), images, region_vertices_mapping, node_size=node_size, show_in_visualize_callback=show_in_visualizer_callback, show_vertically=show_vertically)
    ngv.visualize()

    return ngv


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    if is_flipajs_pc():
        project = Project()
        # project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
        project.load('/Users/flipajs/Documents/wd/GT/C210/C210.fproj')
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
    ex = call_visualizer(0, 700, project, project.solver, 10)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()

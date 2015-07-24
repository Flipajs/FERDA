__author__ = 'flipajs'

import sys
from PyQt4 import QtGui, QtCore
from scripts.region_graph2 import NodeGraphVisualizer, visualize_nodes
from core.settings import Settings as S_
from utils.video_manager import get_auto_video_manager, optimize_frame_access
import numpy as np
from skimage.transform import rescale
from core.project import Project


def call_visualizer(t_start, t_end, project):
    solver = project.saved_progress['solver']
    if t_start == t_end == -1:
            sub_g = solver.g
    else:
        nodes = []
        for n in solver.g.nodes():
            if t_start <= n.frame_ < t_end:
                nodes.append(n)

        sub_g = solver.g.subgraph(nodes)

    vid = get_auto_video_manager(project.video_paths)
    regions = {}

    optimized = optimize_frame_access(sub_g.nodes())

    for n, seq, _ in optimized:
        if n.frame_ in regions:
            regions[n.frame_].append(n)
        else:
            regions[n.frame_] = [n]

        if 'img' not in solver.g.node[n]:
            if seq:
                while vid.frame_number() < n.frame_:
                    vid.move2_next()

                im = vid.img()
            else:
                im = vid.seek_frame(n.frame_)

            if S_.mser.img_subsample_factor > 1.0:
                im = np.asarray(rescale(im, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

            solver.g.node[n]['img'] = visualize_nodes(im, n)
            sub_g.node[n]['img'] = solver.g.node[n]['img']

    ngv = NodeGraphVisualizer(sub_g, [], regions)
    ngv.visualize()

    return ngv



if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    project = Project()
    project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')

    ex = call_visualizer(500, 600, project)
    ex.showMaximized()

    app.exec_()
    app.deleteLater()
    sys.exit()

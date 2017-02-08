import h5py
import numpy as np

from itertools import izip
from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from utils.misc import print_progress
from scipy.spatial.distance import cdist

if __name__ == '__main__':
    p = Project()
    p.load('/Volumes/Seagate Expansion Drive/FERDA-projects/Bjoern-Cam1/cam1.fproj')

    vm = get_auto_video_manager(p)

    MAX_D = 3*2*p.stats.major_axis_median
    MAX_D2 = MAX_D*MAX_D

    NUM_FRAMES = 2

    nodes = []
    node_probs = []

    M = 100000000
    edges = np.empty((M, 2), dtype=np.uint32)
    edge_costs = np.empty((M, 2), dtype=np.float)

    ei = 0

    prev_xs, prev_ys = None, None
    prev_offset = 0

    max_frame = min(NUM_FRAMES, vm.total_frame_count())
    for frame in range(max_frame):
        img = vm.next_frame()

        p.segmentation_model.set_image(img)
        prob_map = p.segmentation_model.predict()

        #remove borders
        b = 5
        prob_map = prob_map[b:-b, b:-b].copy()

        ys, xs = np.where(prob_map)
        print "\nprocessing frame: {}, #pxs: {}".format(frame, len(ys))

        j = 0
        for y, x in izip(ys, xs):
            j += 1
            print_progress(j, len(ys))

            node_probs.append(prob_map[y, x])
            nodes.append([frame, y, x])

            if prev_xs is not None:
                dists = cdist(np.array([[y, x]]), np.vstack((prev_ys, prev_xs)).T)
                for prev_id, (yy, xx) in enumerate(izip(prev_ys, prev_xs)):
                    d = dists[0, prev_id]
                    if d < MAX_D:
                        # FROM ID, TO ID
                        edges[ei, 0] = prev_offset + prev_id
                        edges[ei, 1] = len(nodes) - 1
                        edge_costs[ei] = d

                        ei += 1

        if prev_xs is not None:
            prev_offset += len(xs)

        prev_xs, prev_ys = xs, ys

    print len(nodes)
    print len(node_probs)
    print len(edges)
    print len(edge_costs)

    f = h5py.File("/Volumes/Seagate Expansion Drive/FERDA-projects/Bjoern-Cam1/out_nodes.hdf5", "w")
    f.create_dataset("nodes", data=np.array(nodes, dtype=np.uint), dtype=np.uint32)
    f.close()

    f = h5py.File("/Volumes/Seagate Expansion Drive/FERDA-projects/Bjoern-Cam1/out_node_probs.hdf5", "w")
    f.create_dataset("node_probs", data=np.array(node_probs, dtype=np.float), dtype=np.float32)
    f.close()

    f = h5py.File("/Volumes/Seagate Expansion Drive/FERDA-projects/Bjoern-Cam1/out_edges.hdf5", "w")
    f.create_dataset("edges", data=edges[:ei, :].copy(), dtype=np.uint32)
    f.close()

    f = h5py.File("/Volumes/Seagate Expansion Drive/FERDA-projects/Bjoern-Cam1/out_edge_costs.hdf5", "w")
    f.create_dataset("edge_costs", data=edge_costs[:ei].copy(), dtype=np.float32)
    f.close()
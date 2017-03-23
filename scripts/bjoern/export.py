import h5py
import numpy as np

from itertools import izip
from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from utils.misc import print_progress
from scipy.spatial.distance import cdist
from skimage.transform import rescale
import matplotlib.pyplot as plt
import random
import cv2
import os

def test_output(wd, vm, p, k):
    with h5py.File(wd + "/out_vertices.hdf5", 'r') as hf:
        # uint32 (V x [frame, y, x])
        vertices = hf['vertices'][:]

    with h5py.File(wd + "/out_vertex_probs.hdf5", 'r') as hf:
        # float16 (V x [prob])
        vertex_probs = hf['vertex_probs'][:]

    with h5py.File(wd + "/out_edges.hdf5", 'r') as hf:
        # uint32 (E x [vertex_i, vertex_j])
        edges = hf['edges'][:]

    with h5py.File(wd + "/out_edge_probs.hdf5", 'r') as hf:
        # float16 (E x [prob])
        edge_probs = hf['edge_probs'][:]

    # f = h5py.File(wd + "/out_vertices_u.hdf5", "w")
    # f.create_dataset("vertices", data=np.array(vertices, dtype=np.uint), dtype=np.uint32)
    # f.close()
    #
    # f = h5py.File(wd + "/out_vertex_probs_u.hdf5", "w")
    # f.create_dataset("vertex_probs", data=np.array(vertex_probs, dtype=np.float16), dtype=np.float16)
    # f.close()
    #
    # f = h5py.File(wd + "/out_edges_u.hdf5", "w")
    # f.create_dataset("edges", data=edges, dtype=np.uint32)
    # f.close()
    #
    # f = h5py.File(wd + "/out_edge_probs_u.hdf5", "w")
    # f.create_dataset("edge_probs", data=edge_probs, dtype=np.float16)
    # f.close()

    edges_i = random.sample(range(0, edges.shape[0]), 1000)

    plt.ion()

    for i in edges_i:
        vi = vertices[edges[i, 0]]
        vpi = vertex_probs[edges[i, 0]]
        vj = vertices[edges[i, 1]]
        vpj = vertex_probs[edges[i, 1]]
        ep = edge_probs[i]

        img1 = vm.get_frame(vi[0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if p.other_parameters.img_subsample_factor > 1.0:
            img1 = rescale(img1, 1 / p.other_parameters.img_subsample_factor)

        plt.figure(1)
        plt.imshow(img1)
        plt.hold(True)
        y1, x1 = vi[1], vi[2]
        y2, x2 = vj[1], vj[2]

        d = ((int(y1) - y2) ** 2 + (int(x1) - x2) ** 2) ** 0.5

        plt.scatter(x1, y1)

        if vi[0] == vj[0]:
            plt.title('pi: {:.2f} pj: {:.2f} ep: {:.2f}, ed: {:.2f}'.format(vpi, vpj, ep, ep-np.e**(-k*d)))
            plt.scatter(x2, y2)
            plt.hold(False)

            try:
                plt.close(2)
            except:
                pass
        else:
            plt.hold(False)
            plt.title('pi: {:.2f} ep: {:.2f}, epd: {:.2f}'.format(vpi, ep, ep - np.e**(-k*d)))

            img2 = vm.get_frame(vj[0])
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            if p.other_parameters.img_subsample_factor > 1.0:
                img2 = rescale(img2, 1 / p.other_parameters.img_subsample_factor)

            fig2 = plt.figure(2)
            plt.imshow(img2)
            plt.title('pj: {:.2f}'.format(vpj))
            plt.hold(True)
            plt.scatter(x2, y2)
            plt.hold(False)

        plt.show()
        plt.waitforbuttonpress()
        plt.clf()

    print "test"


if __name__ == '__main__':
    wds = []
    wd = '/Users/flipajs/Documents/wd/FERDA/k-tracking/ants-cam1/'

    p = Project()
    p.load(wd)

    vm = get_auto_video_manager(p)

    MAX_D = 3 * 2 * p.stats.major_axis_median

    import math

    k = -math.log(0.5)/(2*p.stats.major_axis_median/p.other_parameters.img_subsample_factor)
    print "k: {:.3f}".format(k)

    MAX_D /= p.other_parameters.img_subsample_factor
    MAX_D2 = MAX_D * MAX_D
    print MAX_D

    test_output(wd, vm, p, k)

    DO_EXPORT = False
    if DO_EXPORT:
        NUM_FRAMES = 100

        vertices = []
        vertex_probs = []

        M = 1000000000
        edges = np.empty((M, 2), dtype=np.uint32)
        edge_probs = np.empty((M, ), dtype=np.float)

        ei = 0

        prev_xs, prev_ys = None, None
        current_offset = 0
        prev_offset = 0

        try:
            os.mkdir(wd+'/imgs')
        except:
            pass

        max_frame = min(NUM_FRAMES, vm.total_frame_count())
        for frame in range(max_frame):
            img = vm.next_frame()

            p.segmentation_model.set_image(img)
            prob_map = p.segmentation_model.predict()

            #remove borders
            b = 5
            prob_map = prob_map[b:-b, b:-b].copy()

            if p.other_parameters.img_subsample_factor > 1.0:
                prob_map = rescale(prob_map, 1 / p.other_parameters.img_subsample_factor)

            if p.other_parameters.img_subsample_factor > 1.0:
                img = np.asarray(rescale(img, 1 / p.other_parameters.img_subsample_factor) * 255, dtype=np.uint8)

            s_frame = str(frame)
            while len(s_frame) < 4:
                s_frame = '0' + s_frame

            cv2.imwrite(wd+'/imgs/'+s_frame+'.jpg', img)

            ys, xs = np.where(prob_map)
            print "\nprocessing frame: {}, #pxs: {}".format(frame, len(ys))
            i = 0

            if prev_ys is not None:
                current_offset += len(prev_ys)

            for y, x in izip(ys, xs):
                print_progress(i+1, len(ys))

                vertex_probs.append(prob_map[y, x])
                vertices.append([frame, y, x])

                for j in range(i + 1, len(ys)):
                    yy, xx = ys[j], xs[j]

                    d = ((y - yy) ** 2 + (x - xx) ** 2) ** 0.5

                    if d < MAX_D:
                        edges[ei, 0] = current_offset + i
                        edges[ei, 1] = current_offset + j
                        edge_probs[ei] = np.e ** (-k * d)

                        ei += 1

                if prev_xs is not None:
                    dists = cdist(np.array([[y, x]]), np.vstack((prev_ys, prev_xs)).T)
                    for prev_id, (yy, xx) in enumerate(izip(prev_ys, prev_xs)):
                        d = dists[0, prev_id]
                        if d < MAX_D:
                            # FROM ID, TO ID
                            edges[ei, 0] = prev_offset + prev_id
                            edges[ei, 1] = len(vertices) - 1
                            edge_probs[ei] = np.e**(-k*d)

                            ei += 1

                i += 1

            if prev_xs is not None:
                prev_offset += len(prev_xs)
                print prev_offset

            prev_xs, prev_ys = xs, ys

        edges = edges[:ei, :]
        edge_probs = edge_probs[:ei]

        print len(vertices)
        print len(vertex_probs)
        print len(edges)
        print len(edge_probs)

        f = h5py.File(wd+"/out_vertices.hdf5", "w")
        f.create_dataset("vertices", data=np.array(vertices, dtype=np.uint), dtype=np.uint32, compression="gzip", compression_opts=9)
        f.close()

        f = h5py.File(wd+"/out_vertex_probs.hdf5", "w")
        f.create_dataset("vertex_probs", data=np.array(vertex_probs, dtype=np.float16), dtype=np.float16, compression="gzip", compression_opts=9)
        f.close()

        f = h5py.File(wd+"/out_edges.hdf5", "w")
        f.create_dataset("edges", data=edges, dtype=np.uint32, compression="gzip", compression_opts=9)
        f.close()

        f = h5py.File(wd+"/out_edge_probs.hdf5", "w")
        f.create_dataset("edge_probs", data=edge_probs, dtype=np.float16, compression="gzip", compression_opts=9)
        f.close()

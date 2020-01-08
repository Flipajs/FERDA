import numpy as np
from pkg_resources import resource_filename, Requirement
import pickle as pickle
import math

class ColorNaming:
    w2c = None

    def __init__(self):
        pass

    @staticmethod
    def __load_w2c_pkl():
        with open(resource_filename(__name__, "data/w2c.pkl")) as f:
            return pickle.load(f)

    @staticmethod
    def im2colors(im, out_type='color_names'):
        """
        out_type:
            'color_names': returns np.array((im.shape[0], im.shape[1]), dtype=np.uint8) with ids of one of 11 colors
            'probability_vector': returns np.array((im.shape[0], im.shape[1], 11), stype=np.float) with probability
                of each color

        NOTE: first call might take a while as the lookup table is being loaded...

        :param im:
        :param w2c:
        :param out_type:
        :return:
        """

        # black, blue, brown, gray,
        # green, orange, pink, purple
        # red, white, yellow
        # color_values = {[0 0 0], [0 0 1], [.5 .4 .25], [.5 .5 .5],
        #                 [0 1 0], [1 .8 0], [1 .5 1], [1 0 1],
        #                 [1 0 0], [1 1 1], [1 1 0]};

        if ColorNaming.w2c is None:
            ColorNaming.w2c = ColorNaming.__load_w2c_pkl()

        im = np.asarray(im, dtype=np.float)

        h, w = im.shape[0], im.shape[1]

        RR = im[:,:, 0].ravel()
        GG = im[:,:, 1].ravel()
        BB = im[:,:, 2].ravel()

        index_im = np.asarray(np.floor(RR / 8)+32 * np.floor(GG / 8)+32 * 32 * np.floor(BB / 8), dtype=np.uint)

        if out_type == 'colored_image':
            pass
        elif out_type == 'probability_vector':
            out = ColorNaming.w2c[index_im].reshape((h, w, 11))
        else:
            w2cM = np.argmax(ColorNaming.w2c, axis=1)
            out = np.asarray(w2cM[index_im], dtype=np.uint8)
            out.shape = (h, w)

        return out


def __mat2pkl(path, name):
    from scipy.io import loadmat
    import pickle as pickle

    w2c = loadmat(path+'/'+name+'.mat')['w2c']
    with open(path+'/'+name+'.pkl', 'w') as f:
        pickle.dump(w2c, f)


def im2colors(im, out_type='color_names'):
    return ColorNaming.im2colors(im, out_type)

if __name__ == '__main__':
    import pickle as pickle
    from scipy.misc import imread

    # __mat2pkl('data', 'w2c')

    im = imread('data/car.jpg')

    # load lookup table
    with open('data/w2c.pkl') as f:
        w2c = pickle.load(f)

    import numpy as np
    import time
    import matplotlib.pyplot as plt

    color_values = [[0, 0, 0], [0, 0, 1], [.5, .4, .25], [.5, .5, .5], [0, 1, 0], [1, .8, 0], [1, .5, 1], [1, 0, 1],
                    [1, 0, 0], [1, 1, 1], [1, 1, 0]]

    edge_len = math.ceil((255**2 + 128**2)**0.5)
    edge_len_i = int(edge_len)
    im = np.zeros((edge_len_i, edge_len_i, 3), dtype=np.uint8)
    w2cM = np.argmax(w2c, axis=1)

    alpha = math.atan(128.0/255.0)
    beta = math.atan(255.0/(255.0*2**0.5))

    for bp in range(edge_len_i):
        for gp in range(edge_len_i):
            b = min(math.cos(alpha) * bp, 255)
            g = min(math.cos(alpha) * gp, 255)

            i = [edge_len, edge_len]
            rp = (np.dot(i, [bp, gp]) / np.linalg.norm(i)**2) * edge_len
            r = min(math.cos(beta) * rp, 255)

            id_ = int(np.floor(r / 8)+32 * np.floor(g / 8)+32 * 32 * np.floor(b / 8))
            im[bp, gp, :] = np.array(color_values[w2cM[id_]]) * 255

    plt.figure(1)
    plt.imshow(im)
    plt.ylabel('B')
    plt.xlabel('G')

    im = np.zeros((edge_len_i, edge_len_i, 3), dtype=np.uint8)

    for rp in range(edge_len_i):
        for gp in range(edge_len_i):
            r = min((math.cos(alpha) * rp), 255)
            g = min((math.cos(alpha) * gp), 255)

            i = [edge_len, edge_len]
            bp = (np.dot(i, [rp, gp]) / np.linalg.norm(i)**2) * edge_len
            b = min(math.cos(beta) * bp, 255)

            id_ = int(np.floor(r / 8)+32 * np.floor(g / 8)+32 * 32 * np.floor(b / 8))
            im[rp, gp, :] = np.array(color_values[w2cM[id_]]) * 255

    plt.figure(2)
    plt.imshow(im)
    plt.ylabel('R')
    plt.xlabel('G')

    im = np.zeros((edge_len_i, edge_len_i, 3), dtype=np.uint8)

    for bp in range(edge_len_i):
        for rp in range(edge_len_i):
            b = min((math.cos(alpha) * bp), 255)
            r = min((math.cos(alpha) * rp), 255)

            i = [edge_len, edge_len]
            gp = (np.dot(i, [bp, rp]) / np.linalg.norm(i)**2) * edge_len
            g = min(math.cos(beta) * gp, 255)

            id_ = int(np.floor(r / 8)+32 * np.floor(g / 8)+32 * 32 * np.floor(b / 8))
            im[bp, rp, :] = np.array(color_values[w2cM[id_]]) * 255

    plt.figure(3)
    plt.imshow(im)
    plt.ylabel('B')
    plt.xlabel('R')

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(0)
    ax = Axes3D(fig)
    ax.hold(True)

    rs = []
    gs = []
    bs = []
    cs = []
    step = 7
    for r in range(0, 255, step):
        print(r)
        for g in range(0, 255, step):
            for b in range(0, 255, step):
                id_ = int(np.floor(r / 8) + 32 * np.floor(g / 8) + 32 * 32 * np.floor(b / 8))
                c = np.array(color_values[w2cM[id_]])

                rs.append(r)
                gs.append(g)
                bs.append(b)
                cs.append(c)

    rs = np.array(rs)
    gs = np.array(gs)
    bs = np.array(bs)
    cs = np.array(cs)

    ax.scatter(rs, gs, bs, c=cs)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    plt.show()

    time1 = time.time()
    ColorNaming.im2c(im, out_type='probability_vector')
    print(time.time() - time1)

    time1 = time.time()
    ColorNaming.im2c(im, out_type='probability_vector')
    print(time.time() - time1)

    time1 = time.time()
    ColorNaming.im2c(im, out_type='probability_vector')
    print(time.time() - time1)
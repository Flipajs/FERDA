__author__ = 'fnaiser'

import numpy as np
import pickle
from utils.video_manager import get_auto_video_manager
from core.region import mser
from core.settings import Settings as S_
from numpy.linalg import norm
from matplotlib.mlab import normpdf
from gui.img_grid.img_grid_dialog import ImgGridDialog
from utils.drawing.points import draw_points_crop, draw_points
from gui.gui_utils import get_image_label
import cv2
from core.animal import colors_
import matplotlib.pyplot as plt
import numpy as np
import math

def dist_score(pos, region, std=10):
        # half the radius

        # std = norm(animal.init_pos_head_ - animal.init_pos_center_) * 0.5

        d = norm(pos - region.centroid())

        max_val = normpdf(0, 0, std)
        s = normpdf(d, 0, std) / max_val

        return s

def axis_length_score(region, mean=10, std=2.5):
    # mean = norm(animal.init_pos_head_ - animal.init_pos_center_)
    # std = mean * 0.25

    major_axis = region.a_

    max_val = normpdf(mean, mean, std)
    s = normpdf(major_axis, mean, std) / max_val

    return s


def margin_score(region):
    #TODO REMOVE CONSTANT!
    return min(region.margin() / 30.0, 1)

def get_np_array(chunks, id, key):
    l = []
    vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
    vid = get_auto_video_manager(vid_path)

    for i in range(len(chunks[0])):

        if key == 'area':
            l.append(chunks[id][i].area())
        if key == 'major_axis':
            l.append(chunks[id][i].a_)
        if key == 'min_intensity':
            l.append(chunks[id][i].min_intensity_)
        if key == 'axis_ratio':
            l.append(chunks[id][i].a_ / chunks[id][i].b_)
        if key == 'avg_intensity':
            im = vid.seek_frame(i)
            p = chunks[id][i].pts()
            val = np.sum(im[p[:,0], p[:,1],:], axis=0) / float(len(p))
            l.append(val)

    return np.array(l)

def get_np_arrays(chunks, key):
    arrays = []
    for i in range(len(chunks)):
        arrays.append(get_np_array(chunks, i, key))

    return arrays

def plot_mean_std(chunks, key, steps=8, chunk_range=None, percentile_val=100):
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    x_step = 0.09

    data = get_np_arrays(chunks, key)

    if not chunk_range:
        data_step = int(math.floor(len(chunks[0]) / steps))
        ch_from = 0
    else:
        data_step = int(math.floor(len(chunk_range) / steps))
        ch_from = chunk_range[0]

    print data_step, ch_from

    for i in range(steps):
        r1_ = range(ch_from + data_step*i, ch_from + data_step*(i+1))

        means = [np.median(a[r1_]) for a in data]
        stds = [np.std(a[r1_]) for a in data]

        plt.errorbar(x+(x_step*i), means, stds, linestyle='None', marker='*')

    plt.xlim(0, 9)
    plt.savefig(working_dir+'/'+key+'.png')
    plt.clf()


def plot_percentile(chunks, key):
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    x2 = x + 0.1

    len_ = len(chunks[0])
    ############# AREAS
    data = get_np_arrays(chunks, key)

    r1_ = range(len_/2)
    means = [np.percentile(a[r1_], 10) for a in data]
    stds = [np.std(a[r1_]) for a in data]

    plt.errorbar(x, means, linestyle='None', marker='^')

    r2_ = range(len_/2, len_)
    means = [np.percentile(a[r1_], 10) for a in data]
    stds = [np.std(a[r2_]) for a in data]

    plt.errorbar(x2, means, linestyle='None', marker='*')

    plt.xlim(0, 9)
    plt.savefig(working_dir+'/'+key+'_10percentile.png')
    plt.clf()

def test_chunks(chunk_1, chunk_2, chunk_A, chunk_B):
    """
    Decides if the probable connection is 1 -> A or 1 -> B and returns strhength of this decision
    :param chunk_1:
    :param chunk_2:
    :param chunk_A:
    :param chunk_B:
    :return:
    """

    mean1 = np.mean(chunk_1)
    std1 = np.std(chunk_1)
    mean_std1 = std1 / math.sqrt(len(chunk_1))

    mean2 = np.mean(chunk_2)
    std2 = np.std(chunk_2)
    mean_std2 = std2 / math.sqrt(len(chunk_2))

    # print mean1-mean2
    # print mean_std1, mean_std2

    std_ = math.sqrt(mean_std1**2 + mean_std2**2)
    n_ = normpdf(0, 0, std_)

    c_ = 1 - (normpdf(mean1-mean2, 0, std_) / n_)
    # print 1 - (normpdf(mean1-mean2, 0, std_) / n_)

    meanA = np.mean(chunk_A)
    meanB = np.mean(chunk_B)

    decision = False
    if abs(mean1 - meanA) < abs(mean1 - meanB):
        # print "choosing A", c_
        decision = True
    # else:
        # print "choosing B", c_

    return decision, c_


if __name__ == '__main__':
    with open('/Users/fnaiser/Documents/chunks/eight_1505_results.arr', 'rb') as f:
        data = pickle.load(f)

    vid_path = '/Users/fnaiser/Documents/chunks/eight.m4v'
    vid = get_auto_video_manager(vid_path)

    S_.cache.mser = True

    working_dir = '/Users/fnaiser/Documents/chunks'

    try:
        with open(working_dir+'/chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
    except:
        chunks = {}
        for k in data[0]:
            chunks[k] = []

        p1 = np.array([data[0][0]['cy'], data[0][0]['cx']])
        p2 = np.array([data[0][0]['hy'], data[0][0]['hx']])
        a_length = norm(p1-p2)

        for i in range(1000):
            print i
            im = vid.move2_next()
            msers = mser.get_all_msers(i, vid_path, working_dir)

            margin_score_ = np.array([margin_score(r) for r in msers])
            for k in data[i]:
                pos = np.array([data[i][k]['cy'], data[i][k]['cx']])

                d_score = np.array([dist_score(pos, r, a_length*0.5) for r in msers])
                axis_score = np.array([axis_length_score(r, a_length, a_length*0.25) for r in msers])

                s = margin_score_ * d_score * axis_score
                # s = d_score * margin_score
                order = np.argsort(-s)
                chunks[k].append(msers[order[0]])

                c = colors_[int(k)]
                c_ = (c[0], c[1], c[2], 0.3)
                im_crop = draw_points(im, msers[order[0]].pts(), color=c_)

            cv2.imshow('i', im_crop)
            if i > 99:
                cv2.waitKey(0)


        with open(working_dir+'/chunks.pkl', 'wb') as f:
            pickle.dump(chunks, f)


    areas = get_np_arrays(chunks, 'major_axis')

    step = 50
    sequences = 8
    ranges = [range(i*step, (i+1)*step) for i in range(sequences)]

    animal_num = len(areas)

    ids = [i for i in range(animal_num)]
    id1 = 0
    id2 = 2

    decisions = []
    certainties = []

    for fixed_id in ids:
        for id in range(fixed_id+1, animal_num):
            for fixed_r_id in range(sequences):
                for r_id in range(fixed_r_id, sequences):
                    ch1 = areas[fixed_id][ranges[fixed_r_id]]
                    ch2 = areas[id][ranges[fixed_r_id]]
                    chA = areas[fixed_id][ranges[r_id]]
                    chB = areas[id][ranges[r_id]]

                    decision, certainty = test_chunks(ch1, ch2, chA, chB)

                    decisions.append(decision)
                    certainties.append(certainty)


    right = np.sum(np.array(decisions))
    print right, len(decisions), right/float(len(decisions))

    r_1 = range(0, 50)
    r_2 = range(50, 100)


    test_chunks(areas[id1][r_1], areas[id2][r_1], areas[id1][r_2], areas[id2][r_2])

    # plot_mean_std(chunks, 'area', chunk_range=range(0, 400))
    # plot_mean_std(chunks, 'major_axis', chunk_range=range(0, 400))
    # plot_mean_std(chunks, 'axis_ratio', chunk_range=range(0, 400))
    # plot_mean_std(chunks, 'min_intensity', chunk_range=range(0, 400))
    # plot_percentile(chunks, 'min_intensity')
    # plot_mean_std(chunks, 'avg_intensity')
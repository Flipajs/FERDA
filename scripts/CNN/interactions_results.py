import matplotlib.pyplot as plt
import h5py
from imageio import imread
import numpy as np
from matplotlib.patches import Ellipse
import os
from os.path import join
import tqdm
import glob
import scripts.CNN.train_interactions as train_interactions
import pandas as pd
from subprocess import call
import glob


def toarray(struct_array):
    types = [x[1] for x in struct_array.dtype.descr]
    all(x == types[0] for x in types)
    return struct_array.view(types[0]).reshape(struct_array.shape + (-1,))


def tostruct(ndarray):
    n = ndarray.shape[1] / len(train_interactions.COLUMNS)
    names = train_interactions.columns(n)
    formats = len(names) * 'f,'
    return np.core.records.fromarrays(ndarray.transpose(), names=', '.join(names), formats=formats)


def save_prediction_img(i, pred, out_filename, num_objects, gt=None, img_data=None):
    fig = plt.figure()
    if img_data is not None:
        im = img_data[i]
    else:
        im = imread(join(DATA_DIR, 'images_test', '%06d.jpg' % i))
    # im = X_test[i, :, :, :]
    # a = plt.subplot(111, aspect='equal')
    # a.imshow(im)
    plt.imshow(im)
    # plt.hold(True)
    plt.axis('off')
    plot_interaction(num_objects, pred[[i]], gt.iloc[i])
    fig.savefig(out_filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.clf()


def plot_interaction(num_objects, pred, gt=None):
    ax = plt.gca()
    colors = ['red', 'blue']
    for i in range(num_objects):
        ax.add_patch(Ellipse(toarray(pred[['%d_x' % i, '%d_y' % i]]).flatten(),
                             pred['%d_major' % i], pred['%d_minor' % i],
                             angle=pred['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
                             label='object %d prediction' % i))
        plt.scatter(pred['%d_x' % i], pred['%d_y' % i], c=colors[i])
        if gt is not None:
            ax.add_patch(Ellipse(toarray(gt[['%d_x' % i, '%d_y' % i]]).flatten(),
                                 gt['%d_major' % i], gt['%d_minor' % i],
                                 angle=gt['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
                                 linestyle='dotted', label='object %d gt' % i))

    plt.legend()


if __name__ == '__main__':
    # ROOT_DIR = '/Users/flipajs/Downloads/double_regions'
    # ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'

    EXPERIMENT_DIR = '/datagrid/personal/smidm1/ferda/interactions/experiments/180104_0142_single/0.857142857143/'
    DATA_DIR = '/datagrid/personal/smidm1/ferda/interactions/1801_1k_36rot_single'
    n_objects = 1

    # EXPERIMENT_DIR = '/home/matej/prace/ferda/experiments/171206_1209_batch/0.344827586207/'
    # DATA_DIR = '/home/matej/prace/ferda/data/interactions/'

    # NAMES = ['0_x', '0_y', '0_major', '0_minor', '0_angle_deg',
    #          ]

    columns = train_interactions.columns(n_objects)
    for i in range(n_objects):
        columns.remove('%d_dx' % i)
        columns.remove('%d_dy' % i)

    hf_img = h5py.File(join(DATA_DIR, 'images.h5'), 'r')
    X_test = hf_img['test']
    y_test_df = pd.read_csv(join(DATA_DIR, 'test.csv'))
    y_test = y_test_df[columns]
    for i in range(n_objects):
        y_test['%d_angle_deg' % i] *= -1  # convert to anti-clockwise
    with h5py.File(join(EXPERIMENT_DIR, 'predictions.h5'), 'r') as hf_pred:
        pred = hf_pred['data'][:]



    # xy, angle, indices = train_interactions.match_pred_to_gt_dxdy(pred, y_test.values, np)
    xy, angle, indices = train_interactions.match_pred_to_gt(pred, y_test.values, np)

    if n_objects == 1:
        xy_errors = xy
        angle_errors = angle
    elif n_objects == 2:
        xy_errors = (xy[indices[:, 0], indices[:, 1]])
        angle_errors = (angle[indices[:, 0], indices[:, 1]])
        assert pred.shape[1] == 10
        swap = indices[:, 0] == 1
        pred[swap, :5], pred[swap, 5:] = pred[swap, 5:], pred[swap, :5]
    else:
        assert False, 'not implemented'

    # estimate major and minor axis length
    pred = tostruct(pred)
    mean_major = y_test[['%d_major' % i for i in range(n_objects)]].stack().mean()
    mean_minor = y_test[['%d_minor' % i for i in range(n_objects)]].stack().mean()
    for i in range(n_objects):
        pred['%d_major' % i] = mean_major
        pred['%d_minor' % i] = mean_minor

    if not os.path.exists(EXPERIMENT_DIR):
        os.mkdir(EXPERIMENT_DIR)
    out_dir = join(EXPERIMENT_DIR, 'visualization')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        for fn in glob.glob(join(EXPERIMENT_DIR, 'visualization', '*.png')):
            os.remove(fn)

    for i in tqdm.tqdm(angle_errors.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(out_dir, 'bad_angle_%03d.png' % i), n_objects, y_test, X_test)
    for i in tqdm.tqdm(xy_errors.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(out_dir, 'bad_xy_%03d.png' % i), n_objects, y_test, X_test)

    for i in tqdm.tqdm(np.random.randint(0, len(pred), 50)):
        save_prediction_img(i, pred, join(out_dir, 'random_%04d.png' % i), n_objects, y_test, X_test)
        # plt.show()

    hf_img.close()

    for part in ['bad_angle' , 'bad_xy', 'random']:
        cmd = 'montage -verbose -tile 5x5 -geometry +5+5 {input_files} {path}/montage_{part}.jpg'.format(
            input_files=' '.join(glob.glob(join(out_dir, part + '*.png'))),
            path=out_dir, part=part)
        print call(cmd.split(' '))

import matplotlib.pyplot as plt
import h5py
from imageio import imread
import numpy as np
from matplotlib.patches import Ellipse, Arc
import os
from os.path import join
import os
import tqdm
import glob
import scripts.CNN.train_interactions as train_interactions
import pandas as pd
from subprocess import call
import glob
import shlex
import fire
import skimage.transform


def save_prediction_img(i, pred, out_filename, num_objects, gt=None, img_data=None):
    fig = plt.figure()
    if img_data is not None:
        # im = skimage.transform.rotate(img_data[i], 90)
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
        ax.add_patch(Ellipse(train_interactions.TrainInteractions.toarray(pred[['%d_x' % i, '%d_y' % i]]).flatten(),
                             pred['%d_major' % i], pred['%d_minor' % i],
                             angle=pred['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
                             label='object %d prediction' % i))
        # ax.add_patch(Arc(toarray(pred[['%d_x' % i, '%d_y' % i]]).flatten(),
        #                  pred['%d_major' % i], pred['%d_minor' % i],
        #                  angle=pred['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
        #                  linewidth=4,
        #                  theta1=-30, theta2=30))
        plt.scatter(pred['%d_x' % i], pred['%d_y' % i], c=colors[i])

        if gt is not None:
            ax.add_patch(Ellipse(train_interactions.TrainInteractions.toarray(gt[['%d_x' % i, '%d_y' % i]]).flatten(),
                                 gt['%d_major' % i], gt['%d_minor' % i],
                                 angle=gt['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
                                 linestyle='dotted', label='object %d gt' % i))
            # ax.add_patch(Arc(toarray(gt[['%d_x' % i, '%d_y' % i]]).flatten(),
            #                  gt['%d_major' % i], gt['%d_minor' % i],
            #                  angle=gt['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
            #                  linewidth=4,
            #                  theta1=-30, theta2=30, linestyle='dotted'))

    plt.legend()


def visualize_results(experiment_dir, data_dir, n_objects=2):
    ti = train_interactions.TrainInteractions(n_objects)
    # for i in range(n_objects):
    #     columns.remove('%d_dx' % i)
    #     columns.remove('%d_dy' % i)
    hf_img = h5py.File(join(data_dir, 'images.h5'), 'r')
    X_test = hf_img['test']
    y_test_df = pd.read_csv(join(data_dir, 'test.csv'))
    y_test = y_test_df[ti.columns()]
    for i in range(n_objects):
        y_test.loc[:, '%d_angle_deg' % i] *= -1  # convert to counter-clockwise
        # y_test.loc[:, '%d_angle_deg' % i] += 90
        # y_test.loc[:, '%d_angle_deg' % i] %= 360
    with h5py.File(join(experiment_dir, 'predictions.h5'), 'r') as hf_pred:
        pred = hf_pred['data'][:]

    # xy, angle, indices = train_interactions.match_pred_to_gt_dxdy(pred, y_test.values, np)
    xy, angle, indices = ti.match_pred_to_gt(pred, y_test.values, np)
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
    pred = ti.tostruct(pred)
    mean_major = y_test[['%d_major' % i for i in range(n_objects)]].stack().mean()
    mean_minor = y_test[['%d_minor' % i for i in range(n_objects)]].stack().mean()
    for i in range(n_objects):
        pred['%d_major' % i] = mean_major
        pred['%d_minor' % i] = mean_minor
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    out_dir = join(experiment_dir, 'visualization')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        for fn in glob.glob(join(experiment_dir, 'visualization', '*.png')):
            os.remove(fn)
    for i in tqdm.tqdm(angle_errors.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(out_dir, 'bad_angle_%03d.png' % i), n_objects, y_test, X_test)
    for i in tqdm.tqdm(xy_errors.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(out_dir, 'bad_xy_%03d.png' % i), n_objects, y_test, X_test)
    for i in tqdm.tqdm(np.random.randint(0, len(pred), 50)):
        save_prediction_img(i, pred, join(out_dir, 'random_%04d.png' % i), n_objects, y_test, X_test)
    hf_img.close()
    results_df = pd.read_csv(join(experiment_dir, 'results.csv'))
    experiment_str = '...' + os.sep + os.sep.join(experiment_dir.strip(os.sep).split(os.sep)[-3:]) + \
                     ' | xy MEA {xy} (px), angle MEA {angle} (deg)'.format(xy=round(float(results_df['xy MAE']), 1),
                                                                           angle=round(float(results_df['angle MAE']),
                                                                                       1))
    for part in ['bad_angle', 'bad_xy', 'random']:
        input_files = glob.glob(join(out_dir, part + '*.png'))
        cmd = 'montage -verbose -tile 5x5 -geometry +5+5 -title {experiment_str} {input_files} {path}/montage_{part}.jpg'.format(
            experiment_str='\"' + part + ' ' + experiment_str + '\"',
            input_files=' '.join(input_files),
            path=out_dir, part=part)
        print call(shlex.split(cmd))
        for fn in input_files:
            os.remove(fn)


if __name__ == '__main__':
    fire.Fire(visualize_results)

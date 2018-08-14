from __future__ import print_function

import glob
import itertools
import shlex
from subprocess import call

import fire
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tqdm
import warnings
import yaml
from imageio import imread
from matplotlib.patches import Ellipse
from os.path import join

import core.interactions.train as train_interactions


def save_prediction_img(out_filename, num_objects, img, pred=None, gt=None, title=None, scale=1.5):
    """
    Save visualization of detected objects and/or ground truth on an image.

    :param out_filename: str; output filename
    :param num_objects: int
    :param img: ndarray
    :param pred: predictions DataFrame
    :param gt: ground truth DataFrame
    :param title: plot title
    :param scale: image scaling, 1.0 is original image size
    """
    if isinstance(img, str):
        img = imread(img)
    dpi = 80
    height, width = img.shape[:2]
    figsize = scale * width / float(dpi), scale * height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, interpolation='nearest')

    plot_interaction(num_objects, pred, gt)
    if title is not None:
        plt.title(title)
    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    fig.savefig(out_filename, transparent=True, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)


def show_prediction(img, num_objects, prediction=None, gt=None, title=None):
    """
    Visualize detected objects and/or ground truth on an image.

    Angles are in degrees counter-clockwise.

    :param img: image
    :param num_objects: number of objects
    :param prediction: predictions DataFrame
    :param gt: ground truth DataFrame
    :param title: plot title
    :return ax
    """
    ax = plt.gca()
    ax.axis('off')
    ax.imshow(img, animated=True)
    plot_interaction(num_objects, prediction, gt)
    if title is not None:
        plt.title(title)
    return ax


def plot_interaction(num_objects, pred=None, gt=None, ax=None):
    """
    Visualize detected objects and/or ground truth.

    Angles are in degrees counter-clockwise.

    :param num_objects: number of objects
    :param pred: predictions DataFrame
    :param gt: ground truth DataFrame
    :param ax: axis to draw on
    """
    if ax is None:
        ax = plt.gca()
    colors = itertools.cycle(['red', 'blue', 'green', 'yellow', 'white'])
    for i, c in zip(range(num_objects), colors):
        if pred is not None:
            ax.add_patch(Ellipse((pred['%d_x' % i], pred['%d_y' % i]),
                                 pred['%d_major' % i], pred['%d_minor' % i],
                                 angle=pred['%d_angle_deg' % i],  facecolor='none', edgecolor=c,
                                 label='object %d prediction' % i, linewidth=2))
            # ax.add_patch(Arc(toarray(pred[['%d_x' % i, '%d_y' % i]]).flatten(),
            #                  pred['%d_major' % i], pred['%d_minor' % i],
            #                  angle=pred['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
            #                  linewidth=4,
            #                  theta1=-30, theta2=30))
            plt.scatter(pred['%d_x' % i], pred['%d_y' % i], c=c)
        if gt is not None:
            ax.add_patch(Ellipse((gt['%d_x' % i], gt['%d_y' % i]),
                                 gt['%d_major' % i], gt['%d_minor' % i],
                                 angle=gt['%d_angle_deg' % i], edgecolor=c, facecolor='none',
                                 linestyle='dotted', label='object %d gt' % i, linewidth=2))
            # ax.add_patch(Arc(toarray(gt[['%d_x' % i, '%d_y' % i]]).flatten(),
            #                  gt['%d_major' % i], gt['%d_minor' % i],
            #                  angle=gt['%d_angle_deg' % i], edgecolor=colors[i], facecolor='none',
            #                  linewidth=4,
            #                  theta1=-30, theta2=30, linestyle='dotted'))

    plt.legend()


def visualize_results(experiment_dir, data_dir, n_objects=None):
    """
    Visualize experimental results. Save montage of random and worst results.

    :param experiment_dir: predictions on the test dataset (predictions.csv, predictions.yaml)
    :param data_dir: test dateset images and ground truth (images.h5, test.csv)
    :param n_objects: number of objects
    """
    hf_img = h5py.File(join(data_dir, 'images.h5'), 'r')
    X_test = hf_img['test']

    if os.path.exists(join(experiment_dir, 'predictions.csv')) and \
            os.path.exists(join(experiment_dir, 'predictions.yaml')):
        pred = pd.read_csv(join(experiment_dir, 'predictions.csv'))
        with open(join(experiment_dir, 'predictions.yaml'), 'r') as fr:
            metadata = yaml.load(fr)
        ti = train_interactions.TrainInteractions(metadata['num_objects'])
    else:
        # now obsolete, only for backwards compatibility
        ti = train_interactions.TrainInteractions(n_objects)
        with h5py.File(join(experiment_dir, 'predictions.h5'), 'r') as hf_pred:
            data = hf_pred['data'][:]
            COLUMNS = ['x', 'y', 'major', 'minor', 'angle_deg']
            if data.shape[1] % 7 == 0:
                COLUMNS += ['dx', 'dy']
            pred_table = train_interactions.ObjectsArray(COLUMNS, n_objects)
            pred = pred_table.array_to_dataframe(data)

    # for i in range(n_objects):
    #     pred['%d_angle_deg' % i] *= -1  # convert to counter-clockwise

    gt_filename = join(data_dir, 'test.csv')
    if os.path.exists(gt_filename):
        y_test_df = pd.read_csv(join(data_dir, 'test.csv'))
        for i in range(n_objects):
            y_test_df.loc[:, '%d_angle_deg' % i] *= -1  # convert to counter-clockwise
            # y_test.loc[:, '%d_angle_deg' % i] += 90
            # y_test.loc[:, '%d_angle_deg' % i] %= 360

        # # input image and gt rotation
        # tregion = TransformableRegion(X_test[0])
        # tregion.rotate(90, np.array(tregion.img.shape[:2]) / 2)
        # for i in range(ti.num_objects):
        #     y_test.loc[:, ['%d_x' % i, '%d_y' % i]] = tregion.get_transformed_coords(
        #         y_test.loc[:, ['%d_x' % i, '%d_y' % i]].values.T).T
        #     y_test.loc[:, '%d_angle_deg' % i] = tregion.get_transformed_angle(y_test.loc[:, '%d_angle_deg' % i])


        # xy, angle, indices = train_interactions.match_pred_to_gt_dxdy(pred, y_test.values, np)
        xy, angle, indices = ti.match_pred_to_gt(ti.array.dataframe_to_array(y_test_df), pred.values, np)
        if n_objects == 1:
            xy_errors = xy
            angle_errors = angle
        elif n_objects == 2:
            xy_errors = (xy[indices[:, 0], indices[:, 1]])
            angle_errors = (angle[indices[:, 0], indices[:, 1]])
            swap = indices[:, 0] == 1
            for prop in ti.array.properties:
                pred.loc[swap, ['0_%s' % prop, '1_%s' % prop]] = pred.loc[swap, ['1_%s' % prop, '0_%s' % prop]].values
        else:
            assert False, 'not implemented'

        # estimate major and minor axis length
        mean_major = y_test_df[['%d_major' % i for i in range(n_objects)]].stack().mean()
        mean_minor = y_test_df[['%d_minor' % i for i in range(n_objects)]].stack().mean()
    else:
        warnings.warn('Ground truth file test.csv not found. No ground truth in visualizations.')
        y_test_df = None
        mean_major = 64
        mean_minor = 15
        angle_errors = None
        xy_errors = None

    for i in range(n_objects):
        pred['%d_major' % i] = mean_major
        pred['%d_minor' % i] = mean_minor

    out_dir = join(experiment_dir, 'visualization')
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    for fn in glob.glob(join(out_dir, '*.png')):
        os.remove(fn)

    visualizations = ['random']
    for i in tqdm.tqdm(np.random.randint(0, len(pred), 50), desc='random predictions'):
        img = X_test[i]
        # tregion.set_img(X_test[i])
        # img = tregion.get_img()
        save_prediction_img(join(out_dir, 'random_%04d.png' % i), n_objects, img, pred.iloc[i],
                            y_test_df.iloc[i] if y_test_df is not None else None)

    if angle_errors is not None:
        for i, idx in enumerate(tqdm.tqdm(angle_errors.flatten().argsort()[::-1][:20], desc='worst angle errors')):
            img = X_test[idx]
            # tregion.set_img(X_test[i])
            # img = tregion.get_img()
            save_prediction_img(join(out_dir, 'bad_angle_%03d.png' % i), n_objects, img, pred.iloc[idx],
                                y_test_df.iloc[idx] if y_test_df is not None else None,
                                title='mean absolute error {:.1f} deg'.format(angle_errors.flatten()[idx]))
        visualizations.append('bad_angle')

    if xy_errors is not None:
        for i, idx in enumerate(tqdm.tqdm(xy_errors.flatten().argsort()[::-1][:20], desc='worst xy errors')):
            img = X_test[idx]
            # tregion.set_img(X_test[i])
            # img = tregion.get_img()
            save_prediction_img(join(out_dir, 'bad_xy_%03d.png' % i), n_objects, img, pred.iloc[idx],
                                y_test_df.iloc[idx] if y_test_df is not None else None,
                                title='mean absolute error {:.1f} px'.format(xy_errors.flatten()[idx]))
        visualizations.append('bad_xy')

    hf_img.close()
    experiment_str = '...' + os.sep + os.sep.join(experiment_dir.strip(os.sep).split(os.sep)[-3:])
    results_filename = join(experiment_dir, 'results.csv')
    if os.path.exists(results_filename):
        results_df = pd.read_csv(results_filename)
        experiment_str += ' | xy MEA {xy} (px), angle MEA {angle} (deg)'.format(xy=round(float(results_df['xy MAE']), 1),
                                                                                angle=round(float(results_df['angle MAE']), 1))
    for part in visualizations:
        input_files = glob.glob(join(out_dir, part + '*.png'))
        cmd = 'montage -verbose -tile 5x5 -geometry +5+5 -title {experiment_str} {input_files} {path}/montage_{part}.jpg'.format(
            experiment_str='\"' + part + ' ' + experiment_str + '\"',
            input_files=' '.join(sorted(input_files)),
            path=out_dir, part=part)
        print(call(shlex.split(cmd)))
        for fn in input_files:
            os.remove(fn)


if __name__ == '__main__':
    fire.Fire(visualize_results)

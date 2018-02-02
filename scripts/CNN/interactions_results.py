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
from core.region.transformableregion import TransformableRegion
import itertools
import warnings
import yaml


def save_prediction_img(out_filename, num_objects, img, pred=None, gt=None, title=None):
    if isinstance(img, str):
        img = imread(img)
    dpi = 80
    height, width, nbands = img.shape
    figsize = width / float(dpi), height / float(dpi)
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
    plt.clf()


def plot_interaction(num_objects, pred=None, gt=None):
    """
    Angles are in degrees counter-clockwise.
    """
    ax = plt.gca()
    colors = itertools.cycle(['red', 'blue', 'green', 'yellow', 'white'])
    for i, c in zip(range(num_objects), colors):
        if pred is not None:
            ax.add_patch(Ellipse((pred['%d_x' % i], pred['%d_y' % i]),
                                 pred['%d_major' % i], pred['%d_minor' % i],
                                 angle=pred['%d_angle_deg' % i], edgecolor=c, facecolor='none',
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
            pred = pd.DataFrame(data, columns=ti.columns(COLUMNS))

    # for i in range(n_objects):
    #     pred['%d_angle_deg' % i] *= -1  # convert to counter-clockwise

    gt_filename = join(data_dir, 'test.csv')
    if os.path.exists(gt_filename):
        y_test_df = pd.read_csv(join(data_dir, 'test.csv'))
        for i in range(n_objects):
            y_test_df.loc[:, '%d_angle_deg' % i] *= -1  # convert to counter-clockwise
            # y_test.loc[:, '%d_angle_deg' % i] += 90
            # y_test.loc[:, '%d_angle_deg' % i] %= 360
        y_test = y_test_df[ti.columns()]

        # # input image and gt rotation
        # tregion = TransformableRegion(X_test[0])
        # tregion.rotate(90, np.array(tregion.img.shape[:2]) / 2)
        # for i in range(ti.num_objects):
        #     y_test.loc[:, ['%d_x' % i, '%d_y' % i]] = tregion.get_transformed_coords(
        #         y_test.loc[:, ['%d_x' % i, '%d_y' % i]].values.T).T
        #     y_test.loc[:, '%d_angle_deg' % i] = tregion.get_transformed_angle(y_test.loc[:, '%d_angle_deg' % i])


        # xy, angle, indices = train_interactions.match_pred_to_gt_dxdy(pred, y_test.values, np)
        xy, angle, indices = ti.match_pred_to_gt(y_test.values, pred.values, np)
        if n_objects == 1:
            xy_errors = xy
            angle_errors = angle
        elif n_objects == 2:
            xy_errors = (xy[indices[:, 0], indices[:, 1]])
            angle_errors = (angle[indices[:, 0], indices[:, 1]])
            swap = indices[:, 0] == 1
            for col in train_interactions.COLUMNS:
                pred.loc[swap, ['0_%s' % col, '1_%s' % col]] = pred.loc[swap, ['1_%s' % col, '0_%s' % col]].values
        else:
            assert False, 'not implemented'

        # estimate major and minor axis length
        mean_major = y_test[['%d_major' % i for i in range(n_objects)]].stack().mean()
        mean_minor = y_test[['%d_minor' % i for i in range(n_objects)]].stack().mean()
    else:
        warnings.warn('Ground truth file test.csv not found. No ground truth in visualizations.')
        y_test = None
        mean_major = 64
        mean_minor = 15
        angle_errors = None
        xy_errors = None

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

    visualizations = ['random']
    for i in tqdm.tqdm(np.random.randint(0, len(pred), 50), desc='random predictions'):
        img = X_test[i]
        # tregion.set_img(X_test[i])
        # img = tregion.get_img()
        save_prediction_img(join(out_dir, 'random_%04d.png' % i), n_objects, img, pred.iloc[i],
                            y_test.iloc[i] if y_test is not None else None)

    if angle_errors is not None:
        for i, idx in enumerate(tqdm.tqdm(angle_errors.flatten().argsort()[::-1][:20], desc='worst angle errors')):
            img = X_test[idx]
            # tregion.set_img(X_test[i])
            # img = tregion.get_img()
            save_prediction_img(join(out_dir, 'bad_angle_%03d.png' % i), n_objects, img, pred.iloc[idx],
                                y_test.iloc[idx] if y_test is not None else None,
                                title='mean absolute error {:.1f} deg'.format(angle_errors.flatten()[idx]))
        visualizations.append('bad_angle')

    if xy_errors is not None:
        for i, idx in enumerate(tqdm.tqdm(xy_errors.flatten().argsort()[::-1][:20], desc='worst xy errors')):
            img = X_test[idx]
            # tregion.set_img(X_test[i])
            # img = tregion.get_img()
            save_prediction_img(join(out_dir, 'bad_xy_%03d.png' % i), n_objects, img, pred.iloc[idx],
                                y_test.iloc[idx] if y_test is not None else None,
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
        print call(shlex.split(cmd))
        for fn in input_files:
            os.remove(fn)


if __name__ == '__main__':
    fire.Fire(visualize_results)

import matplotlib.pyplot as plt
import h5py
from imageio import imread
import numpy as np
from matplotlib.patches import Ellipse
import os
from os.path import join
import tqdm
import glob
import train_interactions


def toarray(struct_array):
    types = [x[1] for x in struct_array.dtype.descr]
    all(x == types[0] for x in types)
    return struct_array.view(types[0]).reshape(struct_array.shape + (-1,))

def tostruct(ndarray):
    NAMES = 'ant1_x, ant1_y, ant1_major, ant1_minor, ant1_angle, ' \
            'ant2_x, ant2_y, ant2_major, ant2_minor, ant2_angle'
    FORMATS = 10 * 'f,'
    return np.core.records.fromarrays(ndarray.transpose(), names=NAMES, formats=FORMATS)


def save_prediction_img(i, pred, out_filename, gt=None):
    fig = plt.figure()
    im = imread(join(DATA_DIR, 'images_test', '%06d.jpg' % i))
    # im = X_test[i, :, :, :]
    # a = plt.subplot(111, aspect='equal')
    # a.imshow(im)
    plt.imshow(im)
    # plt.hold(True)
    plt.axis('off')
    plot_interaction(pred[[i]], gt[[i]])
    fig.savefig(out_filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.clf()


def plot_interaction(pred, gt=None):
    ax = plt.gca()
    # ax.add_patch(Ellipse((pred[i, 0], pred[i, 1]), pred[i, 2], pred[i, 3], angle=pred[i, 4], edgecolor='red', facecolor='none'))
    ax.add_patch(Ellipse(toarray(pred[['ant1_x', 'ant1_y']]).flatten(), pred['ant1_major'], pred['ant1_minor'],
                         angle=-pred['ant1_angle'], edgecolor='red', facecolor='none', label='ant1 prediction'))
    ax.add_patch(Ellipse(toarray(pred[['ant2_x', 'ant2_y']]).flatten(), pred['ant2_major'], pred['ant2_minor'],
                         angle=-pred['ant2_angle'], edgecolor='blue', facecolor='none', label='ant2 prediction'))
    plt.scatter(pred['ant1_x'], pred['ant1_y'], c='red')
    plt.scatter(pred['ant2_x'], pred['ant2_y'], c='blue')
    if gt is not None:
        ax.add_patch(Ellipse(toarray(gt[['ant1_x', 'ant1_y']]).flatten(), gt['ant1_major'], gt['ant1_minor'],
                             angle=-gt['ant1_angle'], edgecolor='red', linestyle='dotted', facecolor='none',
                             label='ant1 gt'))
        # ax.add_patch(Ellipse((pred[i, 5], pred[i, 6]), pred[i, 7], pred[i, 8], angle=pred[i, 9], edgecolor='blue', facecolor='none'))
        ax.add_patch(Ellipse(toarray(gt[['ant2_x', 'ant2_y']]).flatten(), gt['ant2_major'], gt['ant2_minor'],
                             angle=-gt['ant2_angle'], edgecolor='blue', linestyle='dotted', facecolor='none',
                             label='ant2 gt'))

    # ells[i].set_clip_box(a.bbox)
    # ells[i].set_alpha(0.5)
    # a.add_artist(ells[i])
    #
    # print pred[i]
    plt.legend()


if __name__ == '__main__':
    # ROOT_DIR = '/Users/flipajs/Downloads/double_regions'
    # ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
    EXPERIMENT_DIR = '/datagrid/personal/smidm1/ferda/iteractions/experiments/171129_2121'
    DATA_DIR = '/datagrid/personal/smidm1/ferda/iteractions/'



    # with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
    #     X_test = hf['data'][:]
    #
    with h5py.File(join(DATA_DIR, 'results_inter_test.h5'), 'r') as hf:
        y_test = tostruct(hf['data'][:])

    with h5py.File(join(EXPERIMENT_DIR, 'predictions.h5'), 'r') as hf:
        pred = tostruct(hf['data'][:])
        import pandas as pd
        df = pd.DataFrame(pred)
        # df.to_csv(join(DATA_DIR, 'pred.csv'))
        df.describe()

    # delta = 45.0  # degrees
    #
    # angles = np.arange(0, 360 + delta, delta)
    # ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

    pred[['ant1_major', 'ant2_major']] = toarray(y_test[['ant1_major', 'ant2_major']]).mean()
    pred[['ant1_minor', 'ant2_minor']] = toarray(y_test[['ant1_minor', 'ant2_minor']]).mean()

    xy1_error = np.linalg.norm(toarray(y_test[['ant1_x', 'ant1_y']]) - toarray(pred[['ant1_x', 'ant1_y']]), 2, axis=1)
    xy2_error = np.linalg.norm(toarray(y_test[['ant2_x', 'ant2_y']]) - toarray(pred[['ant2_x', 'ant2_y']]), 2, axis=1)
    angle1_error, angle2_error = train_interactions.angle_absolute_error(toarray(y_test), toarray(pred), np)

    if not os.path.exists(EXPERIMENT_DIR):
        os.mkdir(EXPERIMENT_DIR)
    if not os.path.exists(join(EXPERIMENT_DIR, 'test_predictions')):
        os.mkdir(join(EXPERIMENT_DIR, 'test_predictions'))
    else:
        for fn in glob.glob(join(EXPERIMENT_DIR, 'test_predictions', '*.png')):
            os.remove(fn)

    for i in tqdm.tqdm(angle1_error.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(EXPERIMENT_DIR, 'test_predictions', 'bad_angle1_%03d.png' % i), y_test)
    for i in tqdm.tqdm(angle2_error.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(EXPERIMENT_DIR, 'test_predictions', 'bad_angle2_%03d.png' % i), y_test)
    for i in tqdm.tqdm(xy1_error.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(EXPERIMENT_DIR, 'test_predictions', 'bad_xy1_%03d.png' % i), y_test)
    for i in tqdm.tqdm(xy2_error.flatten().argsort()[::-1][:20]):
        save_prediction_img(i, pred, join(EXPERIMENT_DIR, 'test_predictions', 'bad_xy2_%03d.png' % i), y_test)


    for i in tqdm.tqdm(np.random.randint(0, len(pred), 50)):
        save_prediction_img(i, pred, join(EXPERIMENT_DIR, 'test_predictions', 'random_%04d.png' % i), y_test)
        # plt.show()

    # create montages in fish shell:
    # $ for s in random bad_angle1 bad_angle2 bad_xy1 bad_xy2; montage -verbose -tile 5x5 -geometry +5+5 $s*.png montage_$s.jpg; end

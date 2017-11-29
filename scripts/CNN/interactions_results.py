import matplotlib.pyplot as plt
import h5py
from imageio import imread
import numpy as np
from matplotlib.patches import Ellipse
import os
import tqdm
import glob


# ROOT_DIR = '/Users/flipajs/Downloads/double_regions'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
ROOT_DIR = '/datagrid/personal/smidm1/ferda/iteractions/'
DATA_DIR = ROOT_DIR

NAMES = 'ant1_x, ant1_y, ant1_major, ant1_minor, ant1_angle, ' \
        'ant2_x, ant2_y, ant2_major, ant2_minor, ant2_angle'
FORMATS = 10 * 'f,'

# with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
#     X_test = hf['data'][:]
#
with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
    y_test = np.core.records.fromarrays(hf['data'][:].transpose(), names=NAMES, formats=FORMATS)

with h5py.File(DATA_DIR + '/predictions.h5', 'r') as hf:
    pred = np.core.records.fromarrays(hf['data'][:].transpose(), names=NAMES, formats=FORMATS)
    import pandas as pd
    df = pd.DataFrame(pred)
    # df.to_csv(os.path.join(DATA_DIR, 'pred.csv'))
    df.describe()

# delta = 45.0  # degrees
#
# angles = np.arange(0, 360 + delta, delta)
# ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

mean_major = y_test[['ant1_major', 'ant2_major']].view(y_test.dtype[0]).mean()
mean_minor = y_test[['ant1_minor', 'ant2_minor']].view(y_test.dtype[0]).mean()

for fn in glob.glob(os.path.join(ROOT_DIR, 'test_predictions', '*.png')):
    os.remove(fn)

for i in tqdm.tqdm(np.random.randint(0, len(pred), 50)):
    fig = plt.figure()
    im = imread(os.path.join(DATA_DIR, 'images_test', '%06d.jpg' % i))

    # im = X_test[i, :, :, :]
    # a = plt.subplot(111, aspect='equal')
    # a.imshow(im)
    plt.imshow(im)
    # plt.hold(True)

    ax = plt.gca()
    plt.axis('off')
    # ax.add_patch(Ellipse((pred[i, 0], pred[i, 1]), pred[i, 2], pred[i, 3], angle=pred[i, 4], edgecolor='red', facecolor='none'))
    ax.add_patch(Ellipse(pred[['ant1_x', 'ant1_y']][i], mean_major, mean_minor,
                         angle=-pred[i]['ant1_angle'], edgecolor='red', facecolor='none', label='ant1 prediction'))
    ax.add_patch(Ellipse(y_test[['ant1_x', 'ant1_y']][i], y_test[i]['ant1_major'], y_test[i]['ant1_minor'],
                         angle=-y_test[i]['ant1_angle'], edgecolor='magenta', facecolor='none', label='ant1 gt'))
    # ax.add_patch(Ellipse((pred[i, 5], pred[i, 6]), pred[i, 7], pred[i, 8], angle=pred[i, 9], edgecolor='blue', facecolor='none'))
    ax.add_patch(Ellipse(pred[['ant2_x', 'ant2_y']][i], mean_major, mean_minor,
                         angle=-pred[i]['ant2_angle'], edgecolor='blue', facecolor='none', label='ant2 prediction'))
    ax.add_patch(Ellipse(y_test[['ant2_x', 'ant2_y']][i], y_test[i]['ant2_major'], y_test[i]['ant2_minor'],
                         angle=-y_test[i]['ant2_angle'], edgecolor='cyan', facecolor='none', label='ant2 gt'))
    
    # ells[i].set_clip_box(a.bbox)
    # ells[i].set_alpha(0.5)
    # a.add_artist(ells[i])
    #
    # print pred[i]
    plt.scatter(pred[i]['ant1_x'], pred[i]['ant1_y'], c='red')
    plt.scatter(pred[i]['ant2_x'], pred[i]['ant2_y'], c='blue')
    plt.legend()

    fig.savefig(os.path.join(ROOT_DIR, 'test_predictions', '%03d.png' % i), transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    plt.clf()
    # plt.show()
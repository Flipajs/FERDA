import matplotlib.pyplot as plt
import h5py
import numpy as np
from imageio import imread
import numpy as np
from matplotlib.patches import Ellipse

ROOT_DIR = '/Users/flipajs/Downloads/double_regions'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
DATA_DIR = ROOT_DIR

# with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
#     X_test = hf['data'][:]
#
with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
    y_test = hf['data'][:]

with h5py.File(DATA_DIR + '/predictions.h5', 'r') as hf:
    pred = hf['data'][:]


delta = 45.0  # degrees

angles = np.arange(0, 360 + delta, delta)
ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

for i in range(10, 100):
    s = str(i)
    while len(s) < 6:
        s = "0"+s

    im = imread(DATA_DIR+'/images_test/'+s+'.jpg')

    # im = X_test[i, :, :, :]
    # a = plt.subplot(111, aspect='equal')
    # a.imshow(im)
    plt.imshow(im)
    plt.hold(True)

    ax = plt.gca()
    # ax.add_patch(Ellipse((pred[i, 0], pred[i, 1]), pred[i, 2], pred[i, 3], angle=pred[i, 4], edgecolor='red', facecolor='none'))
    ax.add_patch(Ellipse((pred[i, 0], pred[i, 1]), 40, 20, angle=-pred[i, 2], edgecolor='red', facecolor='none'))
    ax.add_patch(Ellipse((y_test[i, 0], y_test[i, 1]), y_test[i, 2], y_test[i, 3], angle=-y_test[i, 4], edgecolor='magenta', facecolor='none'))
    # ax.add_patch(Ellipse((pred[i, 5], pred[i, 6]), pred[i, 7], pred[i, 8], angle=pred[i, 9], edgecolor='blue', facecolor='none'))
    ax.add_patch(Ellipse((pred[i, 3], pred[i, 4]), 40, 20, angle=-pred[i, 5], edgecolor='blue', facecolor='none'))
    ax.add_patch(
        Ellipse((y_test[i, 5], y_test[i, 6]), y_test[i, 7], y_test[i, 8], angle=-y_test[i, 9], edgecolor='cyan',
                facecolor='none'))
    # ells[i].set_clip_box(a.bbox)
    # ells[i].set_alpha(0.5)
    # a.add_artist(ells[i])
    #
    print pred[i, :]
    plt.scatter(pred[i, 0], pred[i, 1])
    plt.scatter(pred[i, 3], pred[i, 4])

    plt.show()
import matplotlib.pyplot as plt
import h5py
import numpy as np
from imageio import imread
import numpy as np
from matplotlib.patches import Ellipse
import math

def get_head_pt(theta, major_axis, x, y):
    # deg to rad
    theta = np.deg2rad(theta + 90)

    a = major_axis/2.0
    p_ = np.array([a* math.sin(theta), a * math.cos(theta)])
    return np.ceil(np.array([x, y]) + p_)

def rotate_pts(ox, oy, th_deg, x, y):
    th = np.deg2rad(th_deg)
    qx = ox + math.cos(th) * (x - ox) - math.sin(th) * (y - oy)
    qy = oy + math.sin(th) * (x - ox) + math.cos(th) * (y - oy)

    return qx, qy


ROOT_DIR = '/Users/flipajs/Downloads/double_regions'
# ROOT_DIR = '/Users/flipajs/Documents/wd/FERDA/cnn_exp'
DATA_DIR = ROOT_DIR

# with h5py.File(DATA_DIR + '/imgs_inter_test.h5', 'r') as hf:
#     X_test = hf['data'][:]
#
with h5py.File(DATA_DIR + '/results_inter_test.h5', 'r') as hf:
    y_test = hf['data'][:]

with h5py.File(DATA_DIR + '/results_inter_train.h5', 'r') as hf:
    y_train = hf['data'][:]

with h5py.File(DATA_DIR + '/predictions_e0.h5', 'r') as hf:
    pred = hf['data'][:]


delta = 45.0  # degrees

angles = np.arange(0, 360 + delta, delta)
ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

# plt.hist(pred[:, 4])
# plt.hist(pred[:, 9])
# plt.figure()
# plt.hist(y_train[:, 4])
# plt.figure()
# plt.hist(y_train[:, 9])
# plt.show()
f, axs = plt.subplots(3, 5, tight_layout=True, squeeze=True)
axs = axs.flatten()

offset = 200
pred = pred[offset:, :]
y_test = y_test[offset:, :]
for i in range(len(axs)):
    s = str(i+offset)
    while len(s) < 6:
        s = "0"+s

    im = imread(DATA_DIR+'/images_test/'+s+'.jpg')

    # x_batch = []
    # y_batch = []
    # X = im
    # y = y_test[i, :]
    #
    # BATCH_SIZE = 8
    # from scipy.ndimage.interpolation import rotate
    #
    # thetas = np.linspace(50, 360, BATCH_SIZE, endpoint=False)
    # for i in range(BATCH_SIZE):
    #     th = thetas[i]
    #     new_y = np.copy(y)
    #     # print new_y
    #     X_new = rotate(X, angle=th, reshape=False, mode='nearest')
    #     new_y[4] = (new_y[4] + th) % 360
    #     new_y[9] = (new_y[9] + th) % 360
    #
    #     oy, ox = X.shape[0] / 2.0, X.shape[1] / 2.0
    #     x1, y1 = new_y[0], new_y[1]
    #     x2, y2 = new_y[5], new_y[6]
    #
    #     new_y[0], new_y[1] = rotate_pts(ox, oy, -th, x1, y1)
    #     new_y[5], new_y[6] = rotate_pts(ox, oy, -th, x2, y2)
    #
    #     # print new_y
    #     y_batch.append(new_y)
    #
    #     plt.imshow(X_new)
    #     ax = plt.gca()
    #     ax.add_patch(Ellipse((new_y[0], new_y[1]), new_y[2], new_y[3], angle=-new_y[4], edgecolor='red', facecolor='none'))
    #     ax.add_patch(Ellipse((new_y[5], new_y[6]), new_y[7], new_y[8], angle=-new_y[9], edgecolor='blue', facecolor='none'))
    #
    #     plt.show()
    # im = X_test[i, :, :, :]
    # a = plt.subplot(111, aspect='equal')
    # a.imshow(im)
    axs[i].imshow(im)
    plt.hold(True)

    # ax = plt.gca()
    # ax = axs[i]
    x1 = pred[i, 0]
    y1 = pred[i, 1]
    x2 = pred[i, 5]
    y2 = pred[i, 6]
    major1 = pred[i, 2]
    major2=  pred[i, 7]
    minor1 = pred[i, 3]
    minor2 = pred[i, 8]
    theta1 = pred[i, 4] % 360
    theta2 = pred[i, 9] % 360

    # x2 = y_test[i, 5]
    # y2 = y_test[i, 6]
    major1 = y_test[i, 2]
    major2 = y_test[i, 7]
    minor1 = y_test[i, 3]
    minor2 = y_test[i, 8]
    # theta2 = y_test[i, 9]

    axs[i].add_patch(Ellipse((x1, y1), major1, minor1, angle=-theta1, edgecolor='red', facecolor='none'))
    axs[i].add_patch(Ellipse((y_test[i, 0], y_test[i, 1]), y_test[i, 2], y_test[i, 3], angle=-y_test[i, 4], edgecolor='red', lw=2, linestyle='dashed', facecolor='none'))
    axs[i].add_patch(Ellipse((x2, y2), major2, minor2, angle=-theta2, edgecolor='blue', facecolor='none'))
    axs[i].add_patch(Ellipse((y_test[i, 5], y_test[i, 6]), y_test[i, 7], y_test[i, 8], angle=-y_test[i, 9],  edgecolor='blue', lw=2, linestyle='dashed', facecolor='none'))
    # ells[i].set_clip_box(a.bbox)
    # ells[i].set_alpha(0.5)
    # a.add_artist(ells[i])
    #

    # print pred[i, :]
    # print pred[i, 2], pred[i, 5]
    # axs[i].scatter(x1, y1, c='r', marker='x')
    # axs[i].scatter(x2, y2, c='b', marker='x')
    #
    # head1 = get_head_pt(theta1, major1, x1, y1)
    # axs[i].scatter(head1[0], head1[1], c='r', marker='o')
    # head2 = get_head_pt(theta2, major2, x2, y2)
    # axs[i].scatter(head2[0], head2[1], c='b', marker='o')
    #
    #
    # # GT
    # x1 = y_test[i, 0]
    # y1 = y_test[i, 1]
    # x2 = y_test[i, 5]
    # y2 = y_test[i, 6]
    # major1 = y_test[i, 2]
    # major2 = y_test[i, 7]
    # theta1 = y_test[i, 4]
    # theta2 = y_test[i, 9]
    #
    # head1 = get_head_pt(theta1, major1, x1, y1)
    # axs[i].scatter(head1[0], head1[1], c='r', marker='s')
    # head2 = get_head_pt(theta2, major2, x2, y2)
    # axs[i].scatter(head2[0], head2[1], c='b', marker='s')

    axs[i].axis('off')

plt.show()
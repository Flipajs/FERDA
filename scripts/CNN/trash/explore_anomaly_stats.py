import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from utils.img import get_safe_selection
import cv2
from core.project.project import Project

wd = '/Volumes/Seagate Expansion Drive/HH1_POST'


p = Project()
p.load(wd)
MARGIN = 1.25
major_axis = 36

offset = major_axis * MARGIN


WD = '/Volumes/Seagate Expansion Drive/'

with h5py.File(WD + 'labels_multi_train.h5', 'r') as hf:
    y= hf['data'][:]

with h5py.File(WD + 'penultimate_layer.h5', 'r') as hf:
    X = hf['data'][:]

print X.shape

ids = np.unique(y)
avg_vects = []
med_vects = []
std_vects = []

np.set_printoptions(precision=2)

markers = ['^', '*', '.', '<', '+', '>', 'o']
for i, id_ in enumerate(ids):
    print id_
    avg_vects.append(np.mean(X[y == id_, :], axis=0))
    med_vects.append(np.median(X[y == id_, :], axis=0))
    std_vects.append(np.std(X[y == id_, :], axis=0))

    print avg_vects[-1]
    print std_vects[-1]

    xx = np.array(range(len(avg_vects[-1]))) + i / float(len(ids) * 2)
    yy = avg_vects[-1]
    e = std_vects[-1]

    plt.errorbar(xx, yy, e, linestyle='None', marker=markers[i])

    # plt.plot(avg_vects[-1])
    plt.hold(True)

avg_vects = np.array(avg_vects)
C = np.zeros((len(ids), len(ids)))
for i in ids:
    for j in ids:
        C[i, j] = np.linalg.norm(avg_vects[i] - avg_vects[j])

print C

plt.legend([str(i) for i in ids])
plt.grid()

plt.matshow(C)
plt.colorbar()

percentiles1 = []
percentiles5 = []

f, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True)
axs = axs.flatten()
for id_ in ids:
    print "*** ", id_
    d = np.linalg.norm(X[y == id_, :] - avg_vects[id_], axis=1)
    axs[id_].hist(d)
    axs[id_].set_title(str(id_))
    axs[id_].grid()
    percentiles1.append(-np.percentile(-d, 1))
    percentiles5.append(-np.percentile(-d, 5))
    print d.min(), d.max(), np.mean(d), np.std(d), -np.percentile(-d, 1)
    sorted_ids = np.argsort(-d)

    for i in range(10):
        ii = sorted_ids[i]

        print y[y == id_][ii], d[ii]
    print

plt.suptitle('argmax X == i distance to MEAN vector distribution')


f, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True)
axs = axs.flatten()
for id_ in ids:
    print "*** ", id_
    d = np.linalg.norm(X[y == id_, :] - med_vects[id_], axis=1)
    axs[id_].hist(d)
    axs[id_].set_title(str(id_))
    axs[id_].grid()
    percentiles1.append(-np.percentile(-d, 1))
    percentiles5.append(-np.percentile(-d, 5))
    print d.min(), d.max(), np.mean(d), np.std(d), -np.percentile(-d, 1)
    sorted_ids = np.argsort(-d)

    for i in range(10):
        ii = sorted_ids[i]

        print y[y == id_][ii], d[ii]
    print

plt.suptitle('argmax X == i distance to MEDIAN vector distribution')
# plt.show()

# f, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True)
# axs = axs.flatten()
# for id_ in [ids[0]]:
#     for id2 in ids:
#         print "*** ", id_
#         d = np.linalg.norm(X[y == id_, :] - med_vects[id2], axis=1)
#         axs[id2].hist(d)
#         axs[id2].set_title(str(id2))
#         axs[id2].grid()
#         percentiles1.append(-np.percentile(-d, 1))
#         percentiles5.append(-np.percentile(-d, 5))
#         print d.min(), d.max(), np.mean(d), np.std(d), -np.percentile(-d, 1)
#         sorted_ids = np.argsort(-d)
#
#         for i in range(10):
#             ii = sorted_ids[i]
#
#             print y[y == id_][ii], d[ii]
#         print
#
# plt.suptitle('argmax X == 0, compared to all median_vectors')

# plt.show()

with open(WD+'/distance_data.pkl', 'wb') as f:
    pickle.dump([avg_vects, percentiles1, percentiles5], f)


with open(WD+'HH1_POST/softmax_results_map.pkl', 'r') as f:
    results = pickle.load(f)

with open(WD+'HH1_POST/softmax_dist_map.pkl', 'r') as f:
    dists = pickle.load(f)

num_anomalies = [0] * len(ids)
distances = []
for i in range(len(ids)):
    distances.append([])

for id, val in tqdm(dists.iteritems()):
    k = np.argmax(results[id])
    d = np.linalg.norm(avg_vects[k] - val)

    distances[k].append(d)

    if d > percentiles1[k]:
        # print id, k, d
        num_anomalies[k] += 1

        r = p.rm[id]

        frame = r.frame()

        t_id = None
        for t in p.chm.chunks_in_frame(frame):
            if id in t.rid_gen(p.gm):
                t_id = t.id()
            break

        if t_id is None:
            continue

        t = p.chm[t_id]
        if not t.is_single():
            continue

        img = p.img_manager.get_whole_img(r.frame())

        y, x = r.centroid()
        crop = get_safe_selection(img, y - offset, x - offset, 2 * offset, 2 * offset)


        rs = "{}".format(int((d/percentiles1[k])*100))


        ds = "{}".format(int(d))
        while len(ds) < 6:
            ds = "0"+ds

        cv2.putText(crop, "{:.2f}".format((d/percentiles1[k])), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65,
                    (255, 255, 255), 1, cv2.cv.CV_AA)

        cv2.putText(crop, str(k), (3, 20), cv2.FONT_HERSHEY_PLAIN, 0.65,
                    (255, 255, 255), 1, cv2.cv.CV_AA)

        # print "saving ", id
        cv2.imwrite(
            '/Users/flipajs/Documents/dev/ferda/scripts/out9/' + rs + '_' + ds + '_'+ str(k) + '_' + str(id) +'.jpg',
            crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95])


print num_anomalies

f, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True)
axs = axs.flatten()
for id_ in ids:
    axs[id_].hist(distances[id_])
    axs[id_].set_title(str(id_))
    axs[id_].grid()

plt.show()
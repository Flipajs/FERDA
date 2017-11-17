import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
WD = '/Volumes/Seagate Expansion Drive/'

with h5py.File(WD + 'labels_multi_train.h5', 'r') as hf:
    y= hf['data'][:]

with h5py.File(WD + 'penultimate_layer.h5', 'r') as hf:
    X = hf['data'][:]

print X.shape

ids = np.unique(y)
avg_vects = []
std_vects = []

np.set_printoptions(precision=2)

markers = ['^', '*', '.', '<', '+', '>', 'o']
for i, id_ in enumerate(ids):
    print id_
    avg_vects.append(np.mean(X[y == id_, :], axis=0))
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

with open(WD+'/distance_data.pkl', 'wb') as f:
    pickle.dump([avg_vects, percentiles1, percentiles5], f)


with open(WD+'HH1_POST/softmax_results_map.pkl', 'r') as f:
    results = pickle.load(f)

with open(WD+'HH1_POST/softmax_dist_map.pkl', 'r') as f:
    dists = pickle.load(f)


ids = []
vals = []
for id, val in tqdm(dists.iteritems()):
    k = np.argmax(results[id])
    d = np.linalg.norm(avg_vects[k] - val)

    if d > percentiles1[k]:
        print k, id

ids = np.array(ids)
vals = np.array(vals)

plt.show()
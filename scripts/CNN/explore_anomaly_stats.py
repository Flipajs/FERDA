import h5py
import numpy as np
import matplotlib.pyplot as plt

WD = '/Volumes/Seagate Expansion Drive/'

with h5py.File(WD + 'labels_multi_train.h5', 'r') as hf:
    y= hf['data'][:]

with h5py.File(WD + 'penultimate_layer.h5', 'r') as hf:
    X = hf['data'][:]

print X.shape

ids = np.unique(y)[:-1]
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
plt.show()
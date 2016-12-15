import cPickle as pickle
import numpy as np
from core.project.project import Project


paths = ['/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/epsilons.pkl',
         '/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground/temp/epsilons.pkl',
         '/Users/flipajs/Documents/wd/FERDA/Sowbug3/temp/epsilons.pkl',
         '/Users/flipajs/Documents/wd/FERDA/Camera3/temp/epsilons.pkl']

# paths = [paths[-1]]

for path in paths:
    print path.split('/')[-3]
    p = Project()
    p.load_hybrid('/'.join(path.split('/')[:-2]), state='isolation_score')

    with open(path) as f:
        (epsilons, edges, variant, AA, BB) = pickle.load(f)


    epsilons = np.array(epsilons)
    edges = np.array(edges)
    variant = np.array(variant)
    AA = np.array(AA)
    BB = np.array(BB)

    ids_ = variant==1


    a = AA[ids_]
    b = BB[ids_]
    eps_ = epsilons[ids_]

    ids2_ = np.logical_and(epsilons < eps_.max(), variant == 0)

    print "#undecided: {}, eps: {:.3f}, maxA: {:.3f}".format(np.sum(ids2_), eps_.max(), AA[ids2_].max())

    for i in range(10):
        id_ = np.argmax(eps_)
        e = edges[ids_][id_]
        eps_[id_] = 0
        print "observe:", e, p.gm.region(e[0]).frame_, p.gm.region(e[1]).frame_

    print ""
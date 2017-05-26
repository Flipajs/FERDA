import cPickle as pickle
import numpy as np

def detect_interactions(m):
    num_ids = len(m[0])

    prev_id = None
    for animal_id_ in range(num_ids):
        num = 0
        num_interactions = [0 for i in range(7)]
        id_swap = []
        print "ID_ ", animal_id_
        for frame, ids_ in enumerate(m):
            tid_ = ids_[animal_id_]

            if frame > 0:
                if prev_id != tid_:
                    # print frame, tid_
                    num += 1
                    num_interactions[ids_.count(tid_)] += 1

                    if ids_.count(prev_id) > 0:
                        id_swap.append((frame, prev_id, tid_))

            prev_id = tid_

        print num_interactions[1:], "SUM:", num
        print id_swap
        print

if __name__ == '__main__':
    with open('/Users/flipajs/Desktop/temp/match.pkl') as f:
        match = pickle.load(f)

    m = []
    frames = sorted([f for f in match.keys()])
    for f in frames:
        m.append(match[f])

    detect_interactions(m)
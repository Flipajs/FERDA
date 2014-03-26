__author__ = 'flipajs'

import my_utils
import numpy as np

def collision_detection(ants, history=0):
    thresh1 = 70
    thresh2 = 20

    collisions = []
    for i in range(len(ants)):
        ants[i].state.collision_predicted = False

    for i in range(len(ants)):
        a1 = ants[i].state
        a1.collisions = []
        if history > 0:
            a1 = ants[i].history[history-1]
        for j in range(i+1, len(ants)):
            a2 = ants[j].state
            if history > 0:
                a2 = ants[j].history[history-1]

            dist = my_utils.e_distance(a1.position, a2.position)
            if dist < thresh1:
                dists = [0]*9
                dists[0] = my_utils.e_distance(a1.head, a2.head)
                dists[1] = my_utils.e_distance(a1.head, a2.position)
                dists[2] = my_utils.e_distance(a1.head, a2.back)
                dists[3] = my_utils.e_distance(a1.position, a2.head)
                dists[4] = dist
                dists[5] = my_utils.e_distance(a1.position, a2.back)
                dists[6] = my_utils.e_distance(a1.back, a2.head)
                dists[7] = my_utils.e_distance(a1.back, a2.position)
                dists[8] = my_utils.e_distance(a1.back, a2.back)

                min_i = np.argmin(np.array(dists))
                if dists[min_i] < thresh2:
                    ants[i].state.collision_predicted = True
                    ants[i].state.collisions.append((j, dists[min_i], min_i))
                    ants[j].state.collision_predicted = True
                    ants[j].state.collisions.append((i, dists[min_i], min_i))

                    p1 = a1.head
                    if min_i % 3 == 1:
                        p1 = a1.position
                    elif min_i % 3 == 2:
                        p1 = a1.back

                    if min_i < 3:
                        p2 = a2.head
                    elif min_i < 6:
                        p2 = a2.position
                    elif min_i < 9:
                        p2 = a2.back

                    coll_middle = p1+p2
                    coll_middle.x /= 2
                    coll_middle.y /= 2

                    collisions.append((i, j, dists[min_i], min_i, coll_middle))

    return collisions
from __future__ import print_function
__author__ = 'filip@naiser.cz'

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np



#dir = os.path.expanduser('~/dump/eight')
#afile = open(dir+"/eight_ferda_stable_split3_certainty.arr", "rb")

dir = os.path.expanduser('~/dump/eight')
afile = open(dir+"/eight_ferda_stable_split_dmaps2_certainty.arr", "rb")
certainty = pickle.load(afile)
afile.close()

ant_num = len(certainty[0])
ants_c = np.zeros((ant_num, len(certainty)))
for key in certainty:
    for i in range(len(certainty[key])):
        ants_c[i][key] = certainty[key][i] / 2

print(ants_c[0])

for i in range(ant_num):
    plt.plot(ants_c[i] + i)
    plt.plot([0, len(certainty)], [i, i], color='k', linestyle='solid', linewidth=3)


noplast_swaps = [1056, 1220, 1449, 1737, 2059, 1743, 1266, 1748, 1047, 1050]
eight_swaps = [673, 691]

swaps = eight_swaps
for swap in swaps:
    plt.plot([swap, swap], [0, ant_num], color='gray', linestyle='dashed', linewidth=3)

noplast_fps = [2058, 1043, 1044, 1045, 1046, 152, 1049, 155, 1054, 1055, 1058,1966, 175, 176, 177, 178, 179, 180, 950, 183, 184, 1466, 1467, 961, 1219, 182, 1742, 1746, 1747, 2008, 2009, 616, 617, 618, 1264, 1265]
eight_fps = [450, 803, 1284, 806, 807, 1180, 680, 692, 1174, 475, 1179, 1437, 767]

#fps = noplast_fps
#for fp in fps:
#    plt.plot([fp, fp], [0, ant_num], color='gray', linestyle='dotted', linewidth=3)
#plt.plot(600, 4.3, mfc='none')
#

plt.xlim([0, len(certainty)])
plt.ylabel('ant id')
plt.xlabel('#frame')
plt.show()
plt.waitforbuttonpress(0)
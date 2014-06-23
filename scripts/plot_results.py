__author__ = 'flipajs'
import matplotlib.pyplot as plt
import numpy as np
from trajectories_data import eight_idtracker, eight_ctrax, eight_ktrack, eight_gt, messor_gt, messor_idtracker, noplast_ctrax, noplast_gt, noplast_ktrack
import pickle
from clearmetrics import clearmetrics

mismatch_size = 25
fp_size = 10
frames = 1502
#
#

def get_frames(dict):
    frames = []
    for key in dict:
        frames.append(key)

    return frames




clear_precision = 7

#f = open('trajectories_data/eight_ferda_stable_split_dmaps2_xy.arr', 'rb')
#f_data = pickle.load(f)
#f.close()
#
#f_metrics = clearmetrics.ClearMetrics(eight_gt.data, f_data, clear_precision)
#f_metrics.match_sequence()
#evaluation = [f_metrics.get_mota(), f_metrics.get_motp(), f_metrics.get_fn_count(), f_metrics.get_fp_count(), f_metrics.get_mismatches_count(), f_metrics.get_object_count(), f_metrics.get_matches_count()]
#print evaluation
#
#i_metrics = clearmetrics.ClearMetrics(eight_gt.data, eight_idtracker.data, clear_precision)
#i_metrics.match_sequence()
#evaluation = [i_metrics.get_mota(), i_metrics.get_motp(), i_metrics.get_fn_count(), i_metrics.get_fp_count(), i_metrics.get_mismatches_count(), i_metrics.get_object_count(), i_metrics.get_matches_count()]
#print evaluation
#
#k_metrics = clearmetrics.ClearMetrics(eight_gt.data, eight_ktrack.data, clear_precision)
#k_metrics.match_sequence()
#evaluation = [k_metrics.get_mota(), k_metrics.get_motp(), k_metrics.get_fn_count(), k_metrics.get_fp_count(), k_metrics.get_mismatches_count(), k_metrics.get_object_count(), k_metrics.get_matches_count()]
#print evaluation
#
#c_metrics = clearmetrics.ClearMetrics(eight_gt.data, eight_ctrax.data, clear_precision)
#c_metrics.match_sequence()
#evaluation = [c_metrics.get_mota(), c_metrics.get_motp(), c_metrics.get_fn_count(), c_metrics.get_fp_count(), c_metrics.get_mismatches_count(), c_metrics.get_object_count(), c_metrics.get_matches_count()]
#print evaluation
#
#
#plt.figure()
#plt.plot([2.75, 3.25], [813, 813], color='red', linestyle='dashed', linewidth=3)
#mc = get_frames(c_metrics.mismatches_in_frames)
#fp = get_frames(c_metrics.fp_in_frames)
#fn = get_frames(c_metrics.fn_in_frames)
#plt.plot(np.ones(len(mc))*1, mc, 'bo', label='ctrax mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(fp))*1, fp, 'bx', label='ctrax fp', markersize=fp_size)
#plt.plot(np.ones(len(fn))*1, fn, 'b+', label='ctrax fn', markersize=fp_size)
#
#mc = get_frames(k_metrics.mismatches_in_frames)
#fp = get_frames(k_metrics.fp_in_frames)
#fn = get_frames(k_metrics.fn_in_frames)
#plt.plot(np.ones(len(mc))*3, mc, 'go', label='K-Track mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(fp))*3, fp, 'gx', label='K-Track fp', markersize=fp_size)
#plt.plot(np.ones(len(fn))*3, fn, 'g+', label='K-Track fn', markersize=fp_size)
#
#mc = get_frames(i_metrics.mismatches_in_frames)
#fp = get_frames(i_metrics.fp_in_frames)
#fn = get_frames(i_metrics.fn_in_frames)
#plt.plot(np.ones(len(mc))*2, mc, 'ro', label='idTracker mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(fp))*2, fp, 'rx', label='idtracker fp', markersize=fp_size)
#plt.plot(np.ones(len(fn))*2, fn, 'r+', label='idtracker fn', markersize=fp_size)
#
#mc = get_frames(f_metrics.mismatches_in_frames)
#fp = get_frames(f_metrics.fp_in_frames)
#fn = get_frames(f_metrics.fn_in_frames)
#
#plt.plot(np.ones(len(mc))*4, mc, 'mo', label='Ferda mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(fp))*4, fp, 'mx', label='Ferda fp', markersize=fp_size)
#plt.plot(np.ones(len(fn))*4, fn, 'm+', label='Ferda fn', markersize=fp_size)
#
#
##plt.axis([0, frames, 0, 4.5])
#plt.axis([0.5, 4.5, 0, frames])
##plt.legend(loc='bottom center', shadow=True, fontsize=17, numpoints=1, fancybox=True)
#plt.grid(True)
#plt.ylabel('frame number')
#plt.xticks([1, 2, 3, 4],['Ctrax', 'idTracker', 'K-Track', 'Ferda'])
#plt.subplots_adjust(hspace=0, wspace=0, left=0.08, bottom=0.02, right=0.999, top=0.999)
#plt.show()
#







#
frames = 2262
f = open('trajectories_data/noplast_ferda_stable_split_dmaps2_xy.arr', 'rb')
f_data = pickle.load(f)
f.close()

f_metrics = clearmetrics.ClearMetrics(noplast_gt.data, f_data, clear_precision)
f_metrics.match_sequence()
evaluation = [f_metrics.get_mota(), f_metrics.get_motp(), f_metrics.get_fn_count(), f_metrics.get_fp_count(), f_metrics.get_mismatches_count(), f_metrics.get_object_count(), f_metrics.get_matches_count()]
print evaluation

k_metrics = clearmetrics.ClearMetrics(noplast_gt.data, noplast_ktrack.data, clear_precision)
k_metrics.match_sequence()
evaluation = [k_metrics.get_mota(), k_metrics.get_motp(), k_metrics.get_fn_count(), k_metrics.get_fp_count(), k_metrics.get_mismatches_count(), k_metrics.get_object_count(), k_metrics.get_matches_count()]
print evaluation

c_metrics = clearmetrics.ClearMetrics(noplast_gt.data, noplast_ctrax.data, clear_precision)
c_metrics.match_sequence()
evaluation = [c_metrics.get_mota(), c_metrics.get_motp(), c_metrics.get_fn_count(), c_metrics.get_fp_count(), c_metrics.get_mismatches_count(), c_metrics.get_object_count(), c_metrics.get_matches_count()]
print evaluation


plt.figure()
mc = get_frames(c_metrics.mismatches_in_frames)
fp = get_frames(c_metrics.fp_in_frames)
fn = get_frames(c_metrics.fn_in_frames)
plt.plot(np.ones(len(mc))*1, mc, 'bo', label='ctrax mismatches', markersize=mismatch_size)
plt.plot(np.ones(len(fp))*1, fp, 'bx', label='ctrax fp', markersize=fp_size)
plt.plot(np.ones(len(fn))*1, fn, 'b+', label='ctrax fn', markersize=fp_size)

mc = get_frames(k_metrics.mismatches_in_frames)
fp = get_frames(k_metrics.fp_in_frames)
fn = get_frames(k_metrics.fn_in_frames)
plt.plot(np.ones(len(mc))*2, mc, 'go', label='K-Track mismatches', markersize=mismatch_size)
plt.plot(np.ones(len(fp))*2, fp, 'gx', label='K-Track fp', markersize=fp_size)
plt.plot(np.ones(len(fn))*2, fn, 'g+', label='K-Track fn', markersize=fp_size)


mc = get_frames(f_metrics.mismatches_in_frames)
fp = get_frames(f_metrics.fp_in_frames)
fn = get_frames(f_metrics.fn_in_frames)

plt.plot(np.ones(len(mc))*3, mc, 'mo', label='Ferda mismatches', markersize=mismatch_size)
plt.plot(np.ones(len(fp))*3, fp, 'mx', label='Ferda fp', markersize=fp_size)
plt.plot(np.ones(len(fn))*3, fn, 'm+', label='Ferda fn', markersize=fp_size)


#plt.axis([0, frames, 0, 4.5])
plt.axis([0.5, 3.5, 0, frames])
#plt.legend(loc='bottom center', shadow=True, fontsize=17, numpoints=1, fancybox=True)
plt.grid(True)
plt.ylabel('frame number')
plt.xticks([1, 2, 3],['Ctrax', 'K-Track', 'Ferda'])
plt.subplots_adjust(hspace=0, wspace=0, left=0.08, bottom=0.02, right=0.999, top=0.999)
plt.show()


















mismatch_size = 25
fp_size = 10
frames = 6000





clear_precision = 20

f = open('trajectories_data/messor_ferda_13516_xy.arr', 'rb')
f_data = pickle.load(f)
f.close()


ferda_data = {}
for i in range(len(messor_gt.data)):
    ferda_data[i] = f_data[i]

f_metrics = clearmetrics.ClearMetrics(messor_gt.data, ferda_data, clear_precision)
f_metrics.match_sequence()
evaluation = [f_metrics.get_mota(), f_metrics.get_motp(), f_metrics.get_fn_count(), f_metrics.get_fp_count(), f_metrics.get_mismatches_count(), f_metrics.get_object_count(), f_metrics.get_matches_count()]
print evaluation



i_metrics = clearmetrics.ClearMetrics(messor_gt.data, messor_idtracker.data, clear_precision)
i_metrics.match_sequence()
evaluation = [i_metrics.get_mota(), i_metrics.get_motp(), i_metrics.get_fn_count(), i_metrics.get_fp_count(), i_metrics.get_mismatches_count(), i_metrics.get_object_count(), i_metrics.get_matches_count()]
print evaluation

plt.figure()
mc = get_frames(i_metrics.mismatches_in_frames)
fp = get_frames(i_metrics.fp_in_frames)
fn = get_frames(i_metrics.fn_in_frames)
plt.plot(np.ones(len(mc))*2, mc, 'ro', label='idTracker mismatches', markersize=mismatch_size)
plt.plot(np.ones(len(fp))*2, fp, 'rx', label='idtracker fp', markersize=fp_size)
plt.plot(np.ones(len(fn))*2, fn, 'r+', label='idtracker fn', markersize=fp_size)

mc = get_frames(f_metrics.mismatches_in_frames)
fp = get_frames(f_metrics.fp_in_frames)
fn = get_frames(f_metrics.fn_in_frames)

plt.plot(np.ones(len(mc))*1, mc, 'mo', label='Ferda mismatches', markersize=mismatch_size)
plt.plot(np.ones(len(fp))*1, fp, 'mx', label='Ferda fp', markersize=fp_size)
plt.plot(np.ones(len(fn))*1, fn, 'm+', label='Ferda fn', markersize=fp_size)


#plt.axis([0, frames, 0, 4.5])
plt.axis([0, 2.5, 0, frames])
plt.legend(loc='bottom center', shadow=True, fontsize=17, numpoints=1, fancybox=True)
plt.grid(True)
plt.ylabel('frame number')
plt.xticks([1, 2],['Ferda', 'idTracker'])
plt.show()



#c = [476, 620, 672, 673, 1081, 1147, 1175]
#c_fp = [512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 1189, 1190, 1191, 1192, 1193, 686, 687, 689, 690, 691, 693, 704, 705, 706, 708, 709, 714, 717, 1144, 722, 723, 724, 725, 727, 1146, 1284, 804, 1080, 1081, 1170, 1171, 1174, 1145, 451, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]
#
#k = [476, 653, 654, 681]
#k_fp = [770, 577, 649, 650, 651, 653, 654, 655, 656, 657, 658, 659, 708, 661, 662, 663, 664, 665, 666, 667, 474, 709, 804, 805, 679, 680, 682, 683, 684, 685, 686, 669, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 704, 449, 450, 451, 452, 453, 687, 715, 716, 717, 718, 719, 720, 209, 722, 723, 724, 725, 726, 727, 472, 473, 346, 475, 721, 807, 765, 758, 766, 707, 759, 760, 761, 762, 763, 764, 637, 638, 767]
#
#i = [1204, 1291, 1317, 1432, 1433]
#i_fp = [450, 803, 1284, 805, 806, 1180, 689, 692, 1174, 475, 1179, 1437, 767]
#
#f = [673, 691]
#f_fp = [450, 803, 1284, 806, 807, 1180, 689, 692, 1174, 475, 1179, 1437, 767]
#
#
#plt.figure()
#plt.plot([2.75, 3.25], [813, 813], color='red', linestyle='dashed', linewidth=3)
#plt.plot(np.ones(len(c))*4, c, 'bo', label='ctrax mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(c_fp))*4, c_fp, 'bx', label='ctrax fp', markersize=fp_size)
#
#plt.plot(np.ones(len(k))*3, k, 'go', label='K-Track mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(k_fp))*3, k_fp, 'gx', label='K-Track fp', markersize=fp_size)
#
#plt.plot(np.ones(len(i))*2, i, 'ro', label='idTracker mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(i_fp))*2, i_fp, 'rx', label='idtracker fp', markersize=fp_size)
#
#plt.plot(np.ones(len(f))*1, f, 'mo', label='Ferda mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(f_fp))*1, f_fp, 'mx', label='Ferda fp', markersize=fp_size)
#
#
##plt.axis([0, frames, 0, 4.5])
#plt.axis([0, 4.5, 0, frames])
#plt.legend(loc='bottom center', shadow=True, fontsize=17, numpoints=1, fancybox=True)
#plt.grid(True)
#plt.ylabel('frame number')
#plt.xticks([1, 2, 3, 4],['Ferda', 'idTracker', 'K-Track', 'ctrax'])
##plt.show()

#: *[0-9]
#: *\[[0-9]\]
#k = [1059, 1222, 1449, 1739, 1971, 1748, 1751, 1176, 2100, 572]
#k_fp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2062, 2063, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 60, 61, 62, 63, 68, 100, 101, 140, 141, 142, 2192, 145, 151, 152, 153, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 182, 183, 184, 185, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2190, 274, 2193, 434, 435, 436, 437, 456, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 2230, 2231, 2232, 2233, 1175, 1221, 1222, 1224, 1241, 1242, 1243, 1244, 1257, 1258, 1259, 1260, 1388, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1444, 1447, 1520, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1600, 1657, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1702, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1742, 1743, 1744, 1745, 1746, 1748, 1749, 1750, 1825, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1951, 1965, 1966, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2034, 2035]
#
#c = [2060, 913, 275, 916, 1687, 921, 923, 932, 933, 881, 1449, 170, 691, 950, 951, 961, 326, 332, 915, 614, 364, 1265, 882, 883, 888, 1658, 1659]
#c_fp = [939, 2055, 2056, 940, 2058, 855, 955, 941, 921, 856, 870, 857, 858, 944, 859, 945, 860, 946, 861, 947, 862, 2057, 948, 863, 949, 864, 2059, 950, 865, 866, 867, 838, 953, 600, 868, 2063, 869, 611, 612, 613, 956, 871, 872, 839, 873, 1658, 1659, 1660, 874, 875, 1961, 876, 840, 1682, 963, 1684, 1685, 1686, 964, 879, 909, 841, 690, 885, 915, 1958, 891, 806, 843, 807, 1962, 1263, 1264, 808, 809, 897, 1813, 1814, 1815, 1816, 1817, 799, 292, 293, 294, 295, 296, 297, 903, 812, 822, 823, 824, 825, 826, 827, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 844, 845, 846, 847, 336, 849, 850, 851, 852, 853, 854, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 877, 878, 829, 880, 881, 882, 883, 884, 830, 886, 887, 888, 889, 890, 831, 892, 893, 894, 895, 896, 832, 898, 899, 900, 901, 902, 833, 904, 905, 906, 907, 908, 834, 910, 911, 912, 913, 914, 835, 918, 919, 920, 836, 922, 923, 924, 925, 926, 837, 928, 417, 418, 931, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 942, 943, 432, 433, 434, 435, 436, 437, 438, 439, 952, 441, 954, 927, 444, 842, 958, 960, 961, 451, 828, 454, 929, 968, 969, 460, 930, 1963, 467, 1959, 932, 1683, 933, 848, 934, 2023, 2024, 935, 936, 1960, 937, 938]
#
#f = [1056, 1220, 1449, 2059, 1266, 1748, 1047, 1050]
#f_fp = [2058, 1043, 1044, 1045, 1046, 152, 1049, 155, 1054, 1055, 1058, 1966, 175, 176, 177, 178, 179, 180, 950, 183, 184, 1466, 1467, 961, 1219, 182, 1742, 1746, 1747, 2008, 2009, 616, 617, 618, 1264, 1265]
#frames = 2250
#
#plt.figure()
#plt.plot(np.ones(len(c))*3, c, 'bo', label='ctrax mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(c_fp))*3, c_fp, 'bx', label='ctrax fp', markersize=fp_size)
#
#plt.plot(np.ones(len(k))*2, k, 'go', label='K-Track mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(k_fp))*2, k_fp, 'gx', label='K-Track fp', markersize=fp_size)
#
#plt.plot(np.ones(len(f))*1, f, 'mo', label='Ferda mismatches', markersize=mismatch_size)
#plt.plot(np.ones(len(f_fp))*1, f_fp, 'mx', label='Ferda fp', markersize=fp_size)
#
#
##plt.axis([0, frames, 0, 4.5])
#plt.axis([0, 3.5, 0, frames])
#plt.legend(loc='bottom center', shadow=True, fontsize=17, numpoints=1, fancybox=True)
#plt.grid(True)
#plt.ylabel('frame number')
#plt.xticks([1, 2, 3],['Ferda', 'K-Track', 'ctrax'])
#plt.show()
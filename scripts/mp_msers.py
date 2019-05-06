__author__ = 'fnaiser'

import multiprocessing as mp
import random
import string
from core.region.mser import get_regions_in_img
from utils.video_manager import get_auto_video_manager
import numpy as np
from copy import deepcopy
import time

# Define an output queue
output = mp.Queue()

# define a example function
def compute_msers(i, img):
    m = get_regions_in_img(img)
    return i, m

N_FRAMES = 200
vid = get_auto_video_manager('/Users/fnaiser/Documents/eight.m4v')

# Setup a list of processes that we want to run
# processes = [mp.Process(target=compute_msers, args=(i, vid.next_frame().copy(), output)) for i in range(N_FRAMES)]

pool = mp.Pool()

frames = [vid.next_frame().copy() for i in range(N_FRAMES)]
print "INIT DONE"

start = time.time()
results = [pool.apply_async(compute_msers, args=(i, frames[i])) for i in range(N_FRAMES)]
output = [p.get() for p in results]
# print(output)
end = time.time()

for r in output:
    print r

print end - start


print "SEQUENCE"
start = time.time()
results2 = [compute_msers(i, frames[i]) for i in range(N_FRAMES)]
end = time.time()

# for r in results2:
#     print len(r)

print end - start
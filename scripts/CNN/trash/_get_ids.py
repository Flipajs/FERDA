from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
import os
import pickle
WD = '/Users/flipajs/Documents/dev/ferda/scripts/out9'

rids = []
for id in range(6):
    for fname in os.listdir(WD + '/'+str(id)):
        rids.append(fname.split('_')[-1][:-4])

rids = set(rids)
print(len(rids))

with open(WD+'/rids.pkl', 'wb') as f:
    pickle.dump(rids, f)


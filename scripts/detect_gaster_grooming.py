from core.project.project import Project
from core.id_detection.learning_process import LearningProcess
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from core.graph.region_chunk import RegionChunk
import random
import numpy as np

WD_PRE = '/Volumes/Seagate Expansion Drive/HH1_PRE'
WD_POST = '/Volumes/Seagate Expansion Drive/HH1_POST'

p_pre = Project()
p_pre.load(WD_PRE)

positive_examples = [
    (25050, 10691, 10691+203),
    (11083, 20740, 20740+47),
    (2247, 20787, 20787+158),
    (18623, 12264, 12264+147),
    (11065, 22791, 22791+8)
]



lp = LearningProcess(p_pre)

y = []
X = []

positive_examples_tids = set()

print "preparing positive examples..."
for t_id, start_f, end_f in positive_examples:
    print t_id
    positive_examples_tids.add(t_id)
    rt = RegionChunk(p_pre.chm[t_id], p_pre.gm, p_pre.rm)

    for f in range(start_f, end_f):
        r = rt.region_in_t(f)
        if r is None:
            print "frame: ", f
        y.append(1)
        X.append(lp.get_appearance_features(r))

print len(y), len(X)

print "preparing negative examples"

ch_ids = list(p_pre.chm.chunks_.keys())
num_examples = len(y)
with tqdm(total=num_examples) as pbar:
    while len(X)-num_examples < num_examples:
        random_t_id = random.choice(ch_ids)
        if random_t_id in positive_examples_tids:
            continue

        t = p_pre.chm[random_t_id]

        if not t.is_single():
            continue

        r = p_pre.gm.region(t[random.randint(0, len(t) - 1)])
        y.append(0)
        X.append(lp.get_appearance_features(r))

        pbar.update(1)

y = np.array(y)
X = np.array(X)
print y.shape, X.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print np.sum(y_train), np.sum(y_test), np.sum(y_test) * 5

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

print "TEST set accuracy: ", rfc.score(X_test, y_test)

# do P_POST
p_post = Project()
p_post.load(WD_POST)

X_post = []
rids = []
print "preparing HH1_POST data..."
for t in tqdm(p_post.chm.chunk_gen(), total=len(p_post.chm)):
    if t.is_single():
        for r_id in t.rid_gen(p_post.gm):
            r = p_post.rm[r_id]
            X_post.append(lp.get_appearance_features(r))
            rids.append(r.id())

X_post = np.array(X_post)
print X_post.shape

print "Classifying..."
predictions = rfc.predict(X_post)
rids = np.array(rids)

import pickle
print "SAVING..."
with open('/Users/flipajs/Documents/dev/ferda/scripts/gaster_grooming_out/HH1_post_predictions.pkl', 'wb') as f:
    pickle.dump((predictions, rids), f)

print "DONE"

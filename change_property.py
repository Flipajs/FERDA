import sys
import cPickle as pickle

pathToProj=sys.argv[1]
with open(pathToProj,'rb') as f:
   d=pickle.load(f)
if sys.argv[3]=='video_paths':
    d[sys.argv[3]]=[sys.argv[2]]
else:
    d[sys.argv[3]]=sys.argv[2]
pickle.dump(d,open(pathToProj,'wb'))

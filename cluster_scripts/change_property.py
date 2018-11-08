from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import sys
import pickle as pickle

def change_pr(path,value,property):
    pathToProj=path
    with open(pathToProj,'rb') as f:
        d=pickle.load(f)
    if property=='min_area':
        d['mser_parameters'].min_area=value
    if property=='video_paths':
        d[property]=[value]
    if property=='optimisation':
        d['other_parameters'].segmentation_use_roi_prediction_optimisation=value
    else:
        d[property]=value
    pickle.dump(d,open(pathToProj,'wb'))

if __name__ == "__main__":
    change_pr(sys.argv[1],sys.argv[2],sys.argv[3])

import sys
import cPickle as pickle

def change_pr(path,value,property):
    pathToProj=path
    with open(pathToProj,'rb') as f:
        d=pickle.load(f)
    if property=='video_paths':
        d[property]=[value]
    else:
        d[property]=value
    pickle.dump(d,open(pathToProj,'wb'))

if __name__ == "__main__":
    change_pr(sys.argv[1],sys.argv[2],sys.argv[3])

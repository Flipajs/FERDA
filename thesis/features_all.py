from __future__ import print_function
import cPickle as pickle



if __name__ == '__main__':
    with open('/Users/flipajs/Desktop/results_Sowbug3_75.pkl') as f:
        r = pickle.load(f)

    print(r)
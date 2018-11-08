from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
import pickle as pickle



if __name__ == '__main__':
    with open('/Users/flipajs/Desktop/results_Sowbug3_75.pkl') as f:
        r = pickle.load(f)

    print(r)
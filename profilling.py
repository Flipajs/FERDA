__author__ = 'flipajs'

import cPickle as pickle
from core.project import Project
from core.graph.solver import Solver
import cProfile as profile
from line_profiler import LineProfiler

if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/crop_1h00m-01h05m/c1_crop.fproj')

    solver = Solver(p)

    with open(p.working_directory+'/solver.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        solver.g = up.load()

    profiler = LineProfiler()
    profile.runctx('solver.simplify_to_chunks()', globals(), locals())
    # solver.simplify_to_chunks()
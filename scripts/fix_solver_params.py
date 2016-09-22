import sys
from core.project.project import Project
from core.project.solver_parameters import SolverParameters


if __name__ == '__main__':
    wd = sys.argv[1]

    p = Project()
    p.load(wd)

    old_sp = p.solver_parameters
    new_sp = SolverParameters()

    for key, val in old_sp.__dict__.iteritems():
        try:
            setattr(new_sp, key, val)
        except:
            pass
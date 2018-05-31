import sys
from core.project.project import Project
from core.project.solver_parameters import SolverParameters

if __name__ == '__main__':
    # to prepare it for cluster run as
    # python -m scripts.fix_solver_params "project root path" 0 1 1
    wd = sys.argv[1]
    use_emd_for_split_merge_detection = bool(int(sys.argv[2]))
    use_colony_split_merge_relaxation = bool(int(sys.argv[3]))

    p = Project()
    p.load(wd)

    old_sp = p.solver_parameters
    new_sp = SolverParameters()

    for key, val in old_sp.__dict__.iteritems():
        try:
            setattr(new_sp, key, val)
        except:
            pass

    p.save()
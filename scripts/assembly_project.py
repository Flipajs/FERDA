__author__ = 'flipajs'

from core.project.project import Project
import cPickle as pickle


def assembly(project_path, part_names):
    p = Project
    p.load(project_path)

    with open(p.working_directory+'/progress_save.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        g = up.load()
        log = up.load()

        solver = Solver(self)
        solver.g = g
        solver.ignored_nodes = ignored_nodes
        solver.update_nodes_in_t_refs()
        self.saved_progress = {'solver': solver}
        self.log = log
        print "FINISHED..."


if __name__ == "__main__":
    # set this:
    project_path = '/Users/flipajs/Documents/wd/eight_22/eight22.fproj'

    # let it empty if the format of parts is in following format:
    #
    part_names = []

__author__ = 'flipajs'

from core.project.project import Project
import cPickle as pickle
from core.graph.solver import Solver
import os, glob
from core.log import Log


def get_part_names(p):
    part_names = glob.glob(p.working_directory+'/part_*_progress_save.pkl')
    part_names = map(lambda x: x.replace(p.working_directory+'/', ''), part_names)

    nums = []
    for s in part_names:
        l = s.split('_')
        nums.append(int(l[1]))

    part_names = [y for (x, y) in sorted(zip(nums, part_names))]

    return part_names

def assembly(project_path, part_names):
    p = Project()
    p.load(project_path)

    if not part_names:
        part_names = get_part_names(p)

    solver = Solver(p)
    nodes_to_process = []
    end_nodes_prev = []
    for part, i in zip(part_names, range(len(part_names))):
        print "Processing ", part
        # this is slightly changed code from background_computer/assembly_after_parallelization
        with open(p.working_directory+'/'+part, 'rb') as f:
            up = pickle.Unpickler(f)
            g_ = up.load()

            local_solver = Solver(p)
            local_solver.g = g_
            local_solver.update_nodes_in_t_refs()

            for n, d in g_.nodes(data=True):
                solver.g.add_node(n, d)

            for n1, n2, d in g_.edges(data=True):
                solver.g.add_edge(n1, n2, d)

            start_nodes = local_solver.start_nodes()
            end_nodes = local_solver.end_nodes()

            nodes_to_process += end_nodes

            # check last and start frames...
            start_t = local_solver.start_t
            for n in end_nodes_prev[:]:
                if n.frame_ != start_t - 1:
                    end_nodes_prev.remove(n)

            for n in start_nodes[:]:
                if n.frame_ != start_t:
                    start_nodes.remove(n)

            solver.add_edges_(end_nodes_prev, start_nodes, fast=True)
            end_nodes_prev = end_nodes

    print "updating t references..."
    solver.update_nodes_in_t_refs()
    print "simplifying..."
    solver.simplify(nodes_to_process)
    print "chunks updating..."
    solver.simplify_to_chunks(nodes_to_process)
    print "saving..."

    # if there is progress_save, rename it to progress_save (copy) so it won't be overwritten
    try:
        os.rename(p.working_directory+'/progress_save.pkl', p.working_directory+'/progress_save.pkl (copy)')
    except:
        pass

    # clear it to save some space...
    p.log = Log()
    solver.save()
    print "PART ASSEMBLY COMPLETED!"


if __name__ == "__main__":
    # set this:
    project_path = '/Users/flipajs/Documents/wd/eight_22/eight22.fproj'

    # leave it empty if the format of parts is in following format:
    # part_#_progress_save.pkl where # is a number
    # else add the names of files in right order into part_names list:
    # e.g. part_names = ['1_progress_save.pkl', 'different_name.pkl', ...]
    part_names = []

    p = Project()
    p.load(project_path)

    # print get_part_names(p)
    assembly(project_path, part_names)
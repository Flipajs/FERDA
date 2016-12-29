from config import *
from core.project.project import Project
import cPickle as pickle


def load_all_projects(semistate='', update_t_nodes=False, add_single_vertices=False):
    """
    Returns: dictionary with projects

    """

    projects = {}
    for p_name, path in project_paths.iteritems():
        p = Project()
        # p.load(path)

        # from core.graph.chunk_manager import ChunkManager
        #
        p.load_semistate(path, semistate, update_t_nodes=update_t_nodes, one_vertex_chunk=add_single_vertices)
        # p.chm = ChunkManager()
        # with open(p.working_directory + '/temp/isolation_score.pkl', 'rb') as f:
        #     up = pickle.Unpickler(f)
        #     p.gm.g = up.load()
        #     up.load()
        #     chm = up.load()
        #     p.chm = chm

        # from core.region.region_manager import RegionManager
        #
        # p.rm = RegionManager(p.working_directory + '/temp', db_name='part0_rm.sqlite3')
        # p.gm.rm = p.rm

        projects[p_name] = p


    return projects


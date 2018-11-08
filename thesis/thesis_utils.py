from __future__ import absolute_import
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from .config import *
from core.project.project import Project
import pickle as pickle


def load_all_projects(semistate='isolation_score', update_t_nodes=False, add_single_vertices=False):
    """
    Returns: dictionary with projects

    """

    projects = {}
    for p_name, path in project_paths.items():
        p = Project()
        p.load_semistate(path, semistate, update_t_nodes=update_t_nodes, one_vertex_chunk=add_single_vertices)
        projects[p_name] = p

    return projects


def load_p(path, semistate='isolation_score', update_t_nodes=False, add_single_vertices=False):
    p = Project()
    p.load_semistate(path, semistate, update_t_nodes=update_t_nodes, one_vertex_chunk=add_single_vertices)

    return p

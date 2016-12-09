from config import *
from core.project.project import Project

def load_all_projects():
    """
    Returns: dictionary with projects

    """

    p_cam1 = Project(cam1_path)
    p_cam1.load()


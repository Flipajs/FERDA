from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
__author__ = 'flipajs'

from utils.video_manager import get_auto_video_manager
import pickle as pickle
from core.project.mser_parameters import MSERParameters
from utils.color_manager import colorize_project

class CompatibilitySolver(object):
    def __init__(self, project):
        self.project = project

        if project.version == '2.2.8' or project.version == '2.2.7' or project.version == '2.2.6' or project.version == '2.2.5':
            self.fix_225()

        if project.version == '2.2.9':
            self.fix_color_manager()

        if project.version_is_le('2.2.4'):
            raise Exception("Project version is < 2.2.5, if necessary, there will be compability solver implemented in future...")

        if project.version_is_le('2.2.2'):
            print("PROJECT version is <= 2.2.2, there was major speedup of saving and loading in recent versions... Hold on, the project will be repaired.")
            self.fix_chunks()
            self.project.version = "2.2.3"
            # as the files were opened and resaved, it is solved, so save the new version...
            self.project.save_project_file_()

    def fix_color_manager(self):
        if self.project.solver:
            colorize_project(self.project)
            self.project.version = '2.3.0'
            self.project.save_project_file_()

    def fix_225(self):
        assert False, 'MSERParameters constructor changed'
        self.project.mser_parameters = MSERParameters(initial_data=self.project)
        self.project.version = '2.2.9'
        self.project.save_project_file_()

    def fix_chunks(self):
        "fix chunks and resave with new protocol"
        print("compatibility fix in progress...")
        try:
            val = self.project.solver_parameters.frames_in_row
        except:
            self.project.solver_parameters.frames_in_row = 100

        try:
            val = self.project.other_parameters.store_area_info
        except:
            self.project.other_parameters.store_area_info = False

        # fix saved progress...
        if self.project.solver:
            solver = self.project.solver
            g = solver.g

            for n in g:
                is_ch, t_reversed, ch = solver.is_chunk(n)
                if is_ch and not t_reversed:
                    ch.store_area = False

            solver.save()

        # fix temp files...
        vid = get_auto_video_manager(self.project)
        frame_num = int(vid.total_frame_count())
        part_num = int(old_div(frame_num, self.project.solver_parameters.frames_in_row))

        for i in range(part_num):
            try:
                with open(self.project.working_directory+'/temp/g_simplified'+str(i)+'.pkl', 'rb') as f:
                    up = pickle.Unpickler(f)
                    g_ = up.load()
                    start_nodes = up.load()
                    end_nodes = up.load()

                    from core.graph.solver import Solver

                    solver = Solver(self.project)
                    solver.g = g_

                    for n in g_:
                        is_ch, t_reversed, ch = solver.is_chunk(n)
                        if is_ch and not t_reversed:
                            ch.store_area = False

                with open(self.project.working_directory+'/temp/g_simplified'+str(i)+'.pkl', 'wb') as f:
                    p = pickle.Pickler(f, -1)
                    p.dump(g_)
                    p.dump(start_nodes)
                    p.dump(end_nodes)

                print(old_div(i, float(part_num)))
            except:
                pass
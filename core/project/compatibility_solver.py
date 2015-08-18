__author__ = 'flipajs'

from utils.video_manager import get_auto_video_manager
import cPickle as pickle


class CompatibilitySolver:
    def __init__(self, project):
        self.project = project

        if project.version_is_le('2.2.2'):
            print "PROJECT version is <= 2.2.2, there was major speedup of saving and loading in recent versions... Hold on, the project will be repaired."
            self.fix_chunks()
            self.project.version = "2.2.3"
            # as the files were opened and resaved, it is solved, so save the new version...
            self.project.save_project_file_()

    def fix_chunks(self):
        "fix chunks and resave with new protocol"
        print "compatibility fix in progress..."
        try:
            val = self.project.solver_parameters.frames_in_row
        except:
            self.project.solver_parameters.frames_in_row = 100

        # fix saved progress...
        if self.project.saved_progress:
            solver = self.project.saved_progress['solver']
            g = solver.g

            for n in g:
                is_ch, t_reversed, ch = solver.is_chunk(n)
                if is_ch and not t_reversed:
                    ch.store_area = False

            solver.save()

        # fix temp files...
        vid = get_auto_video_manager(self.project)
        frame_num = int(vid.total_frame_count())
        part_num = int(frame_num / self.project.solver_parameters.frames_in_row)

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

                print i / float(part_num)
            except:
                pass

        self.project.saved_progress
import sys
from core.graph.region_chunk import RegionChunk
import numpy as np
import cPickle as pickle
from core.graph.chunk_manager import ChunkManager
from core.project.project import Project
from core.region.region_manager import RegionManager
import scipy.io as sio


class Exporter:
    def __init__(self, chm, gm, rm, pts_export=False, contour_pts_export=True):
        self.chm = chm
        self.gm = gm
        self.rm = rm

        self.pts_export = pts_export
        self.contour_pts_export = contour_pts_export

    def init_struct_(self, region, last_frame):
        d = {'x': [],
             'y': [],
             'area': [],
             'mean_area': 0,
             'velocity_x': [],
             'velocity_y': [],
             'frame_offset': region.frame(),
             # in matlab indexed from 1...
             'first_frame': region.frame() + 1,
             'last_frame': last_frame + 1,
             'num_frames': last_frame + 1 - region.frame(),
             'region_id': [],
             }

        if self.pts_export:
            d['region'] = []

        if self.contour_pts_export:
            d['region_contour'] = []

        return d

    def add_line_mat(self, d, r):
        y, x = r.centroid()
        d['x'].append(x)
        d['y'].append(y)

        d['area'].append(r.area())
        d['mean_area'] += r.area()

        velocity_x = 0
        velocity_y = 0
        if len(d['x']) > 1:
            velocity_x = x - d['x'][-2]
            velocity_y = y - d['y'][-2]

        d['velocity_x'].append(velocity_x)
        d['velocity_y'].append(velocity_y)

        if self.pts_export:
            pts = r.pts()
            self.append_pts_(d, 'region', pts)

        if self.contour_pts_export:
            pts = r.contour_without_holes()

            if pts is None:
                print "WARNING> PTS is None in export_part.py"
                print r
                with open('r_debug.pkl', 'wb') as f:
                    pickle.dump(r, f)

            self.append_pts_(d, 'region_contour', pts)

        d['region_id'].append(r.id_)

    def append_pts_(self, d, key, pts):
        px = []
        py = []
        for pt in pts:
            py.append(pt[0])
            px.append(pt[1])

        d[key].append({'x': np.array(px), 'y': np.array(py)})

    def obj_arr_append_(self, obj_arr, d):
        new_d = {}
        for key, val in d.iteritems():
            if key != 'frame' and key != 'region_id':
                val = np.array(val)

            new_d[key] = val

        obj_arr.append(d)

    def export(self, file_name, min_tracklet_length=1):
        obj_arr = []

        # it is important to go through vertices to have access to active feature...
        # When processing one part there are inactive chunks in chm...

        for ch in self.chm.chunks_.itervalues():
            if ch.length() < min_tracklet_length:
                continue

            rch = RegionChunk(ch, self.gm, self.rm)
            d = self.init_struct_(rch[0], ch.end_frame(self.gm))

            for r in rch.regions_gen():
                self.add_line_mat(d, r)

            d['mean_area'] /= float(ch.length())

            npx = np.array(d['x'])
            npy = np.array(d['y'])
            square_displacement = np.power(npx.mean() - npx, 2) + np.power(npy.mean() - npy, 2)
            displacement = np.sqrt(square_displacement)
            d['mean_squared_displacement'] = square_displacement.mean()
            d['displacement_mean'] = displacement.mean()
            d['displacement_std'] = displacement.std()

            self.obj_arr_append_(obj_arr, d)

        with open(file_name+'.mat', 'wb') as f:
            sio.savemat(f, {'FERDA': obj_arr}, do_compression=True)

class FakeBGComp:
    def __init__(self, project, first_part, part_num):
        self.project = project
        self.part_num = part_num
        self.first_part = first_part
        self.do_semi_merge = True

    def update_callback(self, fake1=None, fake2=None):
        pass

    def finished_callback(self, fake1=None, fake2=None):
        pass


def export_arena(out_path, project):
    with open(out_path + '_arena.mat', 'wb') as f:
        arena = None
        if project.arena_model:
            am = project.arena_model
            try:
                c = am.center
                radius = am.radius
            except AttributeError:
                center = np.array([0, 0])
                num = 0
                # estimate center:
                for y in range(am.im_height):
                    for x in range(am.im_width):
                        if am.mask_[y, x]:
                            center += np.array([y, x])
                            num += 1

                c = center / num
                radius = round((num / np.pi) ** 0.5)

            arena = {'cx': c[1], 'cy': c[0], 'radius': radius}
            try:
                arena['y1'] = project.video_crop_model['y1']
                arena['x1'] = project.video_crop_model['x1']
                arena['y2'] = project.video_crop_model['y2']
                arena['x2'] = project.video_crop_model['x2']
            except:
                arena['y1'] = 0
                arena['x1'] = 0
                arena['y2'] = 0
                arena['x2'] = 0

        sio.savemat(f, {'arena': arena}, do_compression=True)

if __name__ == '__main__':
    working_dir = sys.argv[1]
    out_dir = sys.argv[2]
    first_part = int(sys.argv[3])
    part_num = int(sys.argv[4])
    min_tracklet_length = int(sys.argv[5])
    pts_export = bool(int(sys.argv[6]))

    i = first_part

    p = Project()
    p.load(working_dir)

    if i == 0:
        export_arena(out_dir, p)

    bgcomp = FakeBGComp(p, first_part, part_num)

    from core.bg_computer_assembling import assembly_after_parallelization
    assembly_after_parallelization(bgcomp)

    # rm = RegionManager(db_wd=working_dir+ '/temp',
    #                    db_name='part' + str(i) + '_rm.sqlite3',
    #                    cache_size_limit=1)
    #
    # with open(working_dir+'/temp/part'+str(i)+'.pkl', 'rb') as f:
    #     up = pickle.Unpickler(f)
    #     g_ = up.load()
    #     relevant_vertices = up.load()
    #     chm_ = up.load()
    #
    # chm = ChunkManager()
    # for v_id in relevant_vertices:
    #     if not g_.vp['active'][v_id]:
    #         continue
    #

    #     v = g_.vertex(v_id)
    #     ch_id = g_.vp['chunk_start_id'][v]
    #
    #     if ch_id > 0:
    #         chm.chunks_[ch_id] = chm_[ch_id]

    # p = Project()
    # p.load(working_dir)
    # from core.graph.graph_manager import GraphManager
    # p.gm = GraphManager(p, None)
    # p.gm.g = g_
    # p.gm.rm = rm

    fname = out_dir+'/out_'+str(i)
    if first_part+part_num-1 > i:
        fname += '-'+str(first_part+part_num-1)

    Exporter(p.chm, p.gm, p.rm, pts_export).export(fname, min_tracklet_length=min_tracklet_length)

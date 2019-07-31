from core.region.region import Region
import tqdm
import sys
from UserList import UserList
# from collections import UserList
from core.region.region import RegionExtStorage
import pandas as pd
from os.path import join
import h5py
import numpy as np
import tempfile
from shutil import move, copyfile
import os
import warnings


class RegionManager(UserList):
    def __init__(self, regions=None):
        self.regions_df = None
        self.regions_h5 = None
        self.region_pts_h5_dataset = None
        self.region_contour_h5_dataset = None
        self.is_region_pts_temp = True

        self.init_dataframe()
        self.init_h5_store()
        if regions is None:
            self.data = []
        else:
            self.data = regions[:]

    @classmethod
    def from_dir(cls, directory):
        rm = cls()
        rm.regions_df = pd.read_csv(join(directory, 'regions.csv')).set_index('id_', drop=False)
        try:
            rm.open_h5_store(join(directory, 'regions.h5'))
        except Exception as e:
            warnings.warn(e.message)
            warnings.warn('Initializing a new region manager h5 store in temporary location.')
            rm.init_h5_store()
        if not rm.regions_df.empty:
            n = rm.regions_df.index.max() + 1
            rm.data = [None] * n
            for i in rm.regions_df.index:
                rm.data[i] = RegionExtStorage(i, rm.regions_df, rm.region_pts_h5_dataset, rm.region_contour_h5_dataset)
        return rm

    def __add__(self, other):
        added = super(RegionManager, self).__add__(other)
        added.regions_df = pd.concat([self.regions_df, other.regions_df], ignore_index=True, sort=False)
        added.regions_df.id_ = added.regions_df.index
        added.init_h5_store(filename=None, num_items=len(added.regions_df))
        added.region_pts_h5_dataset[:len(self.regions_df)] = self.region_pts_h5_dataset[:len(self.regions_df)]
        added.region_pts_h5_dataset[len(self.regions_df):] = other.region_pts_h5_dataset[:len(other.regions_df)]
        added.region_contour_h5_dataset[:len(self.regions_df)] = self.region_contour_h5_dataset[:len(self.regions_df)]
        added.region_contour_h5_dataset[len(self.regions_df):] = other.region_contour_h5_dataset[:len(other.regions_df)]
        return added

    def append(self, item):
        if item.id() is None:
            item.id_ = len(self)
        assert item.id() == len(self)
        super(RegionManager, self).append(item)

    def extend(self, other):
        n_self = len(self.regions_df)
        n_other = len(other.regions_df)
        df = pd.concat([self.regions_df, other.regions_df], ignore_index=True, sort=False)
        df.id_ = df.index
        self.regions_df = df

        if not (self.regions_h5.mode in ['r+', 'w', 'a']):
            filename = self.regions_h5.filename
            self.regions_h5.close()
            self.open_h5_store(filename, 'r+')
        self.region_pts_h5_dataset.resize(len(self.regions_df), axis=0)
        self.region_pts_h5_dataset[n_self:] = other.region_pts_h5_dataset[:n_other]
        self.region_contour_h5_dataset.resize(len(self.regions_df), axis=0)
        self.region_contour_h5_dataset[n_self:] = other.region_contour_h5_dataset[:n_other]
        # self.regions_h5.close()
        # self.open_h5_store(filename)

        other_fixed_external = []
        for i, r in enumerate(other):
            if isinstance(r, RegionExtStorage):
                other_fixed_external.append(RegionExtStorage(i + n_self, self.regions_df,
                                                             self.region_pts_h5_dataset, self.region_contour_h5_dataset))
            elif isinstance(r, Region):
                other_fixed_external.append(r)
            else:
                assert False
        super(RegionManager, self).extend(other_fixed_external)

    def regions_to_ext_storage(self):
        self.regions_df = self.get_regions_df()
        for i, r in enumerate(tqdm.tqdm(self.data, desc='converting Regions to RegionExtStorage')):
            if r is not None and not isinstance(r, RegionExtStorage):
                assert isinstance(r, Region)
                assert r.id() == i
                pts = r.pts()
                if pts is None:
                    pts = np.array([])
                else:
                    pts = pts.flatten()
                self.region_pts_h5_dataset[i] = pts
                self.region_contour_h5_dataset[i] = r.contour().flatten()
                self.data[i] = RegionExtStorage(i, self.regions_df,
                                                self.region_pts_h5_dataset, self.region_contour_h5_dataset)

    def init_dataframe(self):
        self.regions_df = pd.DataFrame(columns=Region().__getstate__(flatten=True).keys()).set_index('id_', drop=False)

    def init_h5_store(self, filename=None, num_items=1000):
        if filename is None:
            f = tempfile.NamedTemporaryFile()
            f.close()
            filename = f.name
            self.is_region_pts_temp = True
        self.regions_h5 = h5py.File(filename, mode='w')
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        self.regions_h5.create_dataset('region_pts', (num_items,), dtype=dt, maxshape=(None,))  # , compression='gzip'
        self.region_pts_h5_dataset = self.regions_h5['region_pts']
        self.regions_h5.create_dataset('contour_pts', (num_items,), dtype=dt, maxshape=(None,))  # , compression='gzip'
        self.region_contour_h5_dataset = self.regions_h5['contour_pts']

    def open_h5_store(self, filename, mode='r'):
        self.regions_h5 = h5py.File(filename, mode=mode)  # TODO
        self.region_pts_h5_dataset = self.regions_h5['region_pts']
        self.region_contour_h5_dataset = self.regions_h5['contour_pts']
        self.is_region_pts_temp = False
        for r in self.data:
            if isinstance(r, RegionExtStorage):
                r.pts_h5_dataset = self.region_pts_h5_dataset
                r.contour_h5_dataset = self.region_contour_h5_dataset

    def get_regions_df(self):
        if all([isinstance(r, RegionExtStorage) for r in self.data]) or \
                (len(self.data) == 0 and len(self.regions_df) == 0):
            return self.regions_df
        else:
            return self.regions2dataframe(self.data)

    @staticmethod
    def regions2dataframe(regions):
        null_region_state = {k: np.nan for k in Region().__getstate__(flatten=True).keys()}
        regions_dicts = [r.__getstate__(flatten=True) if r is not None else null_region_state for r in regions]
        return pd.DataFrame(regions_dicts).set_index('id_', drop=False)

    def save(self, directory):
        self.regions_to_ext_storage()
        self.regions_df.to_csv(join(directory, 'regions.csv'), index=False)
        regions_filename = join(directory, 'regions.h5')
        if regions_filename == self.regions_h5.filename:
            self.regions_h5.flush()
        else:
            filename = self.regions_h5.filename
            self.regions_h5.close()
            if not self.is_region_pts_temp:
                copyfile(filename, regions_filename)
            else:
                move(filename, regions_filename)
                self.is_region_pts_temp = False
            self.open_h5_store(join(directory, 'regions.h5'))

    def close(self):
        filename = self.regions_h5.filename
        self.regions_h5.close()
        if self.is_region_pts_temp:
            os.remove(filename)


# def concatenate(regionmanagers):
#     super(RegionManager, self).extend(other)
#     df = pd.concat([self.regions_df, other.regions_df], ignore_index=True, sort=False)
#     df.id_ = df.index
#     filename = self.regions_h5.filename
#     if self.regions_h5.mode != 'a' or self.regions_h5.mode != 'r+':
#         self.regions_h5.close()
#     self.open_h5_store(filename, 'r+')
#     self.region_pts_h5_dataset.resize(len(df), axis=0)
#     self.region_pts_h5_dataset[len(self.regions_df):] = other.region_pts_h5_dataset[:len(other.regions_df)]
#     # self.regions_h5.close()
#     # self.open_h5_store(filename)
#     self.regions_df = df
#
#     out_rm = RegionManager()
#     for rm in regionmanagers:


if __name__ == "__main__":
    from core.project.project import Project
    # p = Project('/home/matej/prace/ferda/projects/2_temp/190126_0800_Cam1_ILP_cardinality_dense_fix_orientation')
    # # p = Project('/home.stud/smidm/ferda/projects/2_temp/190126_0800_Cam1_ILP_cardinality_dense_fix_orientation')
    #
    nrm2 = RegionManager.from_dir('out')

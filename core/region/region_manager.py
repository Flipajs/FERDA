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
from shutil import copyfile
import os


class RegionManager(UserList):
    def __init__(self):
        self.regions_df = None
        self.region_pts_h5 = None
        self.region_pts_h5_dataset = None
        self.is_region_pts_temp = True

        self.init_dataframe()
        self.init_h5_store()
        self.data = []

    @classmethod
    def from_dir(cls, directory):
        rm = cls()
        rm.regions_df = pd.read_csv(join(directory, 'regions.csv')).set_index('id_', drop=False)
        rm.open_h5_store(join(directory, 'regions.h5'))
        if not rm.regions_df.empty:
            n = rm.regions_df.index.max() + 1
            rm.data = [None] * n
            for i in rm.regions_df.index:
                rm.data[i] = RegionExtStorage(i, rm.regions_df, rm.region_pts_h5_dataset)
        return rm

    def regions_to_ext_storage(self):
        for i, r in enumerate(tqdm.tqdm(self.data, desc='converting Regions to RegionExtStorage')):
            if r is not None and not isinstance(r, RegionExtStorage):
                assert isinstance(r, Region)
                assert r.id() == i
                self.regions_df.loc[i] = pd.Series(r.__getstate__(flatten=True))
                pts = r.pts()
                if pts is None:
                    pts = np.array([])
                else:
                    pts = pts.flatten()
                self.region_pts_h5_dataset[i] = pts
                self.data[r.id()] = RegionExtStorage(self.regions_df.loc[i], self.region_pts_h5_dataset)

    def init_dataframe(self):
        self.regions_df = pd.DataFrame(columns=Region().__getstate__(flatten=True).keys()).set_index('id_', drop=False)

    def init_h5_store(self, filename=None, num_items=1000):
        if filename is None:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            filename = f.name
            self.is_region_pts_temp = True
        self.region_pts_h5 = h5py.File(filename, mode='w')
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        self.region_pts_h5.create_dataset('region_pts', (num_items,), dtype=dt, maxshape=(None,))  # , compression='gzip'
        self.region_pts_h5_dataset = self.region_pts_h5['region_pts']

    def open_h5_store(self, filename):
        self.region_pts_h5 = h5py.File(filename, mode='r')  # TODO 'a'
        self.region_pts_h5_dataset = self.region_pts_h5['region_pts']
        self.is_region_pts_temp = False

    @staticmethod
    def regions2dataframe(regions):
        null_region_state = {k: np.nan for k in Region().__getstate__(flatten=True).keys()}
        regions_dicts = [r.__getstate__(flatten=True) if r is not None else null_region_state for r in regions]
        return pd.DataFrame(regions_dicts).set_index('id_', drop=False)

    def save(self, directory):
        self.regions_to_ext_storage()
        self.regions_df.to_csv(join(directory, 'regions.csv'), index=False)
        regions_filename = join(directory, 'regions.h5')
        if regions_filename == self.region_pts_h5.filename:
            self.region_pts_h5.flush()
        else:
            filename = self.region_pts_h5.filename
            self.region_pts_h5.close()
            copyfile(filename, regions_filename)
            if self.is_region_pts_temp:
                os.remove(filename)
                self.is_region_pts_temp = False
            self.open_h5_store(join(directory, 'regions.h5'))


if __name__ == "__main__":
    from core.project.project import Project
    # p = Project('/home/matej/prace/ferda/projects/2_temp/190126_0800_Cam1_ILP_cardinality_dense_fix_orientation')
    # # p = Project('/home.stud/smidm/ferda/projects/2_temp/190126_0800_Cam1_ILP_cardinality_dense_fix_orientation')
    #
    nrm2 = RegionManager.from_dir('out')

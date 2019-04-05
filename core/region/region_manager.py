from core.region.region import Region, encode_RLE

__author__ = 'flipajs'

import sqlite3 as sql
import cPickle as pickle
from core.region.region import encode_RLE
from libs.cachetools import LRUCache
import tqdm
import sys
from UserList import UserList
# from collections import UserList
# from collections.abc import MutableSequence
from core.region.region import RegionExtStorage
import pandas as pd
from os.path import join
import h5py
import numpy as np
import tempfile
from shutil import copyfile
import os


class Dummy():
    def __getitem__(self, item):
        return None

    def __setitem__(self, key, value):
        pass


class RegionManager:
    def __init__(self, db_wd=None, db_name="rm.sqlite3", cache_size_limit=1000, data=None, supress_init_print=False):
        """
        RegionManager is designed to store regions data. By default, all data is stored in memory cache (dictionary) and
        identified using unique ids. Optionally, database can be used, in which case the memory cache size can be
        limited to reduce memory usage.
        :param db_wd: Working directory in which the database file should be stored
        :param db_name: Database name, "regions.db" by default
        :param cache_size_limit: Number of instances to be held in cache
        :return: None
        """
        if cache_size_limit == -1:
            import warnings
            warnings.warn("cache size limit -1 - infinity is not supported right now!!!")

        self.cache_size_limit_ = cache_size_limit

        self.cache = Dummy()
        if cache_size_limit != 0:
            self.cache = LRUCache(maxsize=cache_size_limit, missing=lambda x: None, getsizeof=lambda x: 1)

        if db_wd == None:
            # cache mode (no db set)
            # if cache_size_limit == -1:
            self.use_db = False
            # self.regions_cache_ = {}
            # self.recent_regions_ids = []
            self.id_ = 0
            # else:
            #     raise SyntaxError("Cache limit can only be set when database is used!")
        else:
            self.open_db(db_wd, db_name, supress_init_print)
        self.tmp_ids = []

        if isinstance(data, RegionManager):
            newdata = data[:]
            self.add(newdata)
        elif isinstance(data, list):
            for datas in data:
                if isinstance(datas, RegionManager):
                    self.add(datas[:])

    def open_db(self, db_wd, db_name, supress_init_print=True):
        self.use_db = True
        db_path = db_wd + "/" + db_name
        if not supress_init_print:
            print("Initializing db at %s " % db_path)
        self.con = sql.connect(db_path)
        self.cur = self.con.cursor()
        # DEBUG, do not use! self.cur.execute("DROP TABLE IF EXISTS regions;")
        self.cur.execute("CREATE TABLE IF NOT EXISTS regions(\
                    id INTEGER PRIMARY KEY, \
                    data BLOB);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS regions_index ON regions(id);")
        self.use_db = True
        # self.regions_cache_ = {}
        # self.recent_regions_ids = []
        # if database has been used before, get last used ID and continue from it (IDs always have to stay unique)
        try:
            self.cur.execute("SELECT id FROM regions ORDER BY id DESC LIMIT 1;")
            row = self.cur.fetchone()
            self.id_ = row[0]
        except TypeError:  # TypeError is raised when row is empty (no IDs were found)
            self.id_ = 0

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['cur']
        del d['con']
        return d

    def add(self, regions):
        """
        Save one or more regions in RegionManager
        :param regions: (region/list of regions) - regions that should be added into RegionManager.
        :return (list of int/ints) - ids that were given to appended regions. Regions can be later accessed via these ids
        """

        # use one return list for simplicity - the ids are appended to it from the get_next_id function
        self.tmp_ids = []

        if isinstance(regions, list):
            if self.use_db:
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.executemany("INSERT INTO regions VALUES (?, ?)", self.add_iter_(regions))
                # self.id and self.tmp_ids are updated in self.add_iter_
                self.con.commit()
                return self.tmp_ids
            else:
                for r in regions:
                    if not isinstance(r, Region):
                        raise TypeError ("Region manager can only work with Region objects, not %s" % type(r))
                    id = self.get_next_id()
                    r.id_ = id
                    self.add_to_cache_(id, r)
                return self.tmp_ids

        if not isinstance(regions, Region):
            raise TypeError ("Region manager can only work with Region objects, not %s" % type(regions))

        id = self.get_next_id()
        regions.id_ = id
        # self.add_to_cache_(id, regions)
        if self.use_db:
            self.cur.execute("BEGIN TRANSACTION;")
            self.cur.execute("INSERT INTO regions VALUES (?, ?)", (id, self.prepare_region(regions)))
            self.con.commit()
        else:
            # id = self.get_next_id()
            # r.id_ = id
            self.add_to_cache_(id, regions)

        return self.tmp_ids

    def update(self, rid, region):
        self.cur.execute("BEGIN TRANSACTION;")
        self.cur.execute("UPDATE regions SET data = ? WHERE id = ?", (self.prepare_region(region), rid))
        self.con.commit()

    def clear_cache(self):
        self.cache.clear()
        # self.regions_cache_     = {}
        # self.recent_regions_ids = []

    def add_to_cache_(self, id, region):
        """
        This method adds region with id to the cache. It also updates it's position in recent_regions_ids and checks
        the cache size.
        :param id:
        :param region:
        :return None
        """

        self.cache[id] = region
        #
        # # print "Adding %s %s" % (id, region)
        # # print "Cache: %s" % self.recent_regions_ids
        # if id in self.recent_regions_ids:
        #     # remove region from recent_regions_ids
        #     self.recent_regions_ids.remove(id)
        #     # print "Moving %s up" % id
        #     # add region to fresh position in recent_regions_ids and add it to cache
        #     self.recent_regions_ids.append(id)
        #     self.regions_cache_[id] = region
        # else:
        #     # add region to fresh position in recent_regions_ids and add it to cache
        #     self.recent_regions_ids.append(id)
        #     self.regions_cache_[id] = region
        #
        #     if self.cache_size_limit_ > 0 and len(self.regions_cache_) > self.cache_size_limit_:
        #         pop_id = self.recent_regions_ids.pop(0)
        #         # TODO: Dita why .id() ?
        #         # self.regions_cache_.pop(pop_id, None).id()
        #         self.regions_cache_.pop(pop_id, None)
        #
        #         # print "Cache limit (%s) reached, popping id %s" % (self.cache_size_limit_, pop_id)

    def update_recent_regions_ids(self, key, region):
        """
        Renew the position of key in recent_regions_ids
        :param key: key of the region
        :return: None
        """

        # remove region from recent_regions_ids
        if key in self.recent_regions_ids:
            self.recent_regions_ids.remove(key)
            # add region to fresh position in recent_regions_ids and add it to cache
            self.recent_regions_ids.append(key)
        else:
            self.add_to_cache_(key, region)

    def add_iter_(self, regions):
        """
        Iterator over given regions, yields a tuple used for sql executemany. self.tmp_ids and self.id_ get modified.
        :return tuple (id, binary region data)
        """
        for r in regions:
            if not isinstance(r, Region):
                self.con.rollback()
                raise TypeError ("Region manager can only work with Region objects, not %s" % type(r))
            id = self.get_next_id()
            r.id_ = id

            if self.cache_size_limit_ != 0:
                self.add_to_cache_(id, r)

            yield (id, self.prepare_region(r))

    def get_next_id(self):
        self.id_ += 1
        self.tmp_ids.append(self.id_)
        return self.id_

    def __getitem__(self, key):
        sql_ids = []
        result = []
        if isinstance(key, slice):
            # TODO: check how this example works and if it can be used
            # return [self[ii] for ii in xrange(*key.indices(len(self)))]
            # get values from slice
            start = key.start
            if start == None or start == 0:
                start = 1
            stop = key.stop
            # value of "stop" is 9223372036854775807 when using [:], but it works fine with [::]. Hotfixed.
            if stop == None or stop == 9223372036854775807:
                stop = len(self) +1
            step = key.step
            if step == None:
                step = 1
            # check if slice parameters are ok
            if start < 0 or start > len(self) or stop > len(self)+1 or stop < 0 or (stop < start and step > 0) or step == 0:
                raise ValueError("Invalid slice parameters (%s:%s:%s)" % (start, stop, step))

            # go through slice
            for i in range(start, stop, step):
                r = self.cache[i]
                if r:
                # if i in self.regions_cache_:
                #     # print "%s is in cache" % i
                #     # use cache if region is available
                #     r = self.regions_cache_[i]
                    result.append(r)
                #     # self.update(i, r)
                else:
                    # print "%s is not in cache" % i
                    # if not, add id to the list of ids to be fetched from db
                    sql_ids.append(i)
            if self.use_db:
                self.db_search_(result, sql_ids)

        elif isinstance(key, list):
            # TODO: result might be in different order then keys!!!
            for id in key:
                if not isinstance(id, int):
                    print "TypeError: int expected, %s given! Skipping key '%s'." % (type(id), id)
                    continue
                r = self.cache[id]
                if r:
                    print r
                # if id in self.regions_cache_:
                #     # print "%s was found in cache" % id
                #     r = self.regions_cache_[id]
                    result.append(r)
                    # self.update(id, r)
                else:
                    # print "%s was not found in cache" % id
                    sql_ids.append(id)

            if self.use_db:
                self.db_search_(result, sql_ids)

        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)

            if key not in self:
                raise IndexError("Index %s is out of range (1 - %s)" % (key, len(self)))

            r = self.cache[key]
            if r:
            # if key in self.regions_cache_:
            #     r = self.regions_cache_[key]
            #     result.append(r)
                # self.update(key, r)
                
                return r

            sql_ids.append(key)
            if self.use_db:
                result = self.db_search_(result, sql_ids)
        else:
            raise TypeError, "Invalid argument type. Slice or int expected, %s given." % type(key)

        if len(result) == 0:
            print("!!!!  "+str(key))
            return []
        elif len(result) == 1:
            return result[0]
        else:
            return result

    def db_search_(self, result, sql_ids):
        """
        :param result: The list to which the results should be appended
        :param sql_ids: ids to be fetched from database
        """
        if not self.use_db:
            raise SyntaxError("Don't call this method when database is not used! Most likely, this is an error of region"
                              " manager itself")

        l = len(sql_ids)
        if l == 1:
            # if only one id has to be taken from db
            cmd = "SELECT data FROM regions WHERE id = '%s'" % sql_ids[0]
            self.cur.execute(cmd)

            row = self.cur.fetchone()
            self.con.commit()
            # add it to result
            id = sql_ids[0]
            try:
                region = pickle.loads(str(row[0]))
                result.append(region)
                # add it to cache
                self.add_to_cache_(id, region)
            except TypeError:
                # region was erased
                print("!!!!!!!!! REGION NOT FOUND ????   "+str(id))
                print "TypeError in region_manager.py line 272"

        if l > 1:
            cmd = "SELECT id, data FROM regions WHERE id IN %s;" % self.pretty_list(sql_ids)
            self.cur.execute("BEGIN TRANSACTION;")
            self.cur.execute(cmd)
            rows = self.cur.fetchall()
            self.con.commit()
            tmp_ids = []
            for row in rows:
                if row[0] in tmp_ids:
                    continue
                tmp_ids.append(row[0])
                region = pickle.loads(str(row[1]))
                self.add_to_cache_(row[0], region)
                result.append(region)
        return result

    def removemany_(self, regions):
        sql_ids = []
        if isinstance(regions, list):
            for r in regions:
                if isinstance(r, Region):
                    sql_ids.append(r.id())
                elif isinstance(r, (int, long)):
                    sql_ids.append(r)
                else:
                    raise TypeError("Remove method only accepts Regions or their ids (int)")
        for id_ in sql_ids:
            if id_ in self.regions_cache_:
                self.regions_cache_.pop(id_)
            if id_ in self.recent_regions_ids:
                self.recent_regions_ids.remove(id_)

        cmd = "DELETE FROM regions WHERE id IN %s" % self.pretty_list(sql_ids)
        self.cur.execute("BEGIN TRANSACTION;")
        self.cur.execute(cmd)
        self.con.commit()

    def remove(self, region):
        if isinstance(region, list):
            self.removemany_(region)
            return
        # print "Removing %s " % region
        if isinstance(region, Region):
            id_ = region.id()
        elif isinstance(region, (int, long)):
            id_ = region
        else:
            raise TypeError("Remove method only accepts Regions or their ids (int)")

        if id_ in self:
            if self.use_db:
                cmd = "DELETE FROM regions WHERE id = %s;" % id_
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.execute(cmd)
                self.con.commit()
            if id_ in self.regions_cache_:
                self.regions_cache_.pop(id_)
            if id_ in self.recent_regions_ids:
                self.recent_regions_ids.remove(id_)

    def prepare_region(self, region):
        if region.pts_rle_ == None:
            if not region.is_origin_interaction():
                region.pts_rle_ = encode_RLE(region.pts_)

        tmp_pts = region.pts_
        region.pts_ = None
        data = sql.Binary(pickle.dumps(region, -1))
        region.pts_ = tmp_pts
        return data

    def pretty_list(self, list):
        """
        Converts a list of elements [1, 2, 3] to pretty string "(1, 2, 3)"
        :param list: list to convert
        """

        l = len(list)
        param = "("
        for i in range(0, l):
            param += str(list[i])
            if i != l-1:
                param += ", "
        param += ")"
        return param

    def __len__(self):
        return self.id_

    def __contains__(self, item):
        if isinstance(item, Region):
            return len(self)+1 > item.id() > 0
        return isinstance(item, (int, long)) and len(self)+1 > item > 0


class NewRegionManager(UserList):
    def __init__(self):
        self.regions_df = None
        self.region_pts_h5 = None
        self.region_pts_h5_dataset = None
        self.is_region_pts_temp = True

        self.init_dataframe()
        self.init_h5_store()

    @classmethod
    def from_region_manager(cls, rm):
        nrm = cls()
        # n = 2000
        n = len(rm) + 1
        nrm.data = [None] * n
        for i in tqdm.tqdm(range(1, n), desc='loading regions'):
            assert rm[i].id() == i
            nrm.data[i] = rm[i]

        nrm.init_dataframe()
        nrm.init_h5_store(None, len(nrm.data))
        return nrm

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
        else:
            rm.data = []
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
    # nrm = NewRegionManager.from_region_manager(p.rm)
    # nrm.save('out')

    nrm2 = NewRegionManager.from_dir('out')

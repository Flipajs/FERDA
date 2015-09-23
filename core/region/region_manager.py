__author__ = 'flipajs'

import sqlite3 as sql
import cPickle as pickle


class RegionManager:
    def __init__(self, db_wd=None, db_name="regions.db", cache_size_limit=-1):
        # cache only mode (no db set)
        if db_wd == None:
            if cache_size_limit == -1:
                self.use_db = False
                self.regions_cache_ = {}
                self.cache_size_limit_ = cache_size_limit
            else:
                raise SyntaxError("Cache limit can only be set when database is used!")
        else:
            self.use_db = True
            self.db_path = db_wd+"/"+db_name
            print "Initializing db at %s " % self.db_path
            self.con = sql.connect(self.db_path)
            self.cur = self.con.cursor()
            self.cur.execute("CREATE TABLE IF NOT EXISTS regions(\
                id INTEGER PRIMARY KEY, \
                data BLOB);")
            self.cur.execute("CREATE INDEX IF NOT EXISTS regions_index ON log(id);")
            self.use_db = True
            self.regions_cache_ = {}
            self.cache_size_limit_ = cache_size_limit
        self.id_ = 1

        # there might be problem estimating size based on object size... then change it to cache_region_num_limit...
        # TODO: id parallelisation problems (IGNORE FOR NOW)
        # use self.id = 0, increase for each added region...
        # we will solve parallelisation by merging managers in assembly step

    def add(self, regions):
        """
        Save one or more regions in RegionManager
        :param regions: (region/list of regions) - regions that should be added into RegionManager.
        :return (int/list of ints) - ids that were given to appended regions. Regions can be later accessed via these ids
        """

        # TODO: maybe check if region is a correct object
        # TODO: create a "private" add_() method to add one region per time to cache. This method should handle the
        # TODO:     cache according to cache_size_limit_.
        if isinstance(regions, list):
            if self.use_db:
                self.tmpids = []
                self.cur.executemany("INSERT INTO regions VALUES (?, ?)", self.add_iter_(regions))
                return self.tmpids
            else:
                ids = []
                for r in regions:
                    self.regions_cache_[self.id_] = r
                    ids.append(self.id_)
                    self.id_ += 1
                return ids
        else:
            if self.use_db:
                self.cur.execute("INSERT INTO regions VALUES (?, ?)", (self.id_, sql.Binary(pickle.dumps(regions, -1))))
                self.id_ += 1
                return self.id_ - 1
            else:
                self.regions_cache_[self.id_] = regions
                self.id_ += 1
                return self.id_ - 1

    def add_iter_(self, regions):
        for r in regions:
            self.regions_cache_[self.id_] = r
            yield (self.id, sql.Binary(pickle.dumps(r, -1)))
            self.tmp_ids.append(self.id_)
            self.id_ += 1

    def __getitem__(self, key):
        if isinstance(key, slice):
            # TODO: check how this example works
            # return [self[ii] for ii in xrange(*key.indices(len(self)))]
            start = key.start
            if start == None:
                start = 0
            stop = key.stop
            if stop == None:
                stop = len(self.regions_cache_)
            step = key.step
            if step == None:
                step = 1
            result = {}
            # TODO: check if dictionary can be sliced in a better way
            # TODO: check if start, stop, step are int's in correct bounds
            # TODO: add DB check
            for i in range(start, stop, step):
                result[i] = self.regions_cache_[i]
            return result
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key in self.regions_cache_:
                return self.regions_cache_[key]
            else:
                # TODO: add DB check
                # raise IndexError, "The index (%d) is out of range." % key
                print "Key %s is not in regions_cache_" % key
                return
        raise TypeError, "Invalid argument type. Slice or int expected, %s given." % type(key)

    def __len__(self):
        # TODO: modify this when db access is implemented
        return self.id_ + 1

"""
    def add(self, regions):
        # TODO: assign ids and store regions, return ids

        for r in regions:
            if self.cache_size_limit_ < 0:
                self.regions_cache_[self.id_] = r
                r.id_ = self.id_
                self.id_ += 1

            # test existence of r.pts_rle_, if not, use encode_RLE and create...
            # when saving into DB... save everything but .pts_ into one col -> data
            # it is possible to access class in following way:
            #   d = r.__dict__ ... then make deep copy and use d.pts_ = None
            # use cPickle for data serialisation
            pass
        pass

"""

if __name__ == "__main__":
    rm = RegionManager(cache_size_limit=3)
    #rm.add("zero")
    rm.add(["zero", "one"])
    rm.add("two")
    rm.add("three")
    rm.add("four")
    rm.add("five")
    print rm[10]

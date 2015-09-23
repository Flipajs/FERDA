__author__ = 'flipajs'

import sqlite3 as sql


class RegionManager:
    def __init__(self, cache_size_limit=-1, db_name=None):
        """
        # TODO: implement db_name. It can't be optional (at least the path must be given)
        self.db_path = path+"/regions.db"
        print "Initializing db at %s " % self.db_path
        self.con = sql.connect(self.db_path)
        self.cur = self.con.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS regions(\
            id INTEGER PRIMARY KEY AUTOINCREMENT, \
            data BLOB);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS regions_index ON log(id);")
        """
        # there might be problem estimating size based on object size... then change it to cache_region_num_limit...

        # TODO: prepare cache
        # regions_cache dictionary {id: pickled data}
        self.regions_cache_ = {}
        self.cache_size_limit_ = cache_size_limit
        # cache_size_limit=-1 -> store in cache and always save into DB...
        # cache_size_limit=some_number -> store in cache as queue and always save into DB
        self.id_ = 0
        # TODO: id parallelisation problems (IGNORE FOR NOW)
        # use self.id = 0, increase for each added region...
        # we will solve parallelisation by merging managers in assembly step

        pass


    def add_(self, region):
        self.regions_cache_[self.id_] = region
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
            for i in range(start, stop, step):
                result[i] = self.regions_cache_[i]
            return result
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key in self.regions_cache_:
                return self.regions_cache_[key]
            else:
                print "Key %s is not in regions_cache_" % key
                # TODO: add DB check
                # raise IndexError, "The index (%d) is out of range." % key
        raise TypeError, "Invalid argument type. Slice or int expected, %s given." % type(key)

    def __len__(self):
        # TODO: modify this when db access is implemented
        return len(self.regions_cache_)

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
    rm = RegionManager()
    rm.add_("zero")
    rm.add_("one")
    rm.add_("two")
    rm.add_("three")
    rm.add_("four")
    rm.add_("five")
    print rm["boo"]

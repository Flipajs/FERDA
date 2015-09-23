__author__ = 'flipajs'

import sqlite3 as sql


class RegionManager:
    def __init__(self, path, cache_size_limit=-1):
        self.db_path = path+"/regions.db"
        print "Initializing db at %s " % self.db_path
        self.con = sql.connect(self.db_path)
        self.cur = self.con.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS regions(\
            id INTEGER PRIMARY KEY AUTOINCREMENT, \
            data BLOB);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS regions_index ON log(id);")
        # cache_size_limit=-1 -> store in cache and always save into DB...
        # cache_size_limit=some_number -> store in cache as queue and always save into DB
        # there might be problem estimating size based on object size... then change it to cache_region_num_limit...

        # TODO: prepare cache

        # TODO: id parallelisation problems (IGNORE FOR NOW)
        # use self.id = 0, increase for each added region...
        # we will solve parallelisation by merging managers in assembly step

        pass

    # TODO:
    # https://docs.python.org/2/reference/datamodel.html#emulating-container-types
    # slice access
    def __getitem__(self, key):
        # return regions with ids in key

        # # example:
        # if isinstance( key, slice ) :
        #     #Get the start, stop, and step from the slice
        #     return [self[ii] for ii in xrange(*key.indices(len(self)))]
        # elif isinstance( key, int ) :
        #     if key < 0 : #Handle negative indices
        #         key += len( self )
        #     if key >= len( self ) :
        #         raise IndexError, "The index (%d) is out of range."%key
        #     return self.getData(key) #Get the data from elsewhere
        # else:
        #     raise TypeError, "Invalid argument type."

        pass

    def add(self, regions):
        # TODO: assign ids and store regions, return ids

        for r in regions:
            # test existence of r.pts_rle_, if not, use encode_RLE and create...
            # when saving into DB... save everything but .pts_ into one col -> data
            # it is possible to access class in following way:
            #   d = r.__dict__ ... then make deep copy and use d.pts_ = None
            # use cPickle for data serialisation
            pass
        pass


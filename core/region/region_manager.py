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
                self.recent_regions_ids = []
                self.cache_size_limit_ = cache_size_limit
                self.id_ = 1
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
            self.cur.execute("CREATE INDEX IF NOT EXISTS regions_index ON regions(id);")
            self.use_db = True
            self.regions_cache_ = {}
            self.recent_regions_ids = []
            self.cache_size_limit_ = cache_size_limit
            # if database has been used before, get last used ID and continue from it (IDs always have to stay unique)
            try:
                self.cur.execute("SELECT id FROM regions ORDER BY id DESC LIMIT 1;")
                row = self.cur.fetchone()
                self.id_ = row[0] + 1
            except TypeError: # TypeError is raised when row is empty (no IDs were found)
                self.id_ = 1
        self.tmp_ids = []

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
                self.tmp_ids = []
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.executemany("INSERT INTO regions VALUES (?, ?)", self.add_iter_(regions))
                self.con.commit()
                return self.tmp_ids
            else:
                ids = []
                for r in regions:
                    self.add_to_cache(self.id_, r)
                    ids.append(self.id_)
                    self.id_ += 1
                return ids
        else:
            if self.use_db:
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.execute("INSERT INTO regions VALUES (?, ?)", (self.id_, sql.Binary(pickle.dumps(regions, -1))))
                self.con.commit()
                self.id_ += 1
                return self.id_ - 1
            else:
                self.add_to_cache(self.id_, regions)
                self.id_ += 1
                return self.id_ - 1

    def check_cache_size(self):
        # if size limit is used and the cache size reached it
        if self.cache_size_limit_ > 0:
            # I guess calling a method with 'while' once is better than calling a method with 'if' several times
            while len(self.regions_cache_) >= self.cache_size_limit_:
                self.regions_cache_.pop(self.recent_regions_ids.pop(0), None)

    def add_to_cache(self, id, region):
        if id in self.recent_regions_ids:
            self.recent_regions_ids.pop(0)
        self.recent_regions_ids.append(id)
        self.regions_cache_[id] = region

    def add_iter_(self, regions):
        for r in regions:
            self.add_to_cache(self.id_, r)
            yield (self.id_, sql.Binary(pickle.dumps(r, -1)))
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
                stop = len(self)
            step = key.step
            if step == None:
                step = 1
            if start < 0 or start > len(self) or stop > len(self) or stop < 0 or (stop < start and step > 0) or step == 0:
                raise ValueError("Invalid slice parameters (%s:%s:%s)" % (start, stop, step))

            result = {}
            sql = []

            # TODO: check if dictionary can be sliced in a better way
            for i in range(start, stop, step):
                if i in self.regions_cache_:
                    result[i] = self.regions_cache_[i]
                else:
                    sql.append(i)

            l = len(sql)
            if l == 1:
                cmd = "SELECT data FROM regions WHERE id = %s" % sql[0]
                self.cur.execute(cmd)
                row = self.cur.fetchone()
                result[sql[0]] = row[0]
            if l > 1:
                param = "("
                for i in range(0, l):
                    param += str(sql[i])
                    if i != l-1:
                        param += ", "
                param += ")"
                print param
                cmd = "SELECT data FROM regions WHERE id IN %s;" % param
                self.cur.execute(cmd)
                rows = self.cur.fetchall()
                i = 0
                for row in rows:
                    try:
                        print "sql "+str(sql[i])+", row "+pickle.loads(str(row[i]))
                        result[sql[i]] = pickle.loads(str(row[i]))
                        i += 1
                    except IndexError:
                        print "Error at index %s" % i
                        return



            return result
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)

            if key not in self:
                raise IndexError("Index %s is out of range (1 - %s)" % (key, len(self)))

            if key in self.regions_cache_:
                return self.regions_cache_[key]

            if self.use_db:
                print "DB CHECK"
                self.cur.execute("SELECT data FROM regions WHERE id = %s;" % key)
                row = self.cur.fetchone()
                return pickle.loads(str(row[0]))

            raise SyntaxError("Very severe! Key %s wasn't found, but probably is in RM!" % key)
        raise TypeError, "Invalid argument type. Slice or int expected, %s given." % type(key)

    def info(self):
        pass

    def __len__(self):
        return self.id_ + 1

    def __contains__(self, item):
        return isinstance(item, (int, long)) and item < len(self) and item > 0


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
    rm = RegionManager(db_wd="/home/dita", cache_size_limit=20)
    #rm.add("zero")
    rm.add(["zero", "one"])
    rm.add("two")
    rm.add("three")
    rm.add("four")
    rm.add("five")
    print rm[::]

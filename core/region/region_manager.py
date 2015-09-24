__author__ = 'flipajs'

import sqlite3 as sql
import cPickle as pickle


class RegionManager:
    def __init__(self, db_wd=None, db_name="regions.db", cache_size_limit=-1):
        """
        RegionManager is designed to store regions data. By default, all data is stored in memory cache (dictionary) and
        identified using unique ids. Optionally, database can be used, in which case the memory cache size can be
        limited to reduce memory usage.
        :param db_wd: Working directory in which the database file should be stored
        :param db_name: Database name, "regions.db" by default
        :param cache_size_limit: Number of instances to be held in cache
        :return: None
        """
        if db_wd == None:
            # cache mode (no db set)
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
            self.cur.execute("DROP TABLE IF EXISTS regions;")
            self.cur.execute("CREATE TABLE regions(\
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
        :return (list of int/ints) - ids that were given to appended regions. Regions can be later accessed via these ids
        """

        # TODO: check if region is a correct object

        # use one return list for simplicity
        self.tmp_ids = []

        if isinstance(regions, list):
            if self.use_db:
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.executemany("INSERT INTO regions VALUES (?, ?)", self.add_iter_(regions))
                # self.id and self.tmp_ids are updated in self.add_iter_
                self.con.commit()
            else:
                for r in regions:
                    self.add_to_cache_(self.id_, r)
                    self.tmp_ids.append(self.id_)
                    self.id_ += 1
        else:
            if self.use_db:
                self.add_to_cache_(self.id_, regions)
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.execute("INSERT INTO regions VALUES (?, ?)", (self.id_, sql.Binary(pickle.dumps(regions, -1))))
                self.con.commit()
                self.tmp_ids.append(self.id_)
                self.id_ += 1
            else:
                self.add_to_cache_(self.id_, regions)
                self.tmp_ids.append(self.id_)
                self.id_ += 1

        return self.tmp_ids

    def add_to_cache_(self, id, region):
        """
        This method adds region with id to the cache. It also updates it's position in recent_regions_ids and checks
        the cache size.
        :param id:
        :param region:
        :return None
        """
        # print "Adding %s %s" % (id, region)
        # print "Cache: %s" % self.recent_regions_ids
        if id in self.recent_regions_ids:
            # remove region from recent_regions_ids
            self.recent_regions_ids.pop(0)
            # print "Moving %s up" % id
        # add region to fresh position in recent_regions_ids and add it to cache
        self.recent_regions_ids.append(id)
        self.regions_cache_[id] = region

        if self.cache_size_limit_ > 0 and len(self.regions_cache_) > self.cache_size_limit_:
            pop_id = self.recent_regions_ids.pop(0)
            self.regions_cache_.pop(pop_id, None)
            # print "Cache limit (%s) reached, popping id %s" % (self.cache_size_limit_, pop_id)

    def update(self, key):
        """
        Renew the position of key in recent_regions_ids
        :param key: key of the region
        :return: None
        """

        # remove region from recent_regions_ids
        if key in self.recent_regions_ids:
            self.recent_regions_ids.pop(0)
            # add region to fresh position in recent_regions_ids and add it to cache
            self.recent_regions_ids.append(key)
        else:
            raise KeyError("Key %s is not in cache and can't be updated!" % key)

    def add_iter_(self, regions):
        """
        Iterator over given regions, yields a tuple used for sql executemany. self.tmp_ids and self.id_ get modified.
        :return tuple (id, binary region data)
        """
        for r in regions:
            self.add_to_cache_(self.id_, r)
            yield (self.id_, sql.Binary(pickle.dumps(r, -1)))
            self.tmp_ids.append(self.id_)
            self.id_ += 1

    def __getitem__(self, key):
        sql_ids = []
        result = {}
        if isinstance(key, slice):
            # TODO: check how this example works and if it can be used
            # return [self[ii] for ii in xrange(*key.indices(len(self)))]
            # get values from slice
            start = key.start
            if start == None or start == 0:
                start = 1
            stop = key.stop
            # TODO: value of "stop" is 9223372036854775807 when using [:], but it works fine with [::]. Find out why.
            if stop == None or stop == 9223372036854775807:
                stop = len(self)
            step = key.step
            if step == None:
                step = 1
            # check if slice parameters are ok
            if start < 0 or start > len(self) or stop > len(self) or stop < 0 or (stop < start and step > 0) or step == 0:
                raise ValueError("Invalid slice parameters (%s:%s:%s)" % (start, stop, step))

            # go through slice
            for i in range(start, stop, step):
                if i in self.regions_cache_:
                    # use cache if region is available
                    result[i] = self.regions_cache_[i]
                    self.update(i)
                else:
                    # if not, add id to the list of ids to be fetched from db
                    sql_ids.append(i)
            if self.use_db:
                self.db_search_(result, sql_ids)
            return result

        if isinstance(key, list):
            for id in key:
                if not isinstance(id, int):
                    print "TypeError: int expected, %s given! Skipping key '%s'." % (type(id), id)
                    continue
                if id in self.regions_cache_:
                    # print "%s was found in cache" % id
                    result[id] = self.regions_cache_[id]
                    self.update(id)
                else:
                    # print "%s was not found in cache" % id
                    sql_ids.append(id)

            if self.use_db:
                self.db_search_(result, sql_ids)
            return result

        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)

            if key not in self:
                raise IndexError("Index %s is out of range (1 - %s)" % (key, len(self)))

            if key in self.regions_cache_:
                result[key] = self.regions_cache_[key]
                self.update(key)
                return result

            sql_ids.append(key)
            if self.use_db:
                self.db_search_(result, sql_ids)
            return result

            raise SyntaxError("Very severe! Key %s wasn't found, but probably is in RM!" % key)
        raise TypeError, "Invalid argument type. Slice or int expected, %s given." % type(key)

    def db_search_(self, result, sql_ids):
        """
        :param result: The dictionary to which the results should be appended
        :param sql_ids: ids to be fetched from database
        """
        if not self.use_db:
            raise SyntaxError("Don't call this method when database is not used! Most likely, this is an error of region"
                              " manager itself")

        l = len(sql_ids)
        if l == 1:
            # if only one id has to be taken from db
            cmd = "SELECT data FROM regions WHERE id = %s" % sql_ids[0]
            self.cur.execute(cmd)
            row = self.cur.fetchone()
            # add it to result
            id = sql_ids[0]
            region = pickle.loads(str(row[0]))
            result[id] = region
            # add it to cache
            self.add_to_cache_(id, region)

        if l > 1:
            cmd = "SELECT data FROM regions WHERE id IN %s;" % self.pretty_list(sql_ids)
            self.cur.execute(cmd)
            rows = self.cur.fetchall()
            i = 0
            for row in rows:
                id = sql_ids[i]
                region = pickle.loads(str(row[0]))
                self.add_to_cache_(id, region)
                result[id] = region
                i += 1

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
        return isinstance(item, (int, long)) and len(self) > item > 0


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
    rm = RegionManager(db_wd="/home/dita", cache_size_limit=4)
    # rm = RegionManager()
    rm.add(["one", "two"])
    rm.add("three")
    rm.add(["four", "five", "six"])
    rm.add(["seven", "eight", "nine"])
    print "Cache: %s" % rm.recent_regions_ids

    print "All: %s" % rm[:]
    print "Cache: %s" % rm.recent_regions_ids

    ids = [1, 4]
    print "List: %s" % rm[ids]
    print "Cache: %s" % rm.recent_regions_ids

    print "3: %s" % rm[3]
    print "Cache: %s" % rm.recent_regions_ids

    print "9: %s" % rm[9]
    print "Cache: %s" % rm.recent_regions_ids

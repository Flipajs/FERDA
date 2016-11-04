import sqlite3 as sql
import cPickle as pickle
import random

DEBUG = True
__author__ = 'dita'


class FeatureManager:
    def __init__(self, db_wd=None, db_name="fm.sqlite3", cache_size_limit=1000, data=None):
        """
        RegionManager is designed to store regions data. By default, all data is stored in memory cache (dictionary) and
        identified using unique ids. Optionally, database can be used, in which case the memory cache size can be
        limited to reduce memory usage.
        :param db_wd: Working directory in which the database file should be stored
        :param db_name: Database name, "fm.sqlite3" by default
        :param cache_size_limit: Number of instances to be held in cache
        :return: None
        """

        if db_wd == None:
            # cache mode (no db set)
            if cache_size_limit == -1:
                self.use_db = False
                self.features_cache_ = {}
                self.recent_feature_ids = []
                self.cache_size_limit_ = cache_size_limit
            else:
                raise SyntaxError("Cache limit can only be set when database is used!")
        else:
            self.use_db = True
            self.db_path = db_wd + "/" + db_name
            print "Initializing db at %s " % self.db_path
            self.con = sql.connect(self.db_path)
            self.cur = self.con.cursor()
            # DEBUG, do not use!
            if DEBUG:
                self.cur.execute("DROP TABLE IF EXISTS features;")

            self.cur.execute("CREATE TABLE IF NOT EXISTS features(\
                    id INTEGER PRIMARY KEY, \
                    data BLOB);")
            self.cur.execute("CREATE INDEX IF NOT EXISTS features_index ON features(id);")

            self.use_db = True
            self.features_cache_ = {}
            self.recent_feature_ids = []
            self.cache_size_limit_ = cache_size_limit
        self.tmp_ids = []

        if isinstance(data, FeatureManager):
            # TODO: Fix copying from previous Feature Managers
            newdata = data[:]
            self.add(newdata)
        elif isinstance(data, list):
            # TODO: Fix initial data import
            for datas in data:
                if isinstance(datas, FeatureManager):
                    self.add(datas[:])

    def add(self, f_id, feature):
        """
        Save one or more features in FeatureManager
        :param f_id: (id/list of ids) - id of features that should be added into FeatureManager.
        :param feature: (feature/list of features) - features that should be added into FeatureManager.
        """

        # use one return list for simplicity - the ids are appended to it from the get_next_id function
        self.tmp_ids = []

        if isinstance(f_id, list):
            # check that there is a feature for each id (valid input)
            if not isinstance(feature, list) or len(feature) != len(f_id):
                raise ValueError("If a list of ids is provided, there must be a matching list of features to use.")
            if self.use_db:
                # insert into db - use iterative insert, then commit (end transaction)
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.executemany("INSERT INTO features VALUES (?, ?)", self.add_iter_(f_id, feature))
                self.con.commit()
                return
            else:
                # use only cache - loop all data and insert individually
                for i, f in zip(f_id, feature):
                    self.add_to_cache_(i, f)
                return

        if not isinstance(f_id, (long, int)):
            raise TypeError("IDs can be given as list of ints or int/long, not %s" % type(f_id))

        # self.add_to_cache_(id, regions)
        # TODO: why is this commented?
        if self.use_db:
            self.cur.execute("BEGIN TRANSACTION;")
            self.cur.execute("INSERT INTO regions VALUES (?, ?)", (f_id, pickle_data(feature)))
            self.con.commit()

        return self.tmp_ids

    def clear_cache(self):
        self.features_cache_ = {}
        self.recent_feature_ids = []

    def add_to_cache_(self, id_, region):
        """
        This method adds region with id to the cache. It also updates it's position in recent_regions_ids and checks
        the cache size.
        :param id_:
        :param region:
        :return None
        """
        # print "Adding %s %s" % (id_, region)
        # print "Cache: %s" % self.recent_regions_ids
        if id_ in self.recent_feature_ids:
            # remove region from recent_regions_ids
            self.recent_feature_ids.remove(id_)
            # print "Moving %s up" % id_
            # add region to fresh position in recent_regions_ids and add it to cache
            self.recent_feature_ids.append(id_)
            self.features_cache_[id_] = region
        else:
            # add region to fresh position in recent_regions_ids and add it to cache
            self.recent_feature_ids.append(id_)
            self.features_cache_[id_] = region

            if 0 < self.cache_size_limit_ < len(self.features_cache_):
                pop_id = self.recent_feature_ids.pop(0)
                self.features_cache_.pop(pop_id, None)
                # print "Cache limit (%s) reached, popping id %s" % (self.cache_size_limit_, pop_id)

    def update(self, key, feature):
        """
        Renew the position of key in recent_regions_ids
        :param key: key of the feature
        :param feature: feature data
        :return: None
        """

        # remove region from recent_regions_ids
        if key in self.recent_feature_ids:
            self.recent_feature_ids.remove(key)
            # add region to fresh position in recent_regions_ids and add it to cache
            self.recent_feature_ids.append(key)
        else:
            self.add_to_cache_(key, feature)

    def add_iter_(self, ids, data):
        """
        Iterator over given regions, yields a tuple used for sql executemany. Self.tmp_ids  get modified.
        :return tuple (id, binary region data)
        """
        for id_, d in zip(ids, data):
            if not isinstance(d, tuple):
                self.con.rollback()
                raise TypeError("Feature manager can only work with tuple objects, not %s" % type(d))

            if self.cache_size_limit_ != 0:
                self.add_to_cache_(id_, d)

            yield (id_, pickle_data(d))

    def __getitem__(self, key):
        sql_ids = []
        result_ = []
        if isinstance(key, slice):
            # TODO: check how this example works and if it can be used
            # return [self[ii] for ii in xrange(*key.indices(len(self)))]
            # get values from slice
            start = key.start
            stop = key.stop
            step = key.step
            if start is None or start == 0:
                start = 1
            # value of "stop" is 9223372036854775807 when using [:] and None with [::]. Hotfixed.
            if stop is None or stop == 9223372036854775807:
                raise ValueError("Invalid slice parameters (%s:%s:%s), this can be caused by calling [:] or [::]."
                                 "Please use list of ids to fix this." % (start, stop, step))
            # stop = len(self) + 1
            if step is None:
                step = 1
            # check if slice parameters are ok
            if (start < 0 or stop < 0 or (stop < start and step > 0) or step == 0) and not DEBUG:
                raise ValueError("Invalid slice parameters (%s:%s:%s)" % (start, stop, step))

            # go through slice
            count = len(range(start, stop, step))
            result_ = [None] * count
            pos = {}
            k = 0
            for i in range(start, stop, step):
                pos[i] = k
                k += 1
                if i in self.features_cache_:
                    # print "----%s is in cache" % i
                    # use cache if region is available
                    r = self.features_cache_[i]
                    result_[pos[i]] = r
                    self.update(i, r)
                else:
                    # print "----%s is not in cache" % i
                    # if not, add id to the list of ids to be fetched from db
                    sql_ids.append(i)
            if self.use_db:
                self.db_search_(result_, sql_ids, pos)

        elif isinstance(key, list):
            count = len(key)
            result_ = [None] * count
            pos = {}
            k = 0
            for id in key:
                pos[id] = k
                k += 1
                if not isinstance(id, int):
                    print "TypeError: int expected, %s given! Skipping key '%s'." % (type(id), id)
                    continue
                if id in self.features_cache_:
                    # print "%s was found in cache" % id
                    r = self.features_cache_[id]
                    result_[pos[id]] = r
                    self.update(id, r)
                else:
                    # print "%s was not found in cache" % id
                    sql_ids.append(id)

            if self.use_db:
                self.db_search_(result_, sql_ids, pos)

        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)

            if key in self.features_cache_:
                r = self.features_cache_[key]
                result_[0] = r
                self.update(key, r)

                return r

            sql_ids.append(key)
            result_ = [None] * 1
            pos = {key: 0}
            if self.use_db:
                self.db_search_(result_, sql_ids, pos)
        else:
            raise TypeError, "Invalid argument type. Slice or int expected, %s given." % type(key)

        if not result_ or len(result_) == 0:
            if DEBUG:
                print "[!] FM: Key %d not found." % key
            return []
        elif len(result_) == 1:
            return result_[0]
        else:
            return result_

    def db_search_(self, result_, sql_ids, pos):
        """
        :param result: The list to which the results should be appended
        :param sql_ids: ids to be fetched from database
        """
        if not self.use_db:
            raise SyntaxError(
                "Don't call this method when database is not used! Most likely, this is an error of feature"
                " manager itself")

        l = len(sql_ids)
        if l == 1:
            # if only one id has to be taken from db
            cmd = "SELECT data FROM features WHERE id = '%s'" % sql_ids[0]
            self.cur.execute(cmd)

            row = self.cur.fetchone()
            self.con.commit()
            # add it to result
            id_ = sql_ids[0]
            try:
                data = pickle.loads(str(row[0]))
                result_[pos[id_]] = data
                # add it to cache
                self.add_to_cache_(id_, data)
            except TypeError:
                return None

        if l > 1:
            cmd = "SELECT id, data FROM features WHERE id IN %s;" % pretty_list(sql_ids)
            self.cur.execute("BEGIN TRANSACTION;")
            self.cur.execute(cmd)
            rows = self.cur.fetchall()
            self.con.commit()
            tmp_ids = []
            i = 0
            for row in rows:
                if row[0] in tmp_ids:
                    continue
                tmp_ids.append(row[0])
                data = pickle.loads(str(row[1]))
                self.add_to_cache_(row[0], data)
                result_[pos[row[0]]] = data
                i += 1
        return result_

    def removemany_(self, regions):
        sql_ids = []
        if isinstance(regions, list):
            for r in regions:
                if isinstance(r, (int, long)):
                    sql_ids.append(r)
                else:
                    raise TypeError("Remove method can only work with tuple objects, not %s" % type(r))
        for id_ in sql_ids:
            if id_ in self.features_cache_:
                self.features_cache_.pop(id_)
            if id_ in self.recent_feature_ids:
                self.recent_feature_ids.remove(id_)

        cmd = "DELETE FROM features WHERE id IN %s" % pretty_list(sql_ids)
        self.cur.execute("BEGIN TRANSACTION;")
        self.cur.execute(cmd)
        self.con.commit()

    def remove(self, ids):
        if isinstance(ids, list):
            self.removemany_(ids)
            return
        elif isinstance(ids, (int, long)):
            id_ = ids
            if self.use_db:
                cmd = "DELETE FROM features WHERE id = %s;" % id_
                self.cur.execute("BEGIN TRANSACTION;")
                self.cur.execute(cmd)
                self.con.commit()
            if id_ in self.features_cache_:
                self.features_cache_.pop(id_)
            if id_ in self.recent_feature_ids:
                self.recent_feature_ids.remove(id_)
        else:
            raise TypeError("Remove method only accepts an integer (long) ids and their lists, %s given" % type(ids))


def pickle_data(data):
    """ Convert data object to sql Binary object using pickle."""
    return sql.Binary(pickle.dumps(data, -1))


def pretty_list(list):
    """
    Converts a list of elements [1, 2, 3] to pretty string "(1, 2, 3)"
    :param list: list to convert
    """

    l = len(list)
    param = "("
    for i in range(0, l):
        param += str(list[i])
        if i != l - 1:
            param += ", "
    param += ")"
    return param


def get_rnd_tuple(size=10, min=0, max=100):
    rand_list = []
    for i in range(random.randint(1, size)):
        rand_list.append(random.randint(min, max))
    return tuple(rand_list)


if __name__ == "__main__":

    ids = [1, 2, 5, 200, 38, 14, 3, 151, 16, 18, 17, 98, 71, 4]
    data = [(17, 50, 84, 26, 79, 22, 16),
            (30, 55, 44, 58, 9, 0, 83, 28, 9, 63),
            (78, 22, 67, 87, 95, 80, 8),
            (45, 39, 9, 33, 26, 85, 93, 2, 36, 72),
            (22, 80, 83, 86, 65, 69, 0, 98, 55),
            (85, 68, 45),
            (62, 7, 43, 81, 59, 30, 31, 57, 2),
            (66, 100, 12, 59, 62, 31),
            (11, 68),
            (55, 49),
            (87, 49, 19, 79, 45, 79, 63, 73),
            (74,),
            (81, 69, 66, 58),
            (58, 94, 54, 75, 32, 77, 90, 16)]

    rm = FeatureManager(db_wd="/home/dita", cache_size_limit=1)
    rm.add(ids, data)

    print "1-5: ", rm[1:5]
    print "1: ", rm[1]
    print "2: ", rm[2]
    print "3: ", rm[3]
    print "4: ", rm[4]
    print "5: ", rm[5]

    rm.remove(2)
    rm.remove(3)

    print "1-5: ", rm[1:5]

    #print rm[16:19]
    #print rm[20]

    # db size with 20 pts regions: 306 176 bytes
    # db size with 20 rle regions:  75 776 bytes
    # NOTE: To check db size properly, always start in new file. File size doesn't decrease when items are deleted or
    #       when table is dropped. Instead of delete, sql VACUUM command can be used.

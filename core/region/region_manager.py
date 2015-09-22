__author__ = 'flipajs'


class RegionManager:
    def __init__(self, cache_size_limit=-1, db_name=None):
        # cache_size_limit=-1 -> store in cache and always save into DB...
        # cache_size_limit=some_number -> store in cache as queue and always save into DB
        # there might be problem estimating size based on object size... then change it to cache_region_num_limit...

        # region DB should be:
        # ID | data

        # TODO: initialize database...
        # TODO: prepare cache

        self.regions_cache_ = {}
        self.cache_size_limit_ = cache_size_limit
        self.id_ = 1

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
        if isinstance(key, slice):
            pass
        # #Get the start, stop, and step from the slice
        #     return [self[ii] for ii in xrange(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            # + 1 because ids starts from 1
            if key >= len(self.regions_cache_) + 1:
                raise IndexError, "The index (%d) is out of range." % key

            if key in self.regions_cache_:
                return self.regions_cache_[key]
            else:
                # db query...
                pass
        # else:
        #     raise TypeError, "Invalid argument type."

        pass

    def add(self, regions):
        # TODO: assign ids (region.id = id...) and store regions,

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

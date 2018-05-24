import numpy as np
try:
    import pandas as pd
except ImportError as e:
    pass


class ObjectsArray(object):
    """
    Companion object for ndarrays containing properties of multiple objects.

    ObjectsArray cares for column indexing using object index and property name.

    Example: predictions.col2idx(1, 'x') returns column index of 1st object and property 'x'.

    """
    def __init__(self, properties, n):
        if isinstance(properties, set):
            self.properties = sorted(list(properties))
        elif isinstance(properties, (list, tuple)):
            self.properties = properties
        else:
            assert False, 'properties argument has to be set, list or tuple'
        self.n = n

    def prop2idx(self, object_index, prop):
        if not isinstance(prop, (list, tuple)):
            prop_ = (prop,)
        else:
            prop_ = prop
        for p in prop_:
            assert p in self.properties, 'invalid property name'
        idxs = [object_index * len(self.properties) + self.properties.index(p) for p in prop_]
        if not isinstance(prop, (list, tuple)):
            return idxs[0]
        else:
            return idxs

    def prop2idx_(self, object_index, prop):
        return slice(self.prop2idx(object_index, prop), self.prop2idx(object_index, prop) + 1)

    def columns(self):
        names = []
        for i in range(self.n):
            names.extend(['%d_%s' % (i, c) for c in self.properties])
        return names

    def num_columns(self):
        return len(self.properties) * self.n

    def array_to_struct(self, array):
        formats = self.num_columns() * 'f,'
        return np.core.records.fromarrays(array.transpose(), names=', '.join(self.columns()), formats=formats)

    def array_to_dict(self, array):
        out = {}
        for i, col in enumerate(self.columns()):
            out[col] = array[:, i]
            if len(out[col]) == 1:
                out[col] = out[col].item()
        return out

    def array_to_dataframe(self, array):
        return pd.DataFrame(array, columns=self.columns())

    def dataframe_to_array(self, df):
        return df[self.columns()].values

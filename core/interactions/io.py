from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div
import re
import pandas as pd


def read_gt(filename):
    regexp = re.compile('(\d*)_(\w*)')
    df = pd.read_csv(filename)
    ids = set()
    properties = []
    properties_no_prefix = []
    for col in df.columns:
        match = regexp.match(col)
        if match is not None:
            ids.add(int(match.group(1)))
            properties.append(match.group(2))
        else:
            properties_no_prefix.append(col)
    if not properties:
        # not prefixed columns, assuming single object
        n = 1
        properties = properties_no_prefix
        ids.add(0)
        df.columns = ['0_{}'.format(col) for col in df.columns]
    else:
        n = len(ids)
    assert min(ids) == 0 and max(ids) == n - 1, 'object describing columns have to be prefixed with numbers starting with 0'
    assert len(properties) % n == 0
    properties = properties[:(old_div(len(properties), n))]  # only properties for object 0
    return n, properties, df
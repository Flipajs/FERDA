from __future__ import print_function

class PairedDict:
    """
    Class behaving as dictionary but with bot direction search.
     e.g
     p = PairedDict()
     p[0] = 'FERDA'
     p[1] = 'FRANTA'
     p[2] = 'EMIL'

     print p['FRANTA'], p[0] # prints 1, FERDA

     Works only if len(union(keys, vals)) == len(keys) + len(vals)
    """

    def __init__(self):
        self.d1_ = {}
        self.d2_ = {}

    def __setitem__(self, key, value):
        if key in self.d2_:
            raise Exception('key is already present as a value')

        if value in self.d1_:
            raise Exception('value is already present as a key')

        self.d1_[key] = value
        self.d2_[value] = key

    def __delitem__(self, key):
        val = self.d1_[key]
        del self.d2[val]
        del self.d1_[key]

    def __getitem__(self, key):
        if key in self.d1_:
            return self.d1_[key]

        if key in self.d2_:
            return self.d2_[key]

        raise

    def __len__(self):
        return len(self.d1_)

    def __str__(self):
        return self.d1_.__str__()

if __name__ == '__main__':
    p = PairedDict()
    p[0] = 'FERDA'
    p[1] = 'FRANTA'
    p[2] = 'EMIL'

    print(p['FRANTA'], p[0])
    print(len(p))
    print(p)
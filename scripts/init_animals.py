from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import range
from core.animal import Animal, colors_
from core.project.project import Project
import pickle as pickle


if __name__ == '__main__':
    wd = '/Users/flipajs/Documents/wd/FERDA/Zebrafish_playground'
    num = 5

    p = Project()
    p.load(wd)
    # p.GT_file = '/Users/flipajs/Documents/dev/ferda/data/GT/Camera3.pkl'
    # p.save()

    animals = []
    for id_ in range(num):
        c = colors_[id_]
        c = (c[0], c[2], c[1])
        animals.append(Animal(id_, color=colors_[id_]))

    with open(wd+'/animals.pkl', 'wb') as f:
        pickle.dump(animals, f)


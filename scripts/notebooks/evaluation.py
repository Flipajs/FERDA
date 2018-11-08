from __future__ import print_function
from __future__ import unicode_literals
from builtins import zip
from core.project.project import Project


def test_equals(id1, id2):
    if id1 > -1 and id2 > -1:
        if id1 != id2:
            return False

    return True


if __name__ == "__main__":
    MATCHING_FRAME = 4385

    wd1 = '/Users/flipajs/Documents/wd/zebrafish0'
    wd2 = '/Users/flipajs/Documents/wd/zebrafish4385'
    wd3 = '/Users/flipajs/Documents/wd/zebrafish14206'
    wds = [wd1, wd2, wd3]

    projects = []

    for wd in wds:
        p = Project()
        p.load(wd)

        projects.append(p)

    tracklets = projects[0].chm.tracklets_in_frame(MATCHING_FRAME)

    matchings = []

    ids_ = {}
    ch_ids_2_ids = {}
    for t in tracklets:
        id_ = list(t.P)[0]
        ids_[id_] = id_
        ch_ids_2_ids[t.id()] = id_

    matchings.append(ids_)

    for p in projects[1:]:
        tracklets = p.chm.tracklets_in_frame(MATCHING_FRAME)

        ids_ = {}
        for t in tracklets:
            id_ = list(t.P)[0]
            ids_[id_] = ch_ids_2_ids[t.id()]

        matchings.append(ids_)


    
    for ch1, ch2, ch3 in zip(projects[0].chm.chunk_gen(), projects[1].chm.chunk_gen(), projects[2].chm.chunk_gen()):
        ch1_id_ = -1
        ch2_id_ = -1
        ch3_id_ = -1

        if len(ch1.P):
            ch1_id_ = list(ch1.P)[0]
        if len(ch2.P):
            ch2_id_ = matchings[1][list(ch2.P)[0]]
        if len(ch3.P):
            ch3_id_ = matchings[2][list(ch3.P)[0]]

        b1 = test_equals(ch1_id_, ch2_id_)
        b2 = test_equals(ch1_id_, ch3_id_)
        b3 = test_equals(ch2_id_, ch3_id_)

        if not(b1 and b2 and b3):
            print("MISTAKE ", ch1, ch2, ch3, ch1_id_, ch2_id_, ch3_id_)

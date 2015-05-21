__author__ = 'fnaiser'
import sys
import math
from core.region.mser import get_msers_
from utils.video_manager import get_auto_video_manager
import cPickle as pickle

if __name__ == '__main__':
    vid_path = sys.argv[1]
    frame = int(sys.argv[2])
    frames_in_row = int(sys.argv[3])
    vid = get_auto_video_manager(vid_path)


    img = vid.seek_frame(frame*frames_in_row)

    sum_ = 0

    # regions = {}
    #
    for i in range(frames_in_row):
        m = get_msers_(img)
        # regions[frame*frames_in_row + i] = m

        img = vid.move2_next()
        sum_ += len(m)

        # with open('/Volumes/Seagate Expansion Drive/working_dir/eight1/msers/'+str(frame*frames_in_row+i)+'.pkl', 'wb') as f:
        #     pickle.dump(m, f)

    print frame, " : ", sum_, " msers detected"
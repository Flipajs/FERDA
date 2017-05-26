from utils.video_manager import get_auto_video_manager
from core.project.project import Project
import os
import cv2

wds = [
    # '/Users/flipajs/Documents/wd/FERDA/Cam1_playground',
       '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground']



for wd in wds:
    p = Project()
    p.load(wd)

    os.mkdir(wd+'/vid_dump')

    vid = get_auto_video_manager(p)
    img = vid.next_frame()

    i = 0
    while img is not None:
        si = str(i)
        while len(si) < 5:
            si = '0' + si

        cv2.imwrite(wd+'/vid_dump/'+si+'.png', img)

        img = vid.next_frame()
        i += 1

import cv2
import numpy as np

crf0 = cv2.VideoCapture('/Volumes/Transcend/old_dropbox/FERDA/Cam1_clip_crf0.mp4')
crf15 = cv2.VideoCapture('/Volumes/Transcend/old_dropbox/FERDA/Cam1_clip_crf15.mp4')
crf17 = cv2.VideoCapture('/Volumes/Transcend/old_dropbox/FERDA/Cam1_clip_crf17.mp4')
crf18 = cv2.VideoCapture('/Volumes/Transcend/old_dropbox/FERDA/Cam1_clip_crf18.mp4')
crf23 = cv2.VideoCapture('/Volumes/Transcend/old_dropbox/FERDA/Cam1_clip.mp4')

videos = [crf0, crf15, crf17, crf18, crf23]

frames = []
for vid in videos:
    _, frame = vid.read()
    frames.append(frame)


for frame, crf in zip(frames[1:], [15, 17, 18, 22]):
    frame_orig = np.asarray(frames[0], dtype=np.integer)
    fraem = np.asarray(frame, dtype=np.integer)

    mean_diff = np.mean(frame-frame_orig)
    std_diff = np.std(frame-frame_orig)
    mean_abs_diff = np.mean(np.abs(frame-frame_orig))
    std_abs_diff = np.std(np.abs(frame-frame_orig))
    max_diff = np.max(np.abs(frame - frame_orig))
    total_diff = np.sum(np.abs(frame - frame_orig))

    print("CRF: {}, mean diff: {:.5f}, std_diff: {:.3f}, mean abs diff: {:.3}, std abs diff: {:.3f} max diff: {}, total: {}".format(
        crf, mean_diff, std_diff, mean_abs_diff, std_abs_diff, max_diff, total_diff
    ))
    # print mean_diff, std_diff, mean_abs_diff, max_diff

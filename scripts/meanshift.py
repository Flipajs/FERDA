import numpy as np
import cv2

import matplotlib.pyplot as plt

def run_main():
    cap = cv2.VideoCapture('/Users/flipajs/Documents/wd/C210min.avi')

    # Read the first frame of the video
    ret, frame = cap.read()

    # Set the ROI (Region of Interest). Actually, this is a
    # rectangle of the building that we're tracking
    c,r,w,h = 100,500,80,80
    track_window = (c,r,w,h)

    # cv2.rectangle(frame, (c, r), (c+w, r+h), (255, 0, 0), 3)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)


    # Create mask and normalized histogram
    roi = frame[r:r+h, c:c+w]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # cv2.imshow('roi', hsv_roi)
    # cv2.waitKey(0)
    # mask = cv2.inRange(roi[:,:,0], 0, 80)
    mask = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # plt.figure(1)
    # plt.imshow(mask)
    #
    # plt.figure(2)
    # plt.hist(roi_hist)
    # plt.show()

    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)

    while True:
        ret, frame = cap.read()

        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = frame

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x,y,w,h = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        cv2.putText(frame, 'Tracked', (x-25,y-10), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255,255,255), 2, cv2.CV_AA)

        cv2.imshow('Tracking', frame)

        cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
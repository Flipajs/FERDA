import mser_operations
import experiment_params
import cv2
import visualize


p = experiment_params.Params()
m = mser_operations.MserOperations(p)
im = cv2.imread('/home/flipajs/~dump/eight/209/frame.png')

regs = m.process_image(im)
for r in regs[0]:
	print r['parent_label']


coll = visualize.draw_region_collection(im, regs[0], p)

cv2.imshow("collection", coll)
cv2.imshow("frame", im)
cv2.waitKey(0)

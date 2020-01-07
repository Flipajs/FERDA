import cv2

print(cv2.getBuildInformation())

cap = cv2.VideoCapture()
cap.open('/datagrid/ferda/data/youtube/Sowbug3_cut.mp4')
ret, img = cap.read()
assert ret
print('img.shape: {}'.format(img.shape))

import graph_tool
g = graph_tool.Graph()
print(g)

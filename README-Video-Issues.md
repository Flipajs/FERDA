#Known Issues and encoding suggestions
OpenCV(3.4.1) /tmp/opencv-20180307-60086-ryy1b3/opencv-3.4.1/modules/videoio/src/container_avi.cpp:514: error: (-215) chunk.m_size <= 0xFFFF in function readFrame

This happens when using .avi file with MPEG codec.

Solved with encoding video with h264:
```bash
-i video.avi -vcodec libx264 -preset slow -crf 15 new_video.mp4
```

What needs to be investigated it a speed of different codecs. Mainly the speed of frame seek. To me it seems that h264 is slightly slower but didn’t measured it. Pros of h264 is that it is widely adapted.

The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible. A lower value generally leads to higher quality, and a subjectively sane range is 17–28. Consider 17 or 18 to be visually lossless or nearly so; it should look the same or nearly the same as the input but it isn't technically lossless.

-preset slow is not necessary.

**-crf 17 is in experience quite good setting which we recommends.**

-preset slow seems to be 2x times slowe then default

comparison of various settings on Cam1_clip 5min, 4500 frames. (first row is the original video captured from camera)

codec | -crf | -preset  | size \[MB\] | note
---: | ----: | ----: | ---: | ---
MPEG | - | - | 452,8 | 
h264 | 0 | - | 1680,0 | 
h264 | 15| slow |      276,9 | 
h264 | 17| slow  |    194,4   | 
h264 | 17| -      |                     199,5 |   
h264 | 17| -       |                    160,6| 
h264 | 23 (default) |  - |             37,9 | (the loose of quality if observable by naked eye)
  


(comparison generated using FERDA/scripts/compare_compressions.py be2b7e1)
When compared to CRF 0

CRF:  | mean diff:  | std_diff:  | mean abs diff:  | std abs diff:  | max diff:  | total: 
 ---  | --- | --- | --- | --- | --- | --- 
15 | 0.02070 | 0.753 | 0.387 | 0.647 | 12 | 1217460
17 | 0.02094 | 0.873 | 0.486 | 0.725 | 16 | 1528283
18 | 0.02873 | 0.998 | 0.605 | 0.794 | 15 | 1903443
22 | 0.07024 | 1.451 | 1.01 | 1.047 | 23 | 3166933


When compared to CRF 15 (original is unfortunately not working in OpenCV

CRF:  | mean diff:  | std_diff:  | mean abs diff:  | std abs diff:  | max diff:  | total: 
 ---  | --- | --- | --- | --- | --- | --- 
17 | 0.00024 | 0.908 | 0.534 | 0.734 | 16 | 1678973
18 | 0.00803 | 1.034 | 0.647 | 0.806 | 18 | 2034083
22 | 0.04954 | 1.443 | 1.0 | 1.040 | 20 | 3151839

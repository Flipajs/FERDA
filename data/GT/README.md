# Ground Truth
*Files can be loaded using python pickle module.*

```python
import cPickle as pickle

with open('path/to/gt.pkl') as f:
    gt = pickle.load(f)
    
gt[frame][0]  # is tuple (y, x)
gt[100][2]    # returns (y, x) of id 2 in frame 100

## IMPORTANT!
if gt[frame][ID][0] < 0 or frame not in gt or ID not in gt[frame]:
    # GT for given frame and ID is undefined.
```
| name | video | what | length | note |
| --- | --- | --- | --- | --- |
| __Cam1_sparse.pkl__ | ([Cam1_clip.avi](https://www.dropbox.com/s/afrbhqgucl27xy2/Cam1_clip.avi?dl=0)) | 6 ants with colormarks | 5 minutes | |
| __Cam2_sparse.pkl__ | ([Cam1_clip.avi](https://www.dropbox.com/s/qymo8x3y8af0muv/Cam2_clip.avi?dl=0)) | 6 ants with colormarks | 5 minutes | |

## New GT and Results Format

Defined at https://motchallenge.net/. CSV file with `frame, id, x, y, width, height, confidence` columns.

.pkl files coverted to .txt using:

    python -m utils.gt.pkltomot data/GT/5Zebrafish_nocover_22min.pkl data/GT/5Zebrafish_nocover_22min.txt

## Files Renamed

New filenames match the video filenames.

```
Cam1_.pkl -> Cam1_clip.avi.pkl
Cam1_sparse.pkl ->  Cam1_clip.avi_sparse.pkl
Cam2_sparse.pkl Cam2_clip.avi_sparse.pkl
Camera3.pkl -> Camera3-5min.mp4.pkl
```

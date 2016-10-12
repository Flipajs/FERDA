# Ground Truth
*Files can be loaded using python pickle module.*

```python
import cPickle as pickle

with open('path/to/gt.pkl') as f:
    gt = pickle.load(f)
    
gt[frame][0]  # is tuple (y, x)
gt[100][2]    # returns (y, x) of id 2 in frame 100

# IMPORTANT!
if gt[frame][ID][0] < 0 or frame not in gt or ID not in gt[frame]:
    # GT for given frame and ID is undefined.
```
| name | video | what | length | note |
| --- | --- | --- | --- | --- |
| __Cam1_sparse.pkl__ | ([Cam1_clip.avi](https://www.dropbox.com/s/afrbhqgucl27xy2/Cam1_clip.avi?dl=0)) | 6 ants with colormarks | 5 minutes | |
| __Cam2_sparse.pkl__ | ([Cam1_clip.avi](https://www.dropbox.com/s/qymo8x3y8af0muv/Cam2_clip.avi?dl=0)) | 6 ants with colormarks | 5 minutes | |
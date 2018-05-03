python ferda_cli.py /home/matej/prace/ferda/projects/vaib_results/Cam1_clip_arena_fixed/ --video-file /datagrid/ferda/data/ants_ist/camera_1/Cam1_clip.avi --save-results-mot experiments/tracking/Cam1_clip/180427_vaib/results.txt

python -m utils.gt.mot --load-mot experiments/tracking/Cam1_clip/180427_vaib/results.txt --load-gt data/GT/Cam1_clip.avi.txt --write-eval experiments/tracking/Cam1_clip/180427_vaib/evaluation.csv

python -m utils.gt.mot --load-tox experiments/tracking/Cam1_clip/__toxtrack/Tracking_0.txt --tox-topleft-xy 86 129 --load-gt data/GT/Cam1_clip.avi.txt --write-eval experiments/tracking/Cam1_clip/__toxtrack/evaluation.csv

python -m utils.gt.mot --load-tox experiments/tracking/Cam1_clip/__toxtrack_better_thresh/Tracking_0.txt --tox-topleft-xy 78 132 --load-gt data/GT/Cam1_clip.avi.txt --write-eval experiments/tracking/Cam1_clip/__toxtrack_better_thresh/evaluation.csv

python -m utils.gt.mot --load-idtracker experiments/tracking/Cam1_clip/__idtracker/trajectories_nogaps.txt --load-gt data/GT/Cam1_clip.avi.txt --write-eval experiments/tracking/Cam1_clip/__idtracker/evaluation_nogaps.csv



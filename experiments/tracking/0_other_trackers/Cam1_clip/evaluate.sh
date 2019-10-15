#!/bin/bash

set -ev

ROOT=experiments/tracking/0_other_trackers/Cam1_clip
GT=data/GT/Cam1_clip.avi.txt
VIDEO=/datagrid/ferda/data/ants_ist/camera_1/Cam1_clip.avi

# save results to MOT format

# python ferda_cli.py /home/matej/prace/ferda/projects/vaib_results/Cam1_clip_arena_fixed/ --video-file $VIDEO --save-results-mot $ROOT/180427_vaib/results.txt
# python -m utils.gt.mot --load-tox $ROOT/__toxtrac/Tracking_0.txt --tox-topleft-xy 86 129 --write-mot $ROOT/__toxtrac/results.txt
# python -m utils.gt.mot --load-tox $ROOT/__toxtrac_better_thresh/Tracking_0.txt --tox-topleft-xy 78 132 --write-mot $ROOT/__toxtrac_better_thresh/results.txt
# python -m utils.gt.mot --load-idtracker $ROOT/__idtracker/trajectories_nogaps.txt --write-mot $ROOT/__idtracker/results_nogaps.txt
# python -m utils.gt.mot --load-idtracker $ROOT/__idtracker/trajectories.txt --write-mot $ROOT/__idtracker/results.txt
python -m utils.gt.mot --load-idtrackerai $ROOT/191003_idtrackerai/trajectories.npy --write-mot $ROOT/191003_idtrackerai/results.txt


# evaluate

# python -m utils.gt.mot --load-mot $ROOT/180427_vaib/results.txt --load-gt $GT --write-eval $ROOT/180427_vaib/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__toxtrac/trajectories.txt --load-gt $GT --write-eval $ROOT/__toxtrac/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__toxtrac_better_thresh/trajectories.txt --load-gt $GT --write-eval $ROOT/__toxtrac_better_thresh/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__idtracker/results_nogaps.txt --load-gt $GT --write-eval $ROOT/__idtracker/evaluation_nogaps.csv
# python -m utils.gt.mot --load-mot $ROOT/__idtracker/results.txt --load-gt $GT --write-eval $ROOT/__idtracker/evaluation.csv
python -m utils.gt.mot --load-mot $ROOT/191003_idtrackerai/results.txt --load-gt $GT --write-eval $ROOT/191003_idtrackerai/evaluation.csv

# generate video

# python -m utils.gt.mot --load-mot $ROOT/180427_vaib/results.txt $ROOT/__idtracker/results.txt $ROOT/__toxtrac_better_thresh/results.txt --video-in $VIDEO --video-out Cam1_clip_comparision.avi --input-names 180427_vaib idtracker toxtrac



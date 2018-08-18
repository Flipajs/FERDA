#!/bin/bash

set -ev

ROOT=experiments/tracking/Sowbug3_cut
GT=data/GT/Sowbug3_cut.txt
VIDEO=/datagrid/ferda/data/youtube/Sowbug3_cut.mp4

# save results to MOT format

# python ferda_cli.py /home/matej/prace/ferda/projects/vaib_results/Sowbug3_cut_arena_fixed/ --video-file $VIDEO --save-results-mot $ROOT/180427_vaib/results.txt
# python -m utils.gt.mot --load-tox $ROOT/__toxtrac/Tracking_0.txt --tox-topleft-xy 52 40 --write-mot $ROOT/__toxtrac/results.txt
# python -m utils.gt.mot --load-idtracker $ROOT/__idtracker/trajectories_nogaps.txt --write-mot $ROOT/__idtracker/results_nogaps.txt
# python -m utils.gt.mot --load-idtracker $ROOT/__idtracker/trajectories.txt --write-mot $ROOT/__idtracker/results.txt

# evaluate

# python -m utils.gt.mot --load-mot $ROOT/180427_vaib/results.txt --load-gt $GT --write-eval $ROOT/180427_vaib/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__toxtrac/results.txt --load-gt $GT --write-eval $ROOT/__toxtrac/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__idtracker/results_nogaps.txt --load-gt $GT --write-eval $ROOT/__idtracker/evaluation_nogaps.csv
# python -m utils.gt.mot --load-mot $ROOT/__idtracker/results.txt --load-gt $GT --write-eval $ROOT/__idtracker/evaluation.csv

# generate video

# $ROOT/180427_vaib/results.txt
python -m utils.gt.mot --load-mot /home/matej/prace/ferda/projects/regression/Sowbug3_cut_min1_new/results.csv $ROOT/__idtracker/results_nogaps.txt $ROOT/__toxtrac/results.txt --video-in $VIDEO --video-out Sowbug3_cut_comparision.avi --input-names ferda_new idtracker_nogaps toxtrac



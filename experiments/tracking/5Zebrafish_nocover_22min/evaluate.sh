#!/bin/bash

set -ev

ROOT=experiments/tracking/5Zebrafish_nocover_22min
GT=data/GT/5Zebrafish_nocover_22min.txt
VIDEO=/datagrid/ferda/data/idTracker/5Zebrafish_nocover_22min.avi

# save results to MOT format

# python ferda_cli.py /home/matej/prace/ferda/projects/vaib_results/5Zebrafish_nocover_22min/ --video-file $VIDEO --save-results-mot $ROOT/180427_vaib/results.txt
# python -m utils.gt.mot --load-tox $ROOT/__toxtrac/Tracking_0.txt --tox-topleft-xy 198 1 --write-mot $ROOT/__toxtrac/results.txt
# python -m utils.gt.mot --load-idtracker $ROOT/__idtracker/trajectories_nogaps.txt --write-mot $ROOT/__idtracker/results_nogaps.txt
# python -m utils.gt.mot --load-idtracker $ROOT/__idtracker/trajectories.txt --write-mot $ROOT/__idtracker/results.txt

# evaluate

# python -m utils.gt.mot --load-mot $ROOT/180427_vaib/results.txt --load-gt $GT --write-eval $ROOT/180427_vaib/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__toxtrac/trajectories.txt --load-gt $GT --write-eval $ROOT/__toxtrac/evaluation.csv
# python -m utils.gt.mot --load-mot $ROOT/__idtracker/results_nogaps.txt --load-gt $GT --write-eval $ROOT/__idtracker/evaluation_nogaps.csv
# python -m utils.gt.mot --load-mot $ROOT/__idtracker/results.txt --load-gt $GT --write-eval $ROOT/__idtracker/evaluation.csv

# generate video

python -m utils.gt.mot --load-mot $ROOT/180427_vaib/results.txt $ROOT/__idtracker/results.txt $ROOT/__toxtrac/results.txt --video-in $VIDEO --video-out 5Zebrafish_nocover_22min_comparision.avi --input-names 180427_vaib idtracker toxtrac



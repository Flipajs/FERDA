#!/bin/sh

# python ferda_cli.py /home/matej/prace/ferda/projects/vaib_results/Sowbug3_cut_arena_fixed/ --video-file /datagrid/ferda/data/ants_ist/camera_1/Sowbug3_cut.avi --save-results-mot experiments/tracking/Sowbug3_cut/180427_vaib/results.txt

# python -m utils.gt.mot --load-mot experiments/tracking/Sowbug3_cut/180427_vaib/results.txt --load-gt data/GT/Sowbug3_cut.txt --write-eval experiments/tracking/Sowbug3_cut/180427_vaib/evaluation.csv

python -m utils.gt.mot --load-tox experiments/tracking/Sowbug3_cut/__toxtrack/Tracking_0.txt --tox-topleft-xy 52 40 --load-gt data/GT/Sowbug3_cut.txt --write-eval experiments/tracking/Sowbug3_cut/__toxtrack/evaluation.csv

# python -m utils.gt.mot --load-idtracker experiments/tracking/Sowbug3_cut/__idtracker/trajectories_nogaps.txt --load-gt data/GT/Sowbug3_cut.txt --write-eval experiments/tracking/Sowbug3_cut/__idtracker/evaluation_nogaps.csv



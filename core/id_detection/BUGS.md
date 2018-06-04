2018-06-01 12:53:14,328 - core.id_detection.complete_set_matching - INFO - do_complete_set_matching start
[Errno 2] No such file or directory: '../ferda_projects/5Zebrafish_nocover_22min//learning.pkl'
2018-06-01 12:53:21,503 - core.id_detection.complete_set_matching - INFO - analysing project, searching for complete sets
searching for complete sets: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14998/14998 [00:17<00:00, 880.77it/s]
2018-06-01 12:53:38,534 - core.id_detection.complete_set_matching - INFO - beginning of sequential matching
sequential matching:   0%|▏                                                                                                                                                       | 2/1946 [00:00<03:38,  8.88it/s]/home.dokt/smidm/local/miniconda2/envs/ferda_noqt/lib/python2.7/site-packages/scipy/stats/_distn_infrastructure.py:1735: RuntimeWarning: divide by zero encountered in double_scalars
  x = np.asarray((x - loc)/scale, dtype=dtyp)
sequential matching: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1946/1946 [02:32<00:00, 12.73it/s]
2018-06-01 12:56:11,386 - core.id_detection.complete_set_matching - INFO - sequential CS matching done...
2018-06-01 12:56:11,387 - core.id_detection.complete_set_matching - INFO - beginning of global matching
global matching:   9%|█████████████▊                                                                                                                                            | 15/168 [25:29<3:09:25, 74.29s/it]
Traceback (most recent call last):
  File "ferda_cli.py", line 140, in <module>
    run_tracking(args.project, video_file=args.video_file, reid_model_weights_path=args.reidentification_weights)
  File "ferda_cli.py", line 117, in run_tracking
    do_complete_set_matching(project)
  File "/mnt/home.dokt/smidm/ferda/core/id_detection/complete_set_matching.py", line 1463, in do_complete_set_matching
    csm.start_matching_process()
  File "/mnt/home.dokt/smidm/ferda/core/id_detection/complete_set_matching.py", line 62, in start_matching_process
    self.tracks_CS_matching(track_CSs)
  File "/mnt/home.dokt/smidm/ferda/core/id_detection/complete_set_matching.py", line 595, in tracks_CS_matching
    pivot, CS, use_spatial_probabilities=False
  File "/mnt/home.dokt/smidm/ferda/core/id_detection/complete_set_matching.py", line 797, in cs2cs_matching_prototypes_and_spatial
    assert len(cs2) > 1
AssertionError


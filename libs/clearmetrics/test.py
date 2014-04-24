import clearmetrics
import eight_gt
import eight_idtracker
import eight_ferda
import eight_ctrax

precision = 7

clear = clearmetrics.ClearMetrics(eight_gt.val, eight_idtracker.val, precision)
clear.match_sequence()

evaluationI = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]

clear = clearmetrics.ClearMetrics(eight_gt.val, eight_ctrax.val, precision)
clear.match_sequence()

evaluationC = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]

clear = clearmetrics.ClearMetrics(eight_gt.val, eight_ferda.val, precision)
clear.match_sequence()

evaluationF = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]
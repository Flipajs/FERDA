import clearmetrics
import noplast_gt
import noplast_ferda
# import noplast_ctrax

precision = 7

# clear = clearmetrics.ClearMetrics(noplast_gt.val, noplast_ctrax.val, precision)
# clear.match_sequence()

# evaluationC = [clear.get_mota(),
#               clear.get_motp(),
#               clear.get_fn_count(),
#               clear.get_fp_count(),
#               clear.get_mismatches_count(),
#               clear.get_object_count(),
#               clear.get_matches_count()]

clear = clearmetrics.ClearMetrics(noplast_gt.val, noplast_ferda.val, precision)
clear.match_sequence()

evaluationF = [clear.get_mota(),
              clear.get_motp(),
              clear.get_fn_count(),
              clear.get_fp_count(),
              clear.get_mismatches_count(),
              clear.get_object_count(),
              clear.get_matches_count()]
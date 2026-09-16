"""Published generators reimplemented for comparison on OUR data and OUR eval.

A baseline is only a baseline if it meets the same tables. Each one here trains
on the split_v3 TRAIN patches and writes eval_v4 cases, so `eval_v4 measure`
scores it with the same metrics, against the same real floor, as ldm06.
"""

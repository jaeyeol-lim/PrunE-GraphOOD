# PrunE on DrugOOD IC50

This is a thin data/protocol adapter around the released `modelNew.prune.Model`. Its actual released loss behavior is intentionally retained, including `DataAugLoss`, batch-level edge budget, lowest-K TV loss, and hard Gumbel edge sampling during evaluation.

The only compatibility workaround sets the uninitialized optional `calc_edge_stats` flag to `False`; it does not change the learning objective. The selector is fixed to a 2-layer GIN, so `sweep_ic50.py` enumerates 108 configurations for one seed. The shared DrugOOD batch size is 128.

import numpy as np

from tools.tcrd_common import ScheduleBatchSampler, derive_tail_replay_lrs


def test_schedule_batch_sampler_replays_exact_pairs():
    indices = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)
    seeds = (indices + 100).astype(np.int64)
    first = list(ScheduleBatchSampler(indices, seeds, epoch=1))
    second = list(ScheduleBatchSampler(indices, seeds, epoch=1))
    assert first == second
    assert first[0] == [(12, 112), (13, 113), (14, 114), (15, 115)]


def test_tail_lr_is_derived_from_official_epoch20_point():
    base, groups = derive_tail_replay_lrs(0.01, steps_per_epoch=1171)
    expected = 0.01 * (1.0 - 20.0 / 25.0) ** 0.9
    assert abs(base - expected) < 1e-12
    assert groups == [base, 2 * base, 10 * base, 20 * base]

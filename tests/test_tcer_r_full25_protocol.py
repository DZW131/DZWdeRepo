from tools.tcer_r_full25_common import (
    BATCH_SIZE, EPOCHS, MAX_STEPS, STEPS_PER_EPOCH,
    exploratory_decision,
)


def test_frozen_fresh25_protocol_constants():
    assert EPOCHS == 25
    assert BATCH_SIZE == 20
    assert STEPS_PER_EPOCH == 1171
    assert MAX_STEPS == 29275


def test_exploratory_gate_requires_all_three_criteria():
    assert exploratory_decision(0.15, 0.20, 0.005) == "TCER_R25_EXPLORATORY_GO"
    assert exploratory_decision(0.1499, 0.30, 0.01) == "TCER_R25_EXPLORATORY_CLOSE"
    assert exploratory_decision(0.30, 0.1999, 0.01) == "TCER_R25_EXPLORATORY_CLOSE"
    assert exploratory_decision(0.30, 0.30, 0.0049) == "TCER_R25_EXPLORATORY_CLOSE"

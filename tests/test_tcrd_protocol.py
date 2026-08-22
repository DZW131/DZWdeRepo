import torch

from network.resnet38_cls import Net as OfficialNet
from network.resnet38_cls_tcrd_gate import Net


def test_c0_forward_is_exact_official_a0_path():
    torch.manual_seed(7)
    official = OfficialNet(4)
    control = Net(4, branch="C0")
    control.load_state_dict(official.state_dict(), strict=True)
    official.eval()
    control.eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = official(image)
        actual = control(image)
    assert len(expected) == len(actual)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(expected_tensor, actual_tensor)


def test_only_candidate_branches_add_parameters():
    counts = {}
    for branch in ("C0", "D", "R", "DR"):
        model = Net(4, branch=branch)
        counts[branch] = sum(parameter.numel() for parameter in model.parameters())
    assert counts["D"] - counts["C0"] == 2
    assert counts["R"] - counts["C0"] == 7
    assert counts["DR"] - counts["C0"] == 9


def test_new_parameters_are_in_scratch_weight_group_once():
    for branch in ("D", "R", "DR"):
        model = Net(4, branch=branch)
        groups = model.get_parameter_groups()
        scratch_ids = [id(value) for value in groups[2]]
        for parameter in model.tcrd.parameters():
            assert scratch_ids.count(id(parameter)) == 1
        all_ids = [id(value) for group in groups for value in group]
        assert len(all_ids) == len(set(all_ids))


def test_predicted_presence_uses_frozen_thresholds_and_fallback():
    model = Net(4, branch="R")
    probability = torch.tensor(
        [[0.81, 0.89, 0.81, 0.59], [0.1, 0.2, 0.3, 0.4]]
    )
    active = model._predicted_presence(probability)
    assert torch.equal(active[0], torch.tensor([True, False, True, False]))
    assert torch.equal(active[1], torch.tensor([False, False, False, True]))


def test_training_reaction_requires_image_level_labels():
    model = Net(4, branch="R")
    try:
        model(torch.randn(1, 3, 32, 32))
    except ValueError as error:
        assert "image-level GT labels" in str(error)
    else:
        raise AssertionError("Reaction training silently accepted no labels")

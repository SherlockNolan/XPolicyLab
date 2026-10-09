"""The dual YAM deployment transforms must mirror RLDS training."""

import dataclasses
import pathlib

import numpy as np
import pytest

from openpi import transforms
from openpi.training import config


@pytest.mark.parametrize(
    ("name", "is_pi05", "relative"),
    [
        ("yam_dual_task004_pi0_absolute", False, False),
        ("yam_dual_task004_pi05_relative", True, True),
        ("yam_dual_task006_pi0_relative", False, True),
        ("yam_dual_task006_pi05_relative", True, True),
    ],
)
def test_dual_deployment_uses_quantiles_and_matching_action_representation(name, is_pi05, relative, tmp_path):
    train_config = config.get_config(name)
    data_config = train_config.data.create(tmp_path, train_config.model)

    assert train_config.model.pi05 is is_pi05
    assert data_config.use_quantile_norm is True
    assert any(isinstance(item, transforms.DeltaActions) for item in data_config.data_transforms.inputs) is relative
    assert any(isinstance(item, transforms.AbsoluteActions) for item in data_config.data_transforms.outputs) is relative


def test_seed25_pi0_normalization_is_unchanged(tmp_path):
    train_config = config.get_config("tpec_pi0_yam")
    data_config = train_config.data.create(pathlib.Path(tmp_path), train_config.model)
    assert data_config.use_quantile_norm is False


@pytest.mark.parametrize(
    "name",
    ["yam_dual_task004_pi05_relative", "yam_dual_task006_pi0_relative", "yam_dual_task006_pi05_relative"],
)
def test_relative_dual_decode_preserves_checkpoint_training_mask(name, tmp_path):
    """Do not reinterpret an existing checkpoint's trained delta dimensions."""
    train_config = config.get_config(name)
    data_config = train_config.data.create(tmp_path, train_config.model)
    restore = next(item for item in data_config.data_transforms.outputs if isinstance(item, transforms.AbsoluteActions))
    mask = np.asarray(restore.mask, dtype=bool)
    np.testing.assert_array_equal(mask, [True] * 6 + [False] + [True] * 6 + [False])

    state = np.arange(14, dtype=np.float32) / 10
    absolute = np.arange(14, dtype=np.float32)[None, :] / 7
    trained_delta = absolute.copy()
    trained_delta[:, mask] -= state[mask]
    decoded = restore({"state": state, "actions": trained_delta})["actions"]
    np.testing.assert_allclose(decoded, absolute, atol=1e-6)


def test_explicit_grouped_train_mask_override_controls_relative_decode(tmp_path):
    train_config = config.get_config("yam_dual_task004_pi05_relative")
    mask = (True,) * 6 + (False,) + (True,) * 6 + (False,)
    deployment_data = dataclasses.replace(train_config.data, delta_action_mask_override=mask)
    data_config = deployment_data.create(tmp_path, train_config.model)
    encoder = next(item for item in data_config.data_transforms.inputs if isinstance(item, transforms.DeltaActions))
    decoder = next(item for item in data_config.data_transforms.outputs if isinstance(item, transforms.AbsoluteActions))
    assert tuple(encoder.mask) == tuple(decoder.mask) == mask

    state = np.arange(14, dtype=np.float32) / 10
    absolute = np.arange(14, dtype=np.float32)[None, :] / 7
    training_sample = {"state": state.copy(), "actions": absolute.copy()}
    relative = encoder(training_sample)["actions"]
    np.testing.assert_allclose(relative[:, 12], absolute[:, 12] - state[12])
    np.testing.assert_allclose(relative[:, 6], absolute[:, 6])
    np.testing.assert_allclose(decoder({"state": state, "actions": relative})["actions"], absolute, atol=1e-6)
    with pytest.raises(ValueError, match="exactly 14 booleans"):
        dataclasses.replace(train_config.data, delta_action_mask_override=()).create(tmp_path, train_config.model)

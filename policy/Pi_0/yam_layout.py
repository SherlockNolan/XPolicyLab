"""YAM dual-checkpoint vector layout at the policy-server boundary.

The robot client uses [left arm(6), left gripper, right arm(6), right gripper].
The dual RLDS checkpoints use [left arm(6), right arm(6), left gripper, right gripper].
Keep the conversion outside OpenPI transforms so checkpoint normalization and
the training-time delta mask both see exactly the trained axis order.
"""

from __future__ import annotations

import numpy as np

ARMS_THEN_GRIPPERS = "arms_then_grippers"
INTERLEAVED = "interleaved"
GROUPED_TRAIN_ACTION_REPRESENTATION = "rlds_grouped_train_delta_mask"
BOUNDED_SLEW_ACTION_OVERRIDE = "bounded_slew_v1"
TRAINED_14D_ACTION_NOISE_MASK = "valid_14_zero_pad_v1"
# This is the mask recorded by the existing RLDS checkpoints, *in grouped
# checkpoint order*. It includes the left gripper and excludes right-arm j1.
# Do not silently replace it with a physically intuitive mask at inference.
GROUPED_TRAIN_DELTA_MASK = (True,) * 6 + (False,) + (True,) * 6 + (False,)
_TO_TRAINING = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13)
_TO_HARDWARE = (0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13)
_HARDWARE_ARM_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
_HARDWARE_GRIPPER_INDICES = (6, 13)


def _reorder(value: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 1 or array.shape[-1] != 14:
        raise ValueError(f"YAM vector must end in 14 dimensions, got {array.shape}")
    return np.ascontiguousarray(array[..., indices])


def to_training_layout(interleaved: np.ndarray) -> np.ndarray:
    """Move the left gripper from axis 6 to axis 12, preserving leading axes."""
    return _reorder(interleaved, _TO_TRAINING)


def to_hardware_layout(grouped: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_training_layout` for model output chunks."""
    return _reorder(grouped, _TO_HARDWARE)


def bound_hardware_action_slew(
    actions: np.ndarray,
    observed_state: np.ndarray,
    *,
    arm_step: float,
    gripper_step: float,
) -> np.ndarray:
    """Limit decoded absolute targets, anchored to the observation at chunk start.

    This operates *after* trained-mask decoding and axis reordering. It never
    reinterprets the checkpoint's relative/absolute output dimensions.
    """
    targets = np.asarray(actions, dtype=np.float64)
    initial = np.asarray(observed_state, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 14 or not len(targets):
        raise ValueError(f"YAM action chunk must have shape (time, 14), got {targets.shape}")
    if initial.shape != (14,):
        raise ValueError(f"YAM observed state must have shape (14,), got {initial.shape}")
    if not np.isfinite(targets).all() or not np.isfinite(initial).all():
        raise ValueError("YAM action chunk and observed state must be finite")
    if not (np.isfinite(arm_step) and 0 < arm_step <= 0.02):
        raise ValueError("arm_step must be finite and in (0, 0.02]")
    if not (np.isfinite(gripper_step) and 0 < gripper_step <= 0.10):
        raise ValueError("gripper_step must be finite and in (0, 0.10]")

    result = np.empty_like(targets)
    previous = initial.copy()
    for index, target in enumerate(targets):
        bounded = target.copy()
        bounded[list(_HARDWARE_GRIPPER_INDICES)] = np.clip(
            bounded[list(_HARDWARE_GRIPPER_INDICES)], 0.0, 1.0
        )
        delta = bounded - previous
        delta[list(_HARDWARE_ARM_INDICES)] = np.clip(
            delta[list(_HARDWARE_ARM_INDICES)], -arm_step, arm_step
        )
        delta[list(_HARDWARE_GRIPPER_INDICES)] = np.clip(
            delta[list(_HARDWARE_GRIPPER_INDICES)], -gripper_step, gripper_step
        )
        previous = previous + delta
        result[index] = previous
    return result

"""starVLA behind the existing XPolicyLab CALL protocol."""

import sys
from pathlib import Path

import numpy as np

from rhospolicy.inference.starvla_yam_launcher import BACKEND, MAPPING, model_identity
from XPolicyLab.model_template import ModelTemplate


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.settings = dict(model_cfg)
        if model_cfg.get("policy_backend") != BACKEND or model_cfg.get("embedding_mapping") != MAPPING:
            raise ValueError("starVLA backend/mapping mismatch")
        identity = model_identity(model_cfg)
        if identity != model_cfg.get("model_identity"):
            raise ValueError("model files changed since deployment command was built")
        sys.path.insert(0, str(Path(model_cfg["starvla_root"]).resolve()))
        from starVLA.inference.yam import YamPolicy

        self.policy = YamPolicy(model_cfg["model_path"], model_cfg["model_config"], model_cfg["norm_stats"],
                                model_cfg["base_vlm"], flow_steps=model_cfg["flow_steps"],
                                attention=model_cfg["attn_implementation"])
        self.model = self.policy.model
        self.observation = None
        self.policy.warmup(model_cfg["instruction"])

    def get_deployment_info(self):
        result = {key: self.settings[key] for key in ("policy_backend", "model_identity", "checkpoint",
                  "train_config_name", "state_action_layout", "embedding_mapping", "action_representation",
                  "model_config", "norm_stats", "base_vlm", "flow_steps", "attn_implementation")}
        return {**result, "model_action_horizon": 16, "ready": True,
                "action_override": None, "action_noise_mask": None}

    def update_obs(self, obs):
        state = np.asarray(obs["state"], dtype=np.float32).copy()
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("starVLA requires finite YAM state (14,)")
        cameras = {"main": "cam_high", "left": "cam_left_wrist", "right": "cam_right_wrist"}
        images = {name: np.asarray(obs["images"][source]).copy() for name, source in cameras.items()}
        self.observation = (state, images, obs["instruction"])

    def get_action(self):
        if self.observation is None:
            raise ValueError("update_obs must precede get_action")
        result = self.policy.infer(*self.observation)
        print(f"[INFO] starVLA inference: {self.policy.last_latency_s * 1000:.1f}ms; actions={result.shape}", flush=True)
        return result

    def get_last_inference_info(self):
        import torch

        return {"normalized_actions": self.policy.last_normalized,
                "latency_s": self.policy.last_latency_s, "loaded_keys": self.policy.loaded_keys,
                "attention_backends": self.policy.attention_backends,
                "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "gpu_reserved_bytes": torch.cuda.memory_reserved()}

    def reset(self):
        self.observation = None

"""Full SFT of DM05 on converted piper fold-cloth-in-place data.

Starts from ``./checkpoints/dm05_base``. Action targets are relative (delta)
joint positions; gripper dimensions stay absolute via ``state_desc``.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import tyro

from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
from opendm.exp.dm05_exp import (
    DM05DataConfig as _DM05DataConfig,
)
from opendm.exp.dm05_exp import (
    DM05Exp as _DM05Exp,
)
from opendm.exp.dm05_exp import (
    DM05InferenceConfig as _DM05InferenceConfig,
)
from opendm.exp.dm05_exp import (
    DM05ModelConfig as _DM05ModelConfig,
)
from opendm.exp.dm05_exp import (
    DM05OptimizerConfig as _DM05OptimizerConfig,
)
from opendm.exp.dm05_exp import (
    DM05TrainerConfig as _DM05TrainerConfig,
)


@dataclass
class DM05DataConfig(_DM05DataConfig):
    dataset_name: str = field(default="piper_fold_cloth_in_place")
    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    add_state: bool = field(default=True)


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    model_name_or_path: str | None = field(default="./checkpoints/dm05_base")
    chunk_size: int = field(default=50)
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    # Same scaling as stack-blocks-on-pegs: global batch 8x16=128.
    optim: Literal["adamw", "muon_adamw"] = field(default="muon_adamw")
    base_lr: float = field(default=4e-5)
    warmup_steps: int = field(default=1000)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    output_dir: str = field(
        default=f"user_checkpoints/{os.path.basename(__file__)[:-3]}"
    )
    fsdp1: bool | None = field(default=True)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=1)
    save_steps: int = field(default=10000)
    num_train_steps: int = field(default=40000)
    save_only_model: bool = field(default=False)
    wandb_project: str | None = field(default="opendm-piper-fold-cloth-in-place")


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=14)
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head", "Left wrist", "Right wrist"]
    )

    def _request_default_overrides(self) -> dict:
        return {
            "default_robot_type": RobotType.ALOHA.value,
            "default_state_desc": list(ROBOT_STATE_DESCS[RobotType.ALOHA]),
        }


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


if __name__ == "__main__":
    exp = tyro.cli(DM05Exp)
    if exp.task == "train":
        exp.train()
    elif exp.task == "inference":
        exp.inference()
    else:
        raise ValueError(f"Invalid task: {exp.task}")

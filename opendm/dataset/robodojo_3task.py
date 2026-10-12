"""RoboDojo 3-task SFT dataset registration.

JSONL lives under ``./data/robodojo_3task/jsonl``. Videos stay in the original
LeRobot v2.1 tree and are resolved through ``image_dir``.
"""

from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset

ROBODOJO_IMAGE_DIR = (
    "/kpfs_ssd/data/ruitong_gan/RoboDojo/.cache/"
    "robodojo_data_modelscope_lerobot_v2.1_repo/data/RoboDojo_lerobot_v21_video"
)

ROBODOJO_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
)

register_dataset(
    {
        "robodojo_3task": {
            "jsonl_dir": "./data/robodojo_3task/jsonl",
            "image_dir": ROBODOJO_IMAGE_DIR,
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": "Dual ARX5",
            "state_desc": ROBODOJO_STATE_DESC,
        },
    },
)

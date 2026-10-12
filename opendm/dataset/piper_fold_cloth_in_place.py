"""Piper fold-cloth-in-place dataset registration.

Videos stay on the read-only LeRobot disk. JSONL lives under
``./data/piper_fold_cloth_in_place/jsonl`` and each frame's media ``url`` is
resolved against ``image_dir``.
"""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

FOLD_CLOTH_IMAGE_DIR = "/mnt/xiaoyu_teleop_data/piper/20260911"

register_dataset(
    {
        "piper_fold_cloth_in_place": {
            "jsonl_dir": "./data/piper_fold_cloth_in_place/jsonl",
            "image_dir": FOLD_CLOTH_IMAGE_DIR,
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA],
        },
    },
)

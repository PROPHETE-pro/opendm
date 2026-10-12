"""Piper stack-blocks-on-pegs dataset registration.

Videos stay on the read-only LeRobot disk. JSONL lives under
``./data/stack_blocsk_on_pegs_0908/jsonl`` and each frame's media ``url`` is
resolved against ``image_dir``.
"""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

STACK_BLOCKS_IMAGE_DIR = "/mnt/xiaoyu_teleop_data/piper/20260907"

register_dataset(
    {
        "stack_blocsk_on_pegs_0908": {
            "jsonl_dir": "./data/stack_blocsk_on_pegs_0908/jsonl",
            "image_dir": STACK_BLOCKS_IMAGE_DIR,
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA],
        },
    },
)

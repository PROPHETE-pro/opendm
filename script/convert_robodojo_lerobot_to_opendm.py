#!/usr/bin/env python3
"""Convert selected RoboDojo LeRobot v2.1 tasks into OpenDM JSONL.

Videos stay on the source disk. OpenDM resolves each frame's relative ``url``
against ``image_dir`` (the LeRobot dataset root).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson
import pyarrow.parquet as pq
from tqdm import tqdm

CAMERA_TO_IMAGE_KEY = (
    ("observation.images.cam_high", "images_1"),
    ("observation.images.cam_left_wrist", "images_2"),
    ("observation.images.cam_right_wrist", "images_3"),
)
TASK_PROMPT_TO_NAME = {
    "Build a tower using the wooden blocks and wooden boards.": "build_tower",
    "Insert and tighten each screw into the nut of the same color.": "fasten_screws",
    "Insert the three tubes into the rack one by one.": "insert_tubes",
}
DEFAULT_TASKS = ("build_tower", "fasten_screws", "insert_tubes")
MIN_FRAMES = 2


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert RoboDojo LeRobot v2.1 episodes to OpenDM JSONL. "
            "Videos are referenced in place; they are not copied."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(
            "/kpfs_ssd/data/ruitong_gan/RoboDojo/.cache/"
            "robodojo_data_modelscope_lerobot_v2.1_repo/data/"
            "RoboDojo_lerobot_v21_video"
        ),
        help="LeRobot v2.1 dataset root that contains meta/, data/, and videos/.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=repo_root / "data" / "robodojo_3task",
        help="Output dataset root. JSONL is written to <dst-root>/jsonl.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task names to convert. Default: build_tower fasten_screws insert_tubes.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing JSONL files.",
    )
    return parser.parse_args()


def load_jsonl_objects(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def video_relpath(info: dict, camera_key: str, episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size") or 1000)
    episode_chunk = episode_index // chunks_size
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    return template.format(
        episode_chunk=episode_chunk,
        video_key=camera_key,
        episode_index=episode_index,
    )


def parquet_path(dataset_dir: Path, info: dict, episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size") or 1000)
    episode_chunk = episode_index // chunks_size
    template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    relative = template.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    return dataset_dir / relative


def to_float_list(values) -> list[float]:
    return [float(value) for value in values]


def convert_episode(payload: dict) -> dict:
    jsonl_path = Path(payload["jsonl_path"])
    if jsonl_path.exists() and not payload["force"]:
        n_frames = sum(1 for line in jsonl_path.open("rb") if line.strip())
        return {
            "status": "skipped",
            "jsonl_path": str(jsonl_path),
            "n_frames": n_frames,
            "task_name": payload["task_name"],
            "episode_index": payload["episode_index"],
        }

    parquet_file = Path(payload["parquet_path"])
    if not parquet_file.is_file():
        return {
            "status": "missing_parquet",
            "task_name": payload["task_name"],
            "episode_index": payload["episode_index"],
            "parquet_path": str(parquet_file),
        }

    missing_videos = [url for url in payload["video_urls"] if not Path(url).is_file()]
    if missing_videos:
        return {
            "status": "missing_video",
            "task_name": payload["task_name"],
            "episode_index": payload["episode_index"],
            "missing_videos": missing_videos,
        }

    table = pq.read_table(
        parquet_file,
        columns=["observation.state", "action", "frame_index"],
    )
    states = table.column("observation.state").to_pylist()
    actions = table.column("action").to_pylist()
    frame_indices = table.column("frame_index").to_pylist()
    n_frames = len(states)
    if n_frames < MIN_FRAMES:
        return {
            "status": "too_short",
            "task_name": payload["task_name"],
            "episode_index": payload["episode_index"],
            "n_frames": n_frames,
        }

    image_specs = [{"type": "video", "url": rel_url} for rel_url in payload["video_rel_urls"]]
    prompt = payload["prompt"]
    extra = {
        "source": payload["dataset_name"],
        "task_name": payload["task_name"],
        "episode_index": payload["episode_index"],
    }

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("wb") as handle:
        for frame_i, state, action in zip(frame_indices, states, actions):
            record = {
                "images_1": {**image_specs[0], "frame_idx": int(frame_i)},
                "images_2": {**image_specs[1], "frame_idx": int(frame_i)},
                "images_3": {**image_specs[2], "frame_idx": int(frame_i)},
                "state": to_float_list(state),
                "action": to_float_list(action),
                "prompt": prompt,
                "is_robot": True,
                "extra": extra,
            }
            handle.write(orjson.dumps(record))
            handle.write(b"\n")

    return {
        "status": "converted",
        "jsonl_path": str(jsonl_path),
        "n_frames": n_frames,
        "task_name": payload["task_name"],
        "episode_index": payload["episode_index"],
    }


def build_jobs(
    src_root: Path,
    dst_root: Path,
    selected_tasks: set[str],
    force: bool,
) -> tuple[list[dict], list[dict]]:
    info = json.loads((src_root / "meta" / "info.json").read_text(encoding="utf-8"))
    camera_keys = [camera for camera, _ in CAMERA_TO_IMAGE_KEY]
    missing_cameras = [camera for camera in camera_keys if camera not in info["features"]]
    if missing_cameras:
        raise ValueError(f"{src_root} missing cameras: {missing_cameras}")

    unknown_tasks = selected_tasks - set(TASK_PROMPT_TO_NAME.values())
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {sorted(unknown_tasks)}")

    episodes = load_jsonl_objects(src_root / "meta" / "episodes.jsonl")
    jsonl_dir = dst_root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    per_task: dict[str, dict] = {
        task_name: {
            "name": task_name,
            "prompt": prompt,
            "n_episodes": 0,
            "n_frames_meta": 0,
            "episode_indices": [],
        }
        for prompt, task_name in TASK_PROMPT_TO_NAME.items()
        if task_name in selected_tasks
    }

    for episode in episodes:
        prompt = (episode.get("tasks") or [""])[0]
        task_name = TASK_PROMPT_TO_NAME.get(prompt)
        if task_name not in selected_tasks:
            continue

        episode_index = int(episode["episode_index"])
        video_rel_urls = [
            video_relpath(info, camera, episode_index) for camera, _ in CAMERA_TO_IMAGE_KEY
        ]
        jobs.append(
            {
                "dataset_name": src_root.name,
                "task_name": task_name,
                "episode_index": episode_index,
                "prompt": prompt,
                "parquet_path": str(parquet_path(src_root, info, episode_index)),
                "video_rel_urls": video_rel_urls,
                "video_urls": [str(src_root / rel) for rel in video_rel_urls],
                "jsonl_path": str(
                    jsonl_dir / f"{task_name}_episode_{episode_index:06d}.jsonl"
                ),
                "force": force,
            }
        )
        summary = per_task[task_name]
        summary["n_episodes"] += 1
        summary["n_frames_meta"] += int(episode.get("length") or 0)
        summary["episode_indices"].append(episode_index)

    if not jobs:
        raise FileNotFoundError(
            f"No episodes matched tasks {sorted(selected_tasks)} under {src_root}"
        )
    return jobs, list(per_task.values())


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if not src_root.is_dir():
        raise SystemExit(f"Source root not found: {src_root}")
    if src_root == dst_root or dst_root.is_relative_to(src_root):
        raise SystemExit(
            f"Refusing to write into the source tree: src={src_root} dst={dst_root}"
        )

    selected_tasks = set(args.tasks)
    print(f"Source root (videos stay here): {src_root}")
    print(f"JSONL output: {dst_root / 'jsonl'}")
    print(f"Tasks: {', '.join(sorted(selected_tasks))}")

    jobs, task_summaries = build_jobs(src_root, dst_root, selected_tasks, args.force)
    print("Episodes:")
    for summary in task_summaries:
        print(
            f"  - {summary['name']}: {summary['n_episodes']} episodes, "
            f"{summary['n_frames_meta']} frames"
        )

    stats = {
        "converted": 0,
        "skipped": 0,
        "too_short": 0,
        "missing_parquet": 0,
        "missing_video": 0,
    }
    n_frames = 0
    failures = []

    workers = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_episode, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            result = future.result()
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1
            if status in {"converted", "skipped"}:
                n_frames += int(result.get("n_frames") or 0)
            else:
                failures.append(result)

    manifest = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "image_dir": str(src_root),
        "jsonl_dir": str(dst_root / "jsonl"),
        "videos_copied": False,
        "note": (
            "Videos remain on the source disk. OpenDM should use image_dir=src_root "
            "and the relative urls stored in each JSONL frame."
        ),
        "tasks": sorted(selected_tasks),
        "n_jobs": len(jobs),
        "n_frames_written": n_frames,
        "stats": stats,
        "task_summaries": task_summaries,
        "skipped_or_failed": failures,
        "image_keys": ["images_1", "images_2", "images_3"],
        "image_prompts": ["Head", "Left wrist", "Right wrist"],
        "cameras": [camera for camera, _ in CAMERA_TO_IMAGE_KEY],
        "state_dim": 14,
        "action_dim": 14,
    }
    (dst_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2))
    print(f"Frames in converted/skipped JSONL: {n_frames}")
    print(f"Manifest: {dst_root / 'manifest.json'}")
    if stats.get("missing_parquet") or stats.get("missing_video"):
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())

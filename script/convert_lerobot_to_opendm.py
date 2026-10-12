#!/usr/bin/env python3
"""Convert LeRobot v2.x datasets into OpenDM JSONL without copying videos.

OpenDM trains from episode JSONL plus media URLs resolved against ``image_dir``.
``LoadImages`` joins ``image_dir`` with each frame's relative ``url``, so videos
can stay on a read-only source disk. This script only writes JSONL (and a
manifest) onto the destination filesystem.
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
    ("observation.images.top_head", "images_1"),
    ("observation.images.hand_left", "images_2"),
    ("observation.images.hand_right", "images_3"),
)
MIN_FRAMES = 2


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert LeRobot piper datasets to OpenDM JSONL. "
            "Videos are referenced in place; they are not copied."
        )
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/mnt/xiaoyu_teleop_data/piper/latest"),
        help="Read-only LeRobot root that contains task folders.",
    )
    parser.add_argument(
        "--name-substr",
        default="stack_blocks_on_pegs",
        help="Only convert folders whose names contain this substring.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=repo_root / "data" / "stack_blocsk_on_pegs_0908",
        help="Output dataset root on a writable disk.",
    )
    parser.add_argument("--workers", type=int, default=8)
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


def discover_datasets(src_root: Path, name_substr: str) -> list[Path]:
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {src_root}")
    exact = src_root / name_substr
    if exact.is_dir():
        return [exact]
    datasets = sorted(
        path
        for path in src_root.iterdir()
        if path.is_dir() and name_substr in path.name
    )
    if not datasets:
        raise FileNotFoundError(
            f"No folders containing {name_substr!r} under {src_root}"
        )
    return datasets


def video_relpath(dataset_name: str, info: dict, camera_key: str, episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size") or 1000)
    episode_chunk = episode_index // chunks_size
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    relative = template.format(
        episode_chunk=episode_chunk,
        video_key=camera_key,
        episode_index=episode_index,
    )
    return f"{dataset_name}/{relative}"


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
            "source": payload["dataset_name"],
            "episode_index": payload["episode_index"],
        }

    parquet_file = Path(payload["parquet_path"])
    if not parquet_file.is_file():
        return {
            "status": "missing_parquet",
            "source": payload["dataset_name"],
            "episode_index": payload["episode_index"],
            "parquet_path": str(parquet_file),
        }

    missing_videos = [url for url in payload["video_urls"] if not Path(url).is_file()]
    if missing_videos:
        return {
            "status": "missing_video",
            "source": payload["dataset_name"],
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
            "source": payload["dataset_name"],
            "episode_index": payload["episode_index"],
            "n_frames": n_frames,
        }

    image_specs = [
        {"type": "video", "url": rel_url}
        for rel_url in payload["video_rel_urls"]
    ]
    prompt = payload["prompt"]
    extra = {
        "source": payload["dataset_name"],
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
        "source": payload["dataset_name"],
        "episode_index": payload["episode_index"],
    }


def build_jobs(src_root: Path, datasets: list[Path], dst_root: Path, force: bool) -> list[dict]:
    jsonl_dir = dst_root / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    summaries = []

    for dataset_dir in datasets:
        info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        camera_keys = [camera for camera, _ in CAMERA_TO_IMAGE_KEY]
        missing_cameras = [camera for camera in camera_keys if camera not in info["features"]]
        if missing_cameras:
            raise ValueError(f"{dataset_dir.name} missing cameras: {missing_cameras}")

        tasks = load_jsonl_objects(dataset_dir / "meta" / "tasks.jsonl")
        task_by_index = {int(row.get("task_index", i)): row["task"] for i, row in enumerate(tasks)}
        default_prompt = tasks[0]["task"] if tasks else dataset_dir.name
        episodes = load_jsonl_objects(dataset_dir / "meta" / "episodes.jsonl")

        summaries.append(
            {
                "name": dataset_dir.name,
                "path": str(dataset_dir),
                "robot_type": info.get("robot_type"),
                "fps": info.get("fps"),
                "n_episodes": len(episodes),
                "n_frames_meta": info.get("total_frames"),
                "prompt": default_prompt,
            }
        )

        for episode in episodes:
            episode_index = int(episode["episode_index"])
            prompt = (episode.get("tasks") or [default_prompt])[0]
            video_rel_urls = [
                video_relpath(dataset_dir.name, info, camera, episode_index)
                for camera, _ in CAMERA_TO_IMAGE_KEY
            ]
            jobs.append(
                {
                    "dataset_name": dataset_dir.name,
                    "episode_index": episode_index,
                    "prompt": prompt,
                    "parquet_path": str(parquet_path(dataset_dir, info, episode_index)),
                    "video_rel_urls": video_rel_urls,
                    "video_urls": [str(src_root / rel) for rel in video_rel_urls],
                    "jsonl_path": str(
                        jsonl_dir / f"{dataset_dir.name}_episode_{episode_index:06d}.jsonl"
                    ),
                    "force": force,
                }
            )
    return jobs, summaries


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    if src_root == dst_root or dst_root.is_relative_to(src_root):
        raise SystemExit(
            f"Refusing to write into the source tree: src={src_root} dst={dst_root}"
        )

    datasets = discover_datasets(src_root, args.name_substr)
    print(f"Source root (videos stay here): {src_root}")
    print(f"JSONL output: {dst_root / 'jsonl'}")
    print("Datasets:")
    for dataset_dir in datasets:
        print(f"  - {dataset_dir.name}")

    jobs, dataset_summaries = build_jobs(src_root, datasets, dst_root, args.force)
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
            elif status in {"missing_parquet", "missing_video"}:
                failures.append(result)
            elif status == "too_short":
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
        "n_jobs": len(jobs),
        "n_frames_written": n_frames,
        "stats": stats,
        "datasets": dataset_summaries,
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

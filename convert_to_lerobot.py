#!/usr/bin/env python3
"""Phase 3 — 録画 bag (+ meta.json) を LeRobot データセットへ変換する（IO）。

datasets/raw/<episode>/{bag, meta.json} を走査し、label=success のエピソードを固定 10Hz
で同期して 1 つの LeRobot データセットにまとめる。両モデル（ACT / SmolVLA）が食えるよう
union を出す: observation.images.front + observation.state(=直前の行動) + action(vx,wz)
+ task。

責任分離: 同期・整列の純ロジックは ``mechlab3_policy.convert``（pytest 済み）。ここは IO
だけ —— rosbags での bag デコード / cv2 での JPEG デコード / LeRobot 書き出し。
ROS 非依存: bag は rosbags で読む（ROS インストール不要）。

⚠️ 未検証部分: LeRobotDataset の書き出し API は lerobot のバージョンで変わる。
``build_dataset`` / ``add_episode`` は現行 API 想定で書いてあるが、実データ + 実 lerobot
での検証が要る（純ロジック convert.py は pytest 済み）。docs:
https://huggingface.co/docs/lerobot  契約の正本: kuas-mechlab3 docs/autonomy-plan.md §3。

例:
    python convert_to_lerobot.py --raw-root ../kuas-mechlab3/datasets/raw \\
        --repo-id mechlab3/route_a --out datasets/lerobot
"""

import argparse
import json
import sys
from pathlib import Path

from mechlab3_policy.convert import build_frame_rows, stamp_to_seconds

ACTION_TOPIC = "/cmd_norm"
FRONT_TOPIC = "/front_camera/image_raw/compressed"
CONTROL_HZ = 10.0
TASK = "follow the route"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--raw-root", required=True, help="datasets/raw（エピソード dir 群の親）"
    )
    p.add_argument(
        "--repo-id", required=True, help="LeRobot データセット名（例 mechlab3/route_a）"
    )
    p.add_argument("--out", default="datasets/lerobot", help="出力ルート")
    p.add_argument("--hz", type=float, default=CONTROL_HZ)
    p.add_argument("--action-topic", default=ACTION_TOPIC)
    p.add_argument("--camera-topic", default=FRONT_TOPIC)
    p.add_argument(
        "--image-mode",
        choices=("video", "image"),
        default="video",
        help="video=mp4（要 ffmpeg/av）/ image=PNG",
    )
    return p.parse_args(argv)


def find_success_episodes(raw_root: Path) -> list[tuple[Path, dict]]:
    """raw_root 直下の meta.json を見て label=success のエピソードだけ返す。"""
    episodes = []
    for d in sorted(raw_root.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("label") == "success":
            episodes.append((d, meta))
    return episodes


def read_bag(bag_dir: Path, action_topic: str, camera_topic: str):
    """bag から (action_ts, actions, frame_ts, frames_jpeg) を取り出す（rosbags）。

    action = TwistStamped の (linear.x, angular.z)、frame = CompressedImage の JPEG bytes。
    時刻はどちらも header.stamp（同一 Pi クロック）。各トピックは bag 内で時刻昇順。
    """
    from rosbags.highlevel import AnyReader

    action_ts: list[float] = []
    actions: list[tuple[float, float]] = []
    frame_ts: list[float] = []
    frames: list[bytes] = []
    with AnyReader([bag_dir]) as reader:
        wanted = [
            c for c in reader.connections if c.topic in (action_topic, camera_topic)
        ]
        for con, _ts, raw in reader.messages(connections=wanted):
            msg = reader.deserialize(raw, con.msgtype)
            t = stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
            if con.topic == action_topic:
                action_ts.append(t)
                actions.append((float(msg.twist.linear.x), float(msg.twist.angular.z)))
            else:
                frame_ts.append(t)
                frames.append(bytes(msg.data))
    return action_ts, actions, frame_ts, frames


def decode_jpeg(buf: bytes):
    """JPEG bytes → RGB numpy 配列（LeRobot は RGB 期待）。"""
    import cv2
    import numpy as np

    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG デコードに失敗")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_dataset(
    repo_id: str, out: str, hz: float, height: int, width: int, image_mode: str
):
    """空の LeRobotDataset を作る（features は両モデル対応の union）。

    ⚠️ API は lerobot のバージョン依存。要・実 lerobot での検証。
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(hz)),
        root=Path(out) / repo_id,
        features={
            "observation.images.front": {
                "dtype": image_mode,  # "video"(mp4) or "image"(png)
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["vx", "wz"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["vx", "wz"]},
        },
    )


def add_episode(dataset, rows, frames: list[bytes]) -> None:
    """1 エピソード分のフレームを足して保存する。

    ⚠️ add_frame / save_episode の task の渡し方は lerobot バージョン依存（要検証）。
    古い版では frame dict に "task" を入れ save_episode(task=...) とする場合がある。
    """
    import numpy as np

    for row in rows:
        img = decode_jpeg(frames[row.frame_index])
        dataset.add_frame(
            {
                "observation.images.front": img,
                "observation.state": np.asarray(row.state, dtype=np.float32),
                "action": np.asarray(row.action, dtype=np.float32),
            },
            task=TASK,
        )
    dataset.save_episode()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_root = Path(args.raw_root)
    episodes = find_success_episodes(raw_root)
    if not episodes:
        print(f"success エピソードが見つからない: {raw_root}", file=sys.stderr)
        return 1

    dataset = None
    total = 0
    for episode_dir, meta in episodes:
        bag_dir = episode_dir / meta.get("bag_dir", "bag")
        action_ts, actions, frame_ts, frames = read_bag(
            bag_dir, args.action_topic, args.camera_topic
        )
        rows = build_frame_rows(action_ts, actions, frame_ts, args.hz)
        if not rows:
            print(f"skip（同期フレーム 0）: {episode_dir.name}")
            continue
        if dataset is None:
            h, w = decode_jpeg(frames[rows[0].frame_index]).shape[:2]
            dataset = build_dataset(
                args.repo_id, args.out, args.hz, h, w, args.image_mode
            )
        add_episode(dataset, rows, frames)
        total += len(rows)
        print(f"✓ {episode_dir.name}: {len(rows)} frames")

    if dataset is None:
        print("変換対象なし（全エピソードで同期フレーム 0）", file=sys.stderr)
        return 1
    print(
        f"完了: {len(episodes)} episodes / {total} frames -> {Path(args.out) / args.repo_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase 3 — 録画 bag (+ meta.json) を LeRobot データセットへ変換する。

kuas-mechlab3 の `datasets/raw/<...>/{bag, meta.json}` を読み、固定 10Hz で同期した
(観測, 行動) 列を LeRobot 形式へ書き出す。両モデル（ACT / SmolVLA）が食えるよう
union フィールドを出す: `observation.images.front` + `observation.state`(=直前の行動)
+ `action`(=vx,wz) + `task="follow the route"`。`label==success` のエピソードだけ採用。

ROS 非依存: bag は `rosbags` で読む（ROS インストール不要）。同期・整形の純ロジックは
`mechlab3_policy.sync` / `mechlab3_policy.convert`（pytest 済み・重依存なし）。重い依存
（rosbags / cv2 / numpy / lerobot）はこの IO 層の関数内で遅延 import する
—— `--help` と純ロジックのテストは重依存なしで通る。

使い方:
    # 単一エピソード
    python convert_to_lerobot.py --raw-dir datasets/raw/2025..._route_001 \\
        --out datasets/lerobot --repo-id mechlab3/route

    # ルート配下の全エピソード（*/meta.json）を 1 つのデータセットへ
    python convert_to_lerobot.py --raw-root datasets/raw \\
        --out datasets/lerobot --repo-id mechlab3/route --overwrite

契約の正本: kuas-mechlab3 の docs/autonomy-plan.md §3。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from mechlab3_policy.convert import (
    build_features,
    is_trainable,
    sort_by_time,
    stamp_to_seconds,
    synced_episode,
)

ACTION_TOPIC = "/cmd_norm"
FRONT_TOPIC = "/front_camera/image_raw/compressed"
CONTROL_HZ = 10.0
TASK = "follow the route"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--raw-dir",
        action="append",
        metavar="DIR",
        help="個別エピソード（datasets/raw/<episode>）。複数回指定可",
    )
    p.add_argument(
        "--raw-root",
        metavar="DIR",
        help="配下の */meta.json を全て 1 データセットへまとめる",
    )
    p.add_argument(
        "--out", default="datasets/lerobot", help="LeRobot データセット出力先（root）"
    )
    p.add_argument(
        "--repo-id", default="mechlab3/teleop", help="LeRobot dataset id（保存名）"
    )
    p.add_argument("--hz", type=float, default=CONTROL_HZ)
    p.add_argument("--action-topic", default=ACTION_TOPIC)
    p.add_argument("--camera-topic", default=FRONT_TOPIC)
    p.add_argument(
        "--overwrite", action="store_true", help="既存の出力を消してから作り直す"
    )
    return p.parse_args(argv)


def read_meta(raw_dir: Path) -> dict[str, Any]:
    """meta.json を読む（採否は `is_trainable` が判定する）。"""
    result: dict[str, Any] = json.loads(
        (raw_dir / "meta.json").read_text(encoding="utf-8")
    )
    return result


def discover_episodes(raw_dirs: list[str] | None, raw_root: str | None) -> list[Path]:
    """`--raw-dir`（複数可）と `--raw-root` からエピソード dir 群を重複なく集める。

    `--raw-root` 配下は `*/meta.json` を持つ dir だけを名前順（＝概ね時系列）に拾う。
    """
    dirs: list[Path] = [Path(d) for d in (raw_dirs or [])]
    if raw_root:
        dirs.extend(sorted(p.parent for p in Path(raw_root).glob("*/meta.json")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        key = d.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def read_bag(
    bag_dir: Path, action_topic: str, camera_topic: str
) -> tuple[list[float], list[tuple[float, float]], list[float], list[bytes]]:
    """bag から (action_ts, actions) と (frame_ts, frames) を取り出す（時刻昇順）。

    - action_topic (`geometry_msgs/TwistStamped`): header.stamp -> t,
      (twist.linear.x, twist.angular.z) を (vx, wz) として。正規化 [-1,1]。
    - camera_topic (`sensor_msgs/CompressedImage`): header.stamp -> t, msg.data(JPEG bytes)。

    メッセージは記録時刻順に届くので、各ストリームを header.stamp で昇順に直して返す
    （`sync` は昇順前提）。
    """
    from rosbags.highlevel import AnyReader  # 重依存: 遅延 import

    action_ts: list[float] = []
    actions: list[tuple[float, float]] = []
    frame_ts: list[float] = []
    frames: list[bytes] = []
    wanted = {action_topic, camera_topic}
    with AnyReader([Path(bag_dir)]) as reader:
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, _timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            stamp = msg.header.stamp
            t = stamp_to_seconds(stamp.sec, stamp.nanosec)
            if conn.topic == action_topic:
                action_ts.append(t)
                actions.append((float(msg.twist.linear.x), float(msg.twist.angular.z)))
            else:
                frame_ts.append(t)
                frames.append(bytes(msg.data))
    s_action_ts, s_actions = sort_by_time(action_ts, actions)
    s_frame_ts, s_frames = sort_by_time(frame_ts, frames)
    return s_action_ts, s_actions, s_frame_ts, s_frames


def decode_jpeg(jpeg: bytes) -> Any:
    """JPEG bytes -> HWC uint8 RGB の numpy 配列（LeRobot が期待する並び）。"""
    import cv2  # 重依存: 遅延 import
    import numpy as np

    buf = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG のデコードに失敗（壊れたフレーム）")
    # OpenCV は BGR。LeRobot / 学習系は RGB を期待するので変換する。
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = discover_episodes(args.raw_dir, args.raw_root)
    if not episodes:
        print("エピソードが無い（--raw-dir か --raw-root を指定）", file=sys.stderr)
        return 2

    out = Path(args.out)
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    if out.exists():
        print(
            f"出力が既に存在: {out}（--overwrite で作り直すか別の --out を指定）",
            file=sys.stderr,
        )
        return 2

    import inspect

    import numpy as np  # 重依存: 遅延 import
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # LeRobot のバージョン差を吸収: 新しめは add_frame(frame, task=...)、古いものは
    # frame dict に "task" を入れる。ここで一度だけ判定する。
    add_frame_takes_task = (
        "task" in inspect.signature(LeRobotDataset.add_frame).parameters
    )

    ds = None
    kept = 0
    for raw_dir in episodes:
        meta = read_meta(raw_dir)
        if not is_trainable(meta):
            print(f"skip {raw_dir.name}: label={meta.get('label')}（success のみ採用）")
            continue

        bag_dir = raw_dir / meta["bag_dir"]
        action_ts, actions, frame_ts, frames = read_bag(
            bag_dir, args.action_topic, args.camera_topic
        )
        sframes, sactions, states = synced_episode(
            action_ts, actions, frame_ts, frames, args.hz
        )
        if not sframes:
            print(
                f"skip {raw_dir.name}: 同期後フレーム 0（重なり無し / トピック不一致）"
            )
            continue

        for jpeg, action, state in zip(sframes, sactions, states):
            img = decode_jpeg(jpeg)
            if ds is None:
                # 最初のフレームで解像度が確定してから features / dataset を作る。
                features = build_features(img.shape)
                ds = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    fps=int(round(args.hz)),
                    root=out,
                    features=features,
                    use_videos=True,
                )
            frame: dict[str, Any] = {
                "observation.images.front": img,
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
            }
            if add_frame_takes_task:
                ds.add_frame(frame, task=TASK)
            else:
                ds.add_frame({**frame, "task": TASK})
        ds.save_episode()
        kept += 1
        print(f"episode {raw_dir.name}: {len(sframes)} frames")

    if ds is None:
        print("success エピソードが無く、データセットは作られなかった")
        return 0
    print(f"完了: {kept} episodes -> {out}  (repo_id={args.repo_id}, {args.hz}Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

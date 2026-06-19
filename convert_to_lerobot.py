#!/usr/bin/env python3
"""Phase 3 — 録画 bag (+ meta.json) を LeRobot データセットへ変換する。

kuas-mechlab3 の `datasets/raw/<...>/{bag, meta.json}` を読み、固定 10Hz で同期した
(観測, 行動) 列を LeRobot 形式へ書き出す。両モデル（ACT / SmolVLA）が食えるよう
union フィールドを出す: `observation.images.front` + `observation.state(=直前の行動)`
+ `action=(vx,wz)` + `task="follow the route"`。

ROS 非依存: bag は `rosbags` で読む（ROS インストール不要）。同期の純ロジックは
`mechlab3_policy.sync`（pytest 済み）。

注意: これはスケルトン。bag からの実デコードと LeRobot 書き出し（API はバージョン依存）
は TODO。アルゴリズムの骨格と境界（契約）を固定するのが目的。
契約の正本: kuas-mechlab3 の docs/autonomy-plan.md §3。
"""

import argparse
import json
from pathlib import Path

from mechlab3_policy.sync import build_grid, previous_action_state, resample_latest

ACTION_TOPIC = "/cmd_norm"
FRONT_TOPIC = "/front_camera/image_raw/compressed"
CONTROL_HZ = 10.0
TASK = "follow the route"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, help="datasets/raw/<episode> のパス")
    p.add_argument(
        "--out", default="datasets/lerobot", help="LeRobot データセット出力先"
    )
    p.add_argument("--hz", type=float, default=CONTROL_HZ)
    p.add_argument("--camera-topic", default=FRONT_TOPIC)
    return p.parse_args(argv)


def read_meta(raw_dir: Path) -> dict:
    """meta.json を読む（label!=success は呼び出し側で除外する）。"""
    return json.loads((raw_dir / "meta.json").read_text(encoding="utf-8"))


def read_bag(bag_dir: Path, camera_topic: str):
    """bag から (action_ts, actions[(vx,wz)]) と (frame_ts, jpeg_bytes[]) を取り出す。

    TODO: rosbags で実装する。
        from rosbags.highlevel import AnyReader
        with AnyReader([bag_dir]) as reader:
            - /cmd_norm (TwistStamped): header.stamp -> t, (twist.linear.x, twist.angular.z)
            - camera_topic (CompressedImage): header.stamp -> t, msg.data (JPEG bytes)
    ここでは契約（戻り値の形）だけ固定する。
    """
    raise NotImplementedError("rosbags での bag デコードを実装する（Phase 3）")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = Path(args.raw_dir)
    meta = read_meta(raw_dir)
    if meta.get("label") != "success":
        print(f"skip: label={meta.get('label')}（success のみ採用）")
        return 0

    # --- bag からストリームを取り出す（TODO: read_bag を実装）---
    action_ts, actions, frame_ts, frames = read_bag(
        raw_dir / meta["bag_dir"], args.camera_topic
    )

    # --- 固定 Hz グリッドで同期（純ロジック、テスト済み）---
    t0, t1 = action_ts[0], action_ts[-1]
    grid = build_grid(t0, t1, args.hz)
    grid_actions = resample_latest(grid, action_ts, actions)
    grid_frames = resample_latest(grid, frame_ts, frames)
    # 最初のフレーム/行動が揃う tick まで先頭を捨てる
    pairs = [
        (f, a)
        for f, a in zip(grid_frames, grid_actions)
        if f is not None and a is not None
    ]
    states = previous_action_state([a for _, a in pairs])

    # --- LeRobot データセットへ書き出し（TODO: API はバージョン依存）---
    #   from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    #   ds = LeRobotDataset.create(repo_id=..., fps=args.hz, features={
    #       "observation.images.front": {...image...},
    #       "observation.state": {...shape (2,)...},
    #       "action": {...shape (2,)...},
    #   })
    #   for (jpeg, action), state in zip(pairs, states):
    #       ds.add_frame({"observation.images.front": decode(jpeg),
    #                     "observation.state": state,
    #                     "action": list(action), "task": TASK})
    #   ds.save_episode()
    raise NotImplementedError(
        f"同期まで完了（{len(pairs)} frames, {len(states)} states）。"
        "LeRobot 書き出しを実装する（Phase 3）"
    )


if __name__ == "__main__":
    raise SystemExit(main())

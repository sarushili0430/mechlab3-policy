"""Phase 3 の純ロジック: bag から取り出した時系列を、固定 Hz の学習フレーム列へ。

bag の読み出し（rosbags）・JPEG デコード・LeRobot 書き出しは ``convert_to_lerobot.py``
（IO）が担う。ここは「タイムスタンプと行動の配列 → 各 tick の (フレーム参照, action,
state)」だけを担当し、pytest で検証できる純ロジックに保つ（このリポの pure-vs-IO 分離）。
"""

from typing import NamedTuple, Sequence

from mechlab3_policy.sync import (
    build_grid,
    latest_index_at,
    previous_action_state,
    resample_latest,
)

Action = tuple[float, float]  # (vx, wz)


def stamp_to_seconds(sec: int, nanosec: int) -> float:
    """ROS の header.stamp (sec, nanosec) を float 秒へ。"""
    return sec + nanosec * 1e-9


class FrameRow(NamedTuple):
    """固定 Hz グリッドの 1 tick 分。

    frame_index は IO 側が保持する frame 配列のインデックス（画像本体はここには持たない）。
    """

    frame_index: int
    action: Action  # その tick で有効な (vx, wz)
    state: Action  # 直前 tick の行動（= proprioception 代替）


def build_frame_rows(
    action_ts: Sequence[float],
    actions: Sequence[Action],
    frame_ts: Sequence[float],
    hz: float,
) -> list[FrameRow]:
    """非同期な action / frame ストリームを固定 Hz の学習フレーム列へ揃える。

    各 tick で「その時刻以前で最新のフレーム」と「その時点で有効な行動」を対にし、
    両方が出揃った区間だけ残す（先頭の未確定区間は捨てる）。state は 1 tick 遅れの行動。
    未来のフレームは覗かない（``sync.latest_index_at`` がリークを防ぐ）。
    """
    if len(action_ts) != len(actions):
        raise ValueError("action_ts and actions length mismatch")
    if not action_ts or not frame_ts:
        return []

    t_start = max(action_ts[0], frame_ts[0])  # 両ストリームが出揃ってから
    t_end = action_ts[-1]  # 人間が指令を出していた区間まで
    if t_end < t_start:
        return []

    grid = build_grid(t_start, t_end, hz)
    grid_actions = resample_latest(grid, action_ts, list(actions))

    paired: list[tuple[int, Action]] = []
    for t, act in zip(grid, grid_actions):
        if act is None:
            continue
        frame_index = latest_index_at(frame_ts, t)
        if frame_index < 0:
            continue
        paired.append((frame_index, act))

    states = previous_action_state([act for _, act in paired])
    return [
        FrameRow(frame_index=fi, action=act, state=(s[0], s[1]))
        for (fi, act), s in zip(paired, states)
    ]

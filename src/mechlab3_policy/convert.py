"""bag→LeRobot 変換の純ロジック（重依存・IO なし＝単体テスト可能）。

このリポの pure-vs-IO 分離に従い、ここには「同期後の整形」と「スキーマ生成」という
純粋な変換だけを置く。実際の bag デコード（`rosbags`）・JPEG デコード（`cv2`）・
LeRobot 書き出し（`lerobot`）という重い IO は `convert_to_lerobot.py`（IO 層）が担う。

時刻同期の一次ロジックは `mechlab3_policy.sync`。ここはその上に薄く乗るだけ。
契約の正本: kuas-mechlab3 の docs/autonomy-plan.md §3。
"""

from typing import Any, Mapping, Sequence

from mechlab3_policy.sync import build_grid, previous_action_state, resample_latest


def stamp_to_seconds(sec: int, nanosec: int) -> float:
    """ROS `header.stamp`(sec, nanosec) を float 秒へ。同期グリッドの時刻軸に使う。"""
    return sec + nanosec * 1e-9


def is_trainable(meta: Mapping[str, Any]) -> bool:
    """学習に採用するのは `label==success` のみ（failure / unlabeled は除外）。"""
    return meta.get("label") == "success"


def sort_by_time(
    timestamps: Sequence[float], values: Sequence[Any]
) -> tuple[list[float], list[Any]]:
    """(timestamps, values) を時刻昇順へ揃える。

    bag のメッセージは記録時刻順に届き `header.stamp` 順とは限らないが、`sync` は
    昇順を前提に二分探索する。ここで header.stamp を基準に安定ソートして渡す。
    """
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values length mismatch")
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    return [timestamps[i] for i in order], [values[i] for i in order]


def align_streams(
    grid_frames: Sequence[Any], grid_actions: Sequence[Any]
) -> tuple[list[Any], list[Any]]:
    """None（最初のサンプル前）を落として frames/actions を対で揃える。

    `resample_latest` は最初のサンプルより前の tick だけ None を返す（以後は直近値を
    保持する）ので、実質は「両ストリームが揃う tick までの先頭トリム」。内部に穴は空かない。
    """
    frames: list[Any] = []
    actions: list[Any] = []
    for f, a in zip(grid_frames, grid_actions):
        if f is None or a is None:
            continue
        frames.append(f)
        actions.append(a)
    return frames, actions


def synced_episode(
    action_ts: Sequence[float],
    actions: Sequence[Any],
    frame_ts: Sequence[float],
    frames: Sequence[Any],
    hz: float,
) -> tuple[list[Any], list[Any], list[list[float]]]:
    """非同期な (行動, フレーム) 列を固定 Hz で同期し、学習用の 3 列を返す。

    戻り値 `(frames, actions, states)` は同じ長さ N（両ストリームが重なる区間だけ）:
      - frames[i]  : tick i 以前で最新のフレーム（未来を覗かない＝リーク防止）
      - actions[i] : tick i 時点の行動 (vx, wz)
      - states[i]  : 直前 tick の行動（先頭は [0,0]）＝ `observation.state`

    重なりが無い / 空の場合は ([], [], [])。
    """
    if not action_ts or not frame_ts:
        return [], [], []
    # 両ストリームが揃う区間だけを使う（外側は先頭/末尾の値が stale になるため）。
    t0 = max(action_ts[0], frame_ts[0])
    t1 = min(action_ts[-1], frame_ts[-1])
    if t1 < t0:
        return [], [], []
    grid = build_grid(t0, t1, hz)
    grid_actions = resample_latest(grid, action_ts, actions)
    grid_frames = resample_latest(grid, frame_ts, frames)
    aligned_frames, aligned_actions = align_streams(grid_frames, grid_actions)
    states = previous_action_state(aligned_actions)
    return aligned_frames, aligned_actions, states


def build_features(
    image_shape: Sequence[int],
    *,
    image_dtype: str = "video",
    state_names: Sequence[str] = ("vx", "wz"),
    action_names: Sequence[str] = ("vx", "wz"),
) -> dict[str, dict[str, Any]]:
    """LeRobot `LeRobotDataset.create(features=...)` 用のスキーマ（純 dict）。

    両モデル（ACT / SmolVLA）が食える union を出す:
      - observation.images.front : カメラ（既定 `video`＝mp4 エンコード）
      - observation.state        : 直前の行動 [vx, wz]（エンコーダ無しの proprioception 代替）
      - action                   : 行動 [vx, wz]（正規化 [-1,1]）

    `lerobot` を import せず素の dict を返すので CI（重依存なし）でも検証できる。
    """
    front: dict[str, Any] = {
        "dtype": image_dtype,
        "shape": tuple(int(x) for x in image_shape),
        "names": ["height", "width", "channels"],
    }
    state: dict[str, Any] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": list(state_names),
    }
    action: dict[str, Any] = {
        "dtype": "float32",
        "shape": (len(action_names),),
        "names": list(action_names),
    }
    return {
        "observation.images.front": front,
        "observation.state": state,
        "action": action,
    }

"""録画ストリームを固定レートの (観測, 行動) 列に揃える純ロジック。

torch / rosbags / IO 依存を持たないので単体テストできる（このリポの pure-vs-IO 分離）。

録画は非同期: カメラ ~30Hz、/cmd_norm ~20Hz で、各々が自分のタイムスタンプを持つ。
学習には固定レートのグリッドが要り、各 tick で「**その時刻以前で最新の**フレーム」と
「その時点で有効な行動」を対にする —— 未来のフレームを覗かない（リーク防止）。
"""

from bisect import bisect_right
from typing import Any, Sequence


def build_grid(t_start: float, t_end: float, hz: float) -> list[float]:
    """[t_start, t_end] を `hz` で刻んだ固定レートの時刻列（始点を含む）。"""
    if hz <= 0:
        raise ValueError("hz must be positive")
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")
    step = 1.0 / hz
    n = int((t_end - t_start) / step)
    return [t_start + i * step for i in range(n + 1)]


def latest_index_at(timestamps: Sequence[float], t: float) -> int:
    """`t` 以前で最新のサンプルの index。無ければ -1。

    `timestamps` は昇順前提。grid の各 tick で最新のフレーム/行動を取るために使う
    （未来を覗かない＝lookahead leakage を防ぐ）。
    """
    return bisect_right(timestamps, t) - 1


def resample_latest(
    grid: Sequence[float],
    timestamps: Sequence[float],
    values: Sequence[Any],
) -> list[Any]:
    """grid の各 tick につき「時刻が tick 以前で最新の値」を返す。

    最初のサンプルより前の tick は None（呼び出し側で捨てるか詰める）。
    """
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values length mismatch")
    out: list[Any] = []
    for t in grid:
        idx = latest_index_at(timestamps, t)
        out.append(values[idx] if idx >= 0 else None)
    return out


def previous_action_state(actions: Sequence[Sequence[float]]) -> list[list[float]]:
    """各 tick の `observation.state` = 直前 tick の行動（先頭は [0, 0]）。

    エンコーダ無しのキットなので proprioception の代わり。ACT・SmolVLA とも state
    入力を期待するため、行動の 1 tick 遅れを状態として与える。
    """
    state: list[list[float]] = [[0.0, 0.0]]
    for a in actions[:-1]:
        state.append([float(a[0]), float(a[1])])
    return state

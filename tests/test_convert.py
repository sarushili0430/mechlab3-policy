"""convert.py（bag→LeRobot 変換の純ロジック）の単体テスト。

重依存（rosbags / cv2 / lerobot）は使わない — ここは同期後の整形とスキーマ生成だけ。
"""

import pytest

from mechlab3_policy.convert import (
    align_streams,
    build_features,
    is_trainable,
    sort_by_time,
    stamp_to_seconds,
    synced_episode,
)


def test_stamp_to_seconds_combines_sec_and_nanosec() -> None:
    assert stamp_to_seconds(0, 0) == 0.0
    assert stamp_to_seconds(3, 500_000_000) == pytest.approx(3.5)
    assert stamp_to_seconds(10, 0) == 10.0


def test_is_trainable_only_success() -> None:
    assert is_trainable({"label": "success"}) is True
    assert is_trainable({"label": "failure"}) is False
    assert is_trainable({"label": "unlabeled"}) is False
    assert is_trainable({}) is False  # label 欠落は除外


def test_sort_by_time_orders_values_with_timestamps() -> None:
    ts, vals = sort_by_time([0.2, 0.0, 0.1], ["b", "a", "mid"])
    assert ts == [0.0, 0.1, 0.2]
    assert vals == ["a", "mid", "b"]


def test_sort_by_time_is_stable_for_equal_stamps() -> None:
    ts, vals = sort_by_time([1.0, 1.0], ["first", "second"])
    assert ts == [1.0, 1.0]
    assert vals == ["first", "second"]  # 同時刻は入力順を保つ


def test_sort_by_time_length_mismatch() -> None:
    with pytest.raises(ValueError):
        sort_by_time([0.0], ["x", "y"])


def test_align_streams_trims_leading_none() -> None:
    frames, actions = align_streams(
        [None, "f1", "f2"], [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]
    )
    assert frames == ["f1", "f2"]
    assert actions == [(0.1, 0.0), (0.2, 0.0)]


def test_align_streams_drops_tick_when_either_is_none() -> None:
    frames, actions = align_streams(["f0", None, "f2"], [None, "a1", "a2"])
    assert frames == ["f2"]
    assert actions == ["a2"]


def test_synced_episode_pairs_latest_frame_with_action() -> None:
    # 行動は各 tick に密、フレームは疎。10Hz グリッドで「その時刻以前で最新」を対にする。
    action_ts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    actions = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0), (0.6, 0.0)]
    frame_ts = [0.0, 0.2, 0.5]
    frames = ["fA", "fB", "fC"]

    sframes, sactions, states = synced_episode(
        action_ts, actions, frame_ts, frames, 10.0
    )

    assert len(sframes) == len(sactions) == len(states) == 6
    # tick 0.1 は fA（0.2 の fB は未来なので覗かない）、0.2-0.4 は fB、0.5 で fC。
    assert sframes == ["fA", "fA", "fB", "fB", "fB", "fC"]
    assert sactions == actions  # action_ts が grid tick と一致するのでそのまま
    # observation.state は行動の 1 tick 遅れ（先頭 [0,0]）。
    assert states[0] == [0.0, 0.0]
    assert states[1] == [0.1, 0.0]
    assert states[-1] == [0.5, 0.0]


def test_synced_episode_uses_only_overlap() -> None:
    # フレームが遅れて始まる → 重なり区間だけ使う（先頭に None / stale を詰めない）。
    action_ts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    actions = [(v / 10.0, 0.0) for v in range(7)]
    frame_ts = [0.2, 0.4, 0.6]
    frames = ["late0", "late1", "late2"]

    sframes, sactions, states = synced_episode(
        action_ts, actions, frame_ts, frames, 10.0
    )

    assert (
        sframes[0] == "late0"
    )  # 先頭のフレームは重なり開始のもの（None/stale ではない）
    assert len(sframes) == len(sactions) == len(states)
    assert all(f in {"late0", "late1", "late2"} for f in sframes)
    assert None not in sframes and None not in sactions


def test_synced_episode_empty_when_no_overlap() -> None:
    # 行動と映像の時間帯が重ならない。
    out = synced_episode([0.0, 0.1], [(0, 0), (0, 0)], [5.0, 5.1], ["x", "y"], 10.0)
    assert out == ([], [], [])


def test_synced_episode_empty_on_empty_input() -> None:
    assert synced_episode([], [], [0.0], ["f"], 10.0) == ([], [], [])
    assert synced_episode([0.0], [(0, 0)], [], [], 10.0) == ([], [], [])


def test_build_features_union_schema() -> None:
    feats = build_features((240, 320, 3))
    assert set(feats) == {
        "observation.images.front",
        "observation.state",
        "action",
    }
    front = feats["observation.images.front"]
    assert front["dtype"] == "video"
    assert front["shape"] == (240, 320, 3)
    assert front["names"] == ["height", "width", "channels"]

    for key in ("observation.state", "action"):
        assert feats[key]["dtype"] == "float32"
        assert feats[key]["shape"] == (2,)
        assert feats[key]["names"] == ["vx", "wz"]


def test_build_features_casts_numpy_like_shape_to_int() -> None:
    # numpy の img.shape（np.int 要素）を渡しても素の int tuple になる。
    feats = build_features([bool(1) and 240, 320, 3], image_dtype="image")
    front = feats["observation.images.front"]
    assert front["dtype"] == "image"
    assert front["shape"] == (240, 320, 3)
    assert all(type(x) is int for x in front["shape"])

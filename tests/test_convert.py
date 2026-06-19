"""convert.py（Phase 3 の純ロジック）の単体テスト。"""

from mechlab3_policy.convert import FrameRow, build_frame_rows, stamp_to_seconds


def test_stamp_to_seconds() -> None:
    assert stamp_to_seconds(5, 500_000_000) == 5.5
    assert stamp_to_seconds(0, 0) == 0.0


def test_build_frame_rows_aligns_action_and_frame() -> None:
    action_ts = [0.0, 0.1, 0.2, 0.3]
    actions = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0)]
    frame_ts = [0.05, 0.15, 0.25]  # フレームは行動より少し遅れて到着
    rows = build_frame_rows(action_ts, actions, frame_ts, hz=10.0)

    # grid は max(0.0,0.05)=0.05 から 0.3 まで 10Hz → 0.05 / 0.15 / 0.25
    assert [r.frame_index for r in rows] == [0, 1, 2]
    # 各 tick で「その時刻以前で最新の行動」
    assert [r.action for r in rows] == [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0)]
    # state は 1 tick 遅れの行動（先頭は [0,0]）
    assert [r.state for r in rows] == [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]


def test_build_frame_rows_drops_warmup_before_first_frame() -> None:
    # フレームが行動より大きく遅れて始まる → 先頭区間は捨てられる
    action_ts = [0.0, 0.1, 0.2]
    actions = [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0)]
    frame_ts = [0.15]
    rows = build_frame_rows(action_ts, actions, frame_ts, hz=10.0)
    assert len(rows) == 1
    assert rows[0] == FrameRow(frame_index=0, action=(0.2, 0.0), state=(0.0, 0.0))


def test_build_frame_rows_empty_streams() -> None:
    assert build_frame_rows([], [], [0.0], hz=10.0) == []
    assert build_frame_rows([0.0], [(0.0, 0.0)], [], hz=10.0) == []


def test_build_frame_rows_frames_after_last_action_give_no_rows() -> None:
    # 全フレームが最後の行動より後 → 対にできず空
    assert build_frame_rows([0.0], [(0.1, 0.0)], [1.0], hz=10.0) == []


def test_build_frame_rows_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_frame_rows([0.0, 0.1], [(0.0, 0.0)], [0.0], hz=10.0)

"""sync.py（純ロジック）の単体テスト。"""

from mechlab3_policy.sync import (
    build_grid,
    latest_index_at,
    previous_action_state,
    resample_latest,
)


def test_build_grid_inclusive_fixed_rate() -> None:
    grid = build_grid(0.0, 1.0, 10.0)
    assert len(grid) == 11  # 0.0 .. 1.0 inclusive at 10Hz
    assert grid[0] == 0.0
    assert abs(grid[-1] - 1.0) < 1e-9


def test_build_grid_rejects_bad_args() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_grid(0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        build_grid(1.0, 0.0, 10.0)


def test_latest_index_at_picks_prior_not_future() -> None:
    ts = [0.0, 0.1, 0.2, 0.3]
    assert latest_index_at(ts, 0.25) == 2  # 0.2 を採用（0.3 は未来）
    assert latest_index_at(ts, 0.3) == 3  # ちょうどは採用
    assert latest_index_at(ts, -0.1) == -1  # 最初より前は無し


def test_resample_latest_no_lookahead() -> None:
    grid = [0.0, 0.1, 0.2, 0.3]
    assert resample_latest(grid, [0.0, 0.2], ["a", "b"]) == ["a", "a", "b", "b"]


def test_resample_latest_pads_none_before_first() -> None:
    assert resample_latest([0.0], [1.0], ["x"]) == [None]


def test_resample_latest_length_mismatch() -> None:
    import pytest

    with pytest.raises(ValueError):
        resample_latest([0.0], [0.0, 1.0], ["x"])


def test_previous_action_state_is_one_tick_delayed() -> None:
    actions = [[0.5, 0.0], [0.6, -0.1], [0.7, 0.2]]
    assert previous_action_state(actions) == [[0.0, 0.0], [0.5, 0.0], [0.6, -0.1]]

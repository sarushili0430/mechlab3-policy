# mechlab3-policy

[kuas-mechlab3](https://github.com/sarushili0430/kuas-mechlab3) の teleop ロボットを自律化するための **学習・推論** リポジトリ。映像から `{vx, wz}` を生成し、**WebSocket で Pi に送って**走らせる（ACT / SmolVLA）。

**ROS 非依存**。ロボット本体（ROS2）とは別リポにして、重い torch / lerobot 依存をロボリポの軽量環境・Pi から切り離す。GPU 機に ROS は要らない。

## ロボリポとの境界（契約だけ・コード依存なし）

- **データ受け渡し（入力）**: `kuas-mechlab3` の `datasets/raw/<...>/{bag, meta.json}`。bag は `rosbags` で読む（ROS 不要）。スキーマ = ロボリポの `recording.py` / `docs/autonomy-plan.md §3`。
- **制御（出力）**: WS `{"vx","wz"}`（[-1,1]）→ `ws://<pi>:9001`。`websockets` で送る。人間のテレオプ・クライアントの差し替え。
- **計画の正本**: フェーズ別の詳細は **ロボリポの [`docs/autonomy-plan.md`](https://github.com/sarushili0430/kuas-mechlab3/blob/develop/docs/autonomy-plan.md)**。本リポはその Phase 3 / 5 / 7 を担う。

## 構成

```
src/mechlab3_policy/sync.py    # 純: 時刻同期 / 固定Hzリサンプル（pytest 済み）
src/mechlab3_policy/convert.py # 純: 同期後の整形 / LeRobot features 生成（pytest 済み）
tests/                        # 純ロジックのテスト（sync / convert）
convert_to_lerobot.py         # Phase 3: bag+meta → LeRobot 形式（rosbags 読み → LeRobot 書き）
policy_runner.py              # Phase 7: 映像 → policy → WS {vx,wz}（DummyPolicy 同梱）
pyproject.toml                # 実行時依存 + black/mypy/pytest/coverage 設定
requirements-dev.txt          # 開発ツール（black/mypy/pytest/lefthook/commitizen）
lefthook.yml                  # git フック（pre-commit / commit-msg）
.github/workflows/ci.yml      # CI（black + mypy + pytest。torch 非導入の軽量版）
```

## セットアップ（開発）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt    # lint / type / test ツール + lefthook
lefthook install                        # git フック（pre-commit / commit-msg）を配線
black --check . && mypy && pytest   # CI と同じ検査

# 変換 / 学習 / 推論を実際に回すときは実行時依存も（重い・GPU は別途）:
pip install -e .                        # torch / lerobot / rosbags / opencv / av
```

## CI

`.github/workflows/ci.yml` が push / PR で走る。**torch / lerobot は入れず**、`requirements-dev.txt`（black / mypy / pytest）だけで**純ロジック（`sync.py` / `convert.py`）を検査**する軽量構成:

- **format**: `black --check`
- **type**: `mypy --strict`（`src/mechlab3_policy` + `tests`）
- **test**: `pytest`（カバレッジを Job Summary へ）

変換の**純ロジックは `convert.py` に切り出して検査対象**。IO 層（`convert_to_lerobot.py` の bag デコード / LeRobot 書き出し、`policy_runner.py`）は重い依存を import するため CI 対象外 —— rosbags / lerobot 込みの結合確認はローカルで回す。

## bag → LeRobot 変換（Phase 3）

ロボリポが吐いた `datasets/raw/<episode>/{bag, meta.json}` を LeRobot データセットへ。
`label==success` のエピソードだけを 10Hz で同期して取り込む（両モデル対応の union）。

```bash
pip install -e .   # rosbags / opencv / av / lerobot（重い・GPU は別途）

# ルート配下の success エピソードをまとめて 1 データセットへ
python convert_to_lerobot.py --raw-root datasets/raw \
    --out datasets/lerobot --repo-id mechlab3/route --overwrite

# 個別エピソードを指定（複数可）
python convert_to_lerobot.py --raw-dir datasets/raw/<episode> --out datasets/lerobot
```

出力（`observation.images.front` + `observation.state`=直前の行動 + `action`=(vx,wz)
+ `task`）はそのまま Phase 5 の `lerobot` 学習（ACT / SmolVLA）に渡せる。同期の純ロジックは
`sync.py` / `convert.py`（pytest 済み）、bag デコード（rosbags）と LeRobot 書き出しは
`convert_to_lerobot.py`。

## まず WS 経路を疎通確認（モデル不要）

ロボ側で `start-all.sh`（driver+カメラ+teleop）を起動した状態で:

```bash
python policy_runner.py --pi <ラズパイIP> --dry-run   # 送信せず行動を表示
python policy_runner.py --pi <ラズパイIP>              # DummyPolicy（緩い直進）で実走 ※車輪を浮かせ人間監視
```

`DummyPolicy` を学習済みモデル（Phase 5 の checkpoint）に差し替えれば自律走行になる。

## フェーズ（詳細は autonomy-plan.md）

- **Phase 3** `convert_to_lerobot.py`: 録画 bag → LeRobot データセット（`success` のみ・10Hz 同期・両モデル対応の union）。
- **Phase 5** 学習: `lerobot` で **ACT**（`--policy.type=act`、ゼロ学習）/ **SmolVLA**（`--policy.path=lerobot/smolvla_base`、finetune）。同じデータを 1 フラグ切替。
- **Phase 7** `policy_runner.py`: 推論して WS 送信。アクションチャンクで遅延吸収、人間オーバーライド併設。

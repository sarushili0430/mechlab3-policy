#!/usr/bin/env python3
"""Phase 7 — この PC 上のポリシーで、WebSocket 越しにロボットを動かす。

`teleop_ws_client`（人間の WASD）の人間部分をモデルに差し替えたもの: カメラの最新
フレームを取り、ポリシーを回し、`{"vx","wz"}` を `ws://<pi>:9001` へ一定レートで送る。
Pi 側（teleop_server + ウォッチドッグ）は無変更 —— 人間と同じスロットを埋めるだけ。

`DummyPolicy`（定数の緩い前進）を同梱。モデルが無くても **WS 経路を端から端まで疎通**
確認できる。学習済みモデルは後で差し替える（Phase 5 の checkpoint）。

例:
    python policy_runner.py --pi 192.168.1.42 --dry-run      # 送信せず行動を print
    python policy_runner.py --pi 192.168.1.42                # DummyPolicy で実走（要・人間監視）
"""

import argparse
import json
import signal
import sys
import time

CONTROL_HZ = 10.0
FORWARD_VX = 0.15  # 正規化 [-1,1] の緩い前進


class DummyPolicy:
    """定数の緩い前進。WS 経路の疎通確認用であって、実走行用ではない。"""

    def select_action(self, obs: dict) -> list[tuple[float, float]]:
        return [(FORWARD_VX, 0.0)]


def load_policy(checkpoint: str | None):
    """checkpoint が無ければ DummyPolicy。あれば学習済みポリシー（TODO）。"""
    if checkpoint is None:
        return DummyPolicy()
    # TODO(Phase 5/7): LeRobot の学習済みポリシー（ACT / SmolVLA）を checkpoint からロード。
    #   from lerobot.common.policies.factory import make_policy / load from pretrained
    raise NotImplementedError(
        "実モデルのロードは Phase 5 の checkpoint と一緒に実装する。"
        "今は --checkpoint 無し（DummyPolicy）で WS 経路を確認する。"
    )


def mjpeg_url(pi: str, port: int, topic: str) -> str:
    return f"http://{pi}:{port}/stream?topic={topic}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pi", required=True, help="ラズパイの IP / ホスト（例 192.168.1.42）")
    p.add_argument("--ws-port", type=int, default=9001)
    p.add_argument("--cam-port", type=int, default=8080)
    p.add_argument("--topic", default="/front_camera/image_raw/compressed")
    p.add_argument("--checkpoint", default=None, help="学習済みモデル。未指定なら DummyPolicy（直進）")
    p.add_argument("--hz", type=float, default=CONTROL_HZ)
    p.add_argument("--dry-run", action="store_true", help="WS 送信せず行動を print だけ")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy(args.checkpoint)

    import cv2  # 遅延 import（--help は依存なしで出る）

    url = mjpeg_url(args.pi, args.cam_port, args.topic)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"カメラ MJPEG を開けない: {url}", file=sys.stderr)
        return 1

    ws = None
    if not args.dry_run:
        from websockets.sync.client import connect

        ws = connect(f"ws://{args.pi}:{args.ws_port}")

    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))

    prev = (0.0, 0.0)
    period = 1.0 / args.hz
    print(f"開始: {url} -> ws://{args.pi}:{args.ws_port}  ({'dry-run' if args.dry_run else '実走'}, {args.hz}Hz)")
    print("Ctrl+C で停止（終了時に停止指令を送る）。")
    try:
        while not stop["v"]:
            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                continue
            obs = {
                "observation.images.front": frame,
                "observation.state": [prev[0], prev[1]],
                "task": "follow the route",
            }
            vx, wz = policy.select_action(obs)[0]
            prev = (float(vx), float(wz))
            msg = json.dumps({"vx": float(vx), "wz": float(wz)})
            if args.dry_run:
                print(msg)
            else:
                ws.send(msg)
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        if ws is not None:
            try:
                ws.send(json.dumps({"vx": 0.0, "wz": 0.0}))  # 明示停止
            except Exception:
                pass
            ws.close()
        cap.release()
        print("\n停止しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

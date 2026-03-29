# app1.py
import os
import time
import json
import math
import threading

# # --- プロファイラを明示インストール（sitecustomize不要でも動く保険） ---
# try:
#     import pyhook_agent  # あなたが用意したエージェント
#     pyhook_agent.bootstrap()
# except Exception:
#     pass

def compute(x: int, y: int = 3) -> int:
    # いくつかの Python/C 呼び出しを混ぜてイベントを発生させる
    z = x * x + y
    # C関数呼び出し（c_call）も出したいので len/sorted/json を使う
    _ = len([1, 2, 3, 4])
    _ = sorted([3, 1, 2])
    _ = json.dumps({"x": x, "y": y, "z": z})
    return z

def nested(n: int) -> int:
    s = 0
    for i in range(n):
        s += compute(i, y=i % 5)
        time.sleep(0.01)  # スケジューリングイベントも少し
    return s

def worker(name: str, n: int = 5):
    tid = threading.get_native_id() if hasattr(threading, "get_native_id") else threading.get_ident()
    print(f"[worker {name}] start (tid={tid})")
    total = nested(n)
    print(f"[worker {name}] done total={total}")

def main():
    pid = os.getpid()
    tid = threading.get_native_id() if hasattr(threading, "get_native_id") else threading.get_ident()
    print(f"[main] pid={pid} tid={tid}")

    # INCLUDE/EXCLUDE を設定している場合に __main__ を拾えるよう注意
    # 例: export PYHOOK_INCLUDE="__main__"
    #     export PYHOOK_ENABLE=1
    #     export PYHOOK_SOCK=/tmp/pyhook.sock

    # メインスレッドでも少し呼ぶ
    _ = nested(5)

    # スレッドを2本起動
    t1 = threading.Thread(target=worker, args=("A", 6), daemon=True)
    t2 = threading.Thread(target=worker, args=("B", 7), daemon=True)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # 追加で少し計算
    v = sum(int(abs(math.sin(i)) * 100) for i in range(10))
    print(f"[main] extra={v}")
    print("[main] done")

if __name__ == "__main__":
    main()

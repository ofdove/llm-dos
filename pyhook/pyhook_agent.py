# pyhook_agent.py
import os, sys, time, json, socket, atexit, threading
import uuid

SOCK_PATH  = os.environ.get("PYHOOK_SOCK", "/tmp/pyhook.sock")
ENABLED    = os.environ.get("PYHOOK_ENABLE", "0") == "1"
MAX_BYTES  = int(os.environ.get("PYHOOK_MAX_BYTES", "8192"))

# 既定で「Torch/Dynamoまわりは触らない」= True
SAFE_FOR_TORCH = os.environ.get("PYHOOK_SAFE_FOR_TORCH", "1") == "1"
SKIP_GENERATORS = os.environ.get("PYHOOK_SKIP_GENERATORS", "1") == "1"

def _parse_list(envkey: str):
    s = os.environ.get(envkey)
    if not s: return ()
    return tuple(p.strip() for p in s.split(",") if p.strip())

INCLUDE = _parse_list("PYHOOK_INCLUDE")   # 例: "__main__,vllm,app."
EXCLUDE = _parse_list("PYHOOK_EXCLUDE")   # 追加除外があれば

# 既定の除外（Torch/Dynamo/Triton/コンパイル周辺）
DEFAULT_FILE_EXCLUDE = (
    "/site-packages/torch/",
    "/torch/_dynamo/",
    "/torch/_inductor/",
    "/site-packages/triton/",
    "/vllm/compilation/",
)

_sock = None
_tls_guard = threading.local()  # 再入防止

# code.co_flags の判定用
CO_GENERATOR       = 0x20
CO_COROUTINE       = 0x80
CO_ASYNC_GENERATOR = 0x200

def _connect_once():
    global _sock
    if _sock is not None:
        return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.setblocking(False)
        s.connect(SOCK_PATH)
        _sock = s
    except Exception:
        _sock = None

def _should_send(modname) -> bool:
    mn = modname if isinstance(modname, str) else ""
    if INCLUDE and not any(mn.startswith(p) for p in INCLUDE):
        return False
    if EXCLUDE and any(mn.startswith(p) for p in EXCLUDE):
        return False
    return True

def _file_excluded(filename: str) -> bool:
    if not isinstance(filename, str):
        return False
    if SAFE_FOR_TORCH and any(x in filename for x in DEFAULT_FILE_EXCLUDE):
        return True
    return False

def _send(obj: dict):
    if _sock is None:
        return
    try:
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        _sock.send(data[:MAX_BYTES])
    except Exception:
        pass

def _prof(frame, event, arg):
    # 何があっても例外は外へ出さない & 再入を防ぐ
    if getattr(_tls_guard, "busy", False):
        return _prof
    _tls_guard.busy = True
    try:
        # if event not in ("call", "c_call", "return", "c_return"):
        if event not in ("call", "c_call"):
            return _prof

        tid = (threading.get_native_id()
               if hasattr(threading, "get_native_id")
               else threading.get_ident())
        pid = os.getpid()
        ts = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        # ts  = time.time_ns()
        trace_id = uuid.uuid4()

        if event == "call":
            code = getattr(frame, "f_code", None)
            glbs = getattr(frame, "f_globals", None)
            mod  = glbs.get("__name__", "") if isinstance(glbs, dict) else ""

            if not _should_send(mod):
                return _prof

            filename = code.co_filename if code and isinstance(getattr(code, "co_filename", None), str) else ""
            if _file_excluded(filename):
                return _prof

            # 3) ジェネレーター/コルーチン/async generator は除外
            if SKIP_GENERATORS and code:
                flags = int(getattr(code, "co_flags", 0))
                if flags & (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR):
                    return _prof

            func_name = (code.co_name if code and isinstance(getattr(code, "co_name", None), str) else "<unknown>")
            if func_name != 'chat':
                return _prof
            firstline = int(getattr(code, "co_firstlineno", 0)) if code else 0

            _send({
                "ts": ts, "pid": pid, "tid": tid, "ev": "call",
                "mod": mod, "func": func_name, "file": filename, "line": firstline, "trace_id": str(trace_id),
            })

        # elif event == 'c_call':  # c_call
        #     # C関数名を必ず文字列化
        #     name = getattr(arg, "__name__", None)
        #     if not isinstance(name, str):
        #         try: name = str(arg)
        #         except Exception: name = "<cfunc>"
        #
        #     # C側も Torch 系はざっくり除外（名前ベース）
        #     if SAFE_FOR_TORCH and (name.startswith(("torch", "triton"))):
        #         return _prof
        #
        #     # INCLUDE/EXCLUDE を C関数名でも適用（任意）
        #     if INCLUDE and not any(name.startswith(p) for p in INCLUDE):
        #         return _prof
        #     if EXCLUDE and any(name.startswith(p) for p in EXCLUDE):
        #         return _prof
        #
        #     _send({"ts": ts, "pid": pid, "tid": tid, "ev": "c_call", "func": name, "trace_id": str(trace_id),})
        # elif event == "return":
        #     code = getattr(frame, "f_code", None)
        #     glbs = getattr(frame, "f_globals", None)
        #     mod  = glbs.get("__name__", "") if isinstance(glbs, dict) else ""
        #
        #     if not _should_send(mod):
        #         return _prof
        #
        #     filename = code.co_filename if code and isinstance(getattr(code, "co_filename", None), str) else ""
        #     if _file_excluded(filename):
        #         return _prof
        #
        #     # 3) ジェネレーター/コルーチン/async generator は除外
        #     if SKIP_GENERATORS and code:
        #         flags = int(getattr(code, "co_flags", 0))
        #         if flags & (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR):
        #             return _prof
        #
        #     func_name = (code.co_name if code and isinstance(getattr(code, "co_name", None), str) else "<unknown>")
        #     if func_name != 'chat':
        #         return _prof
        #     firstline = int(getattr(code, "co_firstlineno", 0)) if code else 0
        #
        #     _send({
        #         "ts": ts, "pid": pid, "tid": tid, "ev": "return",
        #         "mod": mod, "func": func_name, "file": filename, "line": firstline, "trace_id": str(trace_id),
        #     })
        else:
            pass

    except Exception:
        pass
    finally:
        _tls_guard.busy = False
    return _prof

def install():
    if not ENABLED:
        return
    _connect_once()
    sys.setprofile(_prof)
    threading.setprofile(_prof)
    atexit.register(lambda: (_sock and _sock.close()))

def bootstrap():
    install()

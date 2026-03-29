import os, socket, json, ctypes

libbpf_path = os.environ.get("PYHOOK_LIBBPF", "libbpf.so")
LIBBPF = ctypes.CDLL(libbpf_path)
bpf_obj_get = LIBBPF.bpf_obj_get
bpf_obj_get.argtypes = [ctypes.c_char_p]
bpf_obj_get.restype = ctypes.c_int

bpf_map_update_elem = LIBBPF.bpf_map_update_elem
bpf_map_update_elem.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong]
bpf_map_update_elem.restype = ctypes.c_int

bpf_map_delete_elem = LIBBPF.bpf_map_delete_elem
bpf_map_delete_elem.argtypes = [ctypes.c_int, ctypes.c_void_p]
bpf_map_delete_elem.restype = ctypes.c_int

MAP_PATH = b"/sys/fs/bpf/active_traces"
MAP_FD = bpf_obj_get(MAP_PATH)
if MAP_FD < 0:
    raise RuntimeError("cannot open pinned BPF map at /sys/fs/bpf/active_traces (loaderでpinしてください)")

class TraceID(ctypes.Structure):
    _fields_ = [("hi", ctypes.c_uint64), ("lo", ctypes.c_uint64)]

def hex32_to_hi_lo(h: str):
    h = h.replace("-", "")
    x = int(h, 16)
    hi = (x >> 64) & ((1<<64)-1)
    lo = x & ((1<<64)-1)
    return hi, lo

sock_path = os.environ.get("PYHOOK_SOCK", "/tmp/pyhook.sock")
if os.path.exists(sock_path):
    os.unlink(sock_path)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(sock_path)
os.chmod(sock_path, 0o666)

print(f"[collector] listening on {sock_path}")
while True:
    data, _ = sock.recvfrom(64 * 1024)
    try:
        evt = json.loads(data.decode(errors="ignore"))

        if evt['ev'] == 'call':
            hi, lo = hex32_to_hi_lo(evt['trace_id'])
            tid = evt['tid']
            key = ctypes.c_uint32(tid)
            val = TraceID(hi, lo)
            bpf_map_update_elem(MAP_FD, ctypes.byref(key), ctypes.byref(val), ctypes.c_ulonglong(0))
        # elif evt['ev'] == 'return':
        #     tid = evt['tid']
        #     key = ctypes.c_uint32(tid)
        #     bpf_map_delete_elem(MAP_FD, ctypes.byref(key))
        print(evt)
    except Exception:
        pass

"""Small versioned JSON frames; SCM_RIGHTS carries arenas, never tensor objects."""
import array
import hashlib
import json
import math
import os
import socket
import struct
import sys
import uuid
from pathlib import Path

PROTOCOL_VERSION = 1
LAYOUT_VERSION = 1
MAX_FRAME = 16 * 1024 * 1024
ALIGNMENT = 4096
DTYPES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8,
          "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


class SharedModelError(RuntimeError):
    pass


def normalize_device_uuid(value):
    """Torch uses a bare UUID; NVML prefixes the same physical GPU with GPU-."""
    if not isinstance(value, str):
        raise SharedModelError('GPU device UUID must be a string')
    raw = value.strip()
    if raw[:4].lower() == 'gpu-':
        raw = raw[4:]
    try:
        return str(uuid.UUID(raw))
    except (ValueError, AttributeError) as exc:
        raise SharedModelError(f'Invalid physical GPU device UUID: {value!r}') from exc


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_socket():
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return str(Path(runtime) / "aitk-shared-models" / "broker.sock")


def process_identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    # comm can contain spaces and ')' characters.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def same_process(identity):
    try:
        return process_identity(identity["pid"]) == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


def send_frame(sock, message, fd=None):
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if len(payload) > MAX_FRAME:
        raise SharedModelError("Protocol frame exceeds metadata limit")
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    header = struct.pack("!I", len(payload))
    sent = sock.sendmsg([header], ancillary)
    if sent < len(header):
        sock.sendall(header[sent:])
    sock.sendall(payload)


def recv_frame(sock):
    header, ancillary, flags, _ = sock.recvmsg(4, socket.CMSG_SPACE(4 * 4))
    received = []
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds = array.array("i")
            fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
            received.extend(fds)
    try:
        if not header:
            raise EOFError("Broker connection closed")
        while len(header) < 4:
            chunk = sock.recv(4 - len(header))
            if not chunk:
                raise EOFError("Incomplete protocol header")
            header += chunk
        length, = struct.unpack("!I", header)
        if flags & socket.MSG_CTRUNC or len(received) > 1 or length > MAX_FRAME:
            raise SharedModelError("Invalid descriptor or oversized frame")
        payload = bytearray()
        while len(payload) < length:
            chunk = sock.recv(min(1024 * 1024, length - len(payload)))
            if not chunk:
                raise EOFError("Incomplete protocol frame")
            payload.extend(chunk)
        result = json.loads(payload)
        if not isinstance(result, dict):
            raise SharedModelError("Protocol message must be an object")
        for fd in received:
            os.set_inheritable(fd, False)
        return result, received[0] if received else None
    except BaseException:
        for fd in received:
            os.close(fd)
        raise


def validate_manifest(manifest, file_size=None):
    if manifest.get("protocol_version") != PROTOCOL_VERSION or manifest.get("layout_version") != LAYOUT_VERSION:
        raise SharedModelError("Unsupported shared model protocol/layout")
    if manifest.get("endianness") != "little" or sys.byteorder != "little":
        raise SharedModelError("Only little-endian storage is supported")
    size = manifest.get("arena_bytes")
    if type(size) is not int or size <= 0 or (file_size is not None and size != file_size):
        raise SharedModelError("Invalid arena extent")
    for required in ("arena_uuid", "content_digest", "config_digest", "variant", "quantization"):
        if not isinstance(manifest.get(required), str) or not manifest[required]:
            raise SharedModelError(f"Missing manifest field {required}")
    if len(manifest["content_digest"]) != 64 or len(manifest["config_digest"]) != 64:
        raise SharedModelError("Invalid content/config digest")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise SharedModelError("Empty tensor table")
    occupied = []
    for name, desc in tensors.items():
        if not isinstance(name, str) or not isinstance(desc, dict):
            raise SharedModelError("Invalid tensor descriptor")
        width = DTYPES.get(desc.get("dtype"))
        shape, strides = desc.get("shape"), desc.get("strides")
        if width is None or not isinstance(shape, list) or not isinstance(strides, list) or len(shape) != len(strides):
            raise SharedModelError(f"{name}: invalid dtype/shape/strides")
        if any(type(x) is not int or x < 0 for x in shape + strides):
            raise SharedModelError(f"{name}: negative or noninteger dimensions")
        offset, length = desc.get("offset"), desc.get("length")
        if type(offset) is not int or type(length) is not int or offset < 0 or length < 0 or offset % width:
            raise SharedModelError(f"{name}: invalid byte extent/alignment")
        span = 0 if math.prod(shape) == 0 else (1 + sum((n - 1) * s for n, s in zip(shape, strides))) * width
        if span > length or offset + length > size:
            raise SharedModelError(f"{name}: out-of-bounds tensor extent")
        if not desc.get("alias_of"):
            occupied.append((offset, offset + length, name))
    occupied.sort()
    for left, right in zip(occupied, occupied[1:]):
        if left[1] > right[0]:
            raise SharedModelError(f"Unmarked overlapping tensor ranges: {left[2]}, {right[2]}")
    for name, desc in tensors.items():
        alias = desc.get("alias_of")
        if alias:
            master = tensors.get(alias)
            if master is None or master.get("alias_of") or desc["offset"] < master["offset"] or desc["offset"] + desc["length"] > master["offset"] + master["length"]:
                raise SharedModelError(f"{name}: invalid alias range")
    markers = manifest.get("quantization_markers", {})
    if not isinstance(markers, dict) or any(not isinstance(v, dict) or v.get("format") != manifest["quantization"] for v in markers.values()):
        raise SharedModelError("Invalid quantization marker table")
    return manifest


def require_compatible(manifest, expected):
    validate_manifest(manifest)
    for key, value in expected.items():
        if value is not None and manifest.get(key) != value:
            raise SharedModelError(f"Shared model compatibility mismatch: {key}")

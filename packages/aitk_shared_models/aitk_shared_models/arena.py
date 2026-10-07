"""Bounded safetensors-to-arena construction. No model or torch imports here."""
import fcntl
import ctypes
import hashlib
import json
import math
import mmap
import os
import struct
import tempfile
import uuid
from pathlib import Path

from .diagnostics import memory_available
from .protocol import ALIGNMENT, DTYPES, LAYOUT_VERSION, PROTOCOL_VERSION, SharedModelError, digest_json, validate_manifest

CHUNK_BYTES = 8 * 1024 * 1024
MAX_HEADER = 32 * 1024 * 1024
# Some Conda Python builds omit Linux constants despite supporting memfd.
F_ADD_SEALS = getattr(fcntl, 'F_ADD_SEALS', 1033)
F_GET_SEALS = getattr(fcntl, 'F_GET_SEALS', 1034)
SEALS = 8 | 4 | 2 | 1  # Linux UAPI: WRITE, GROW, SHRINK, SEAL


def create_memfd():
    if hasattr(os, 'memfd_create'):
        return os.memfd_create('aitk-shared-model', 3)  # CLOEXEC | ALLOW_SEALING
    # Conda's Python 3.11 build on this host omits os.memfd_create. The libc
    # entrypoint has the same kernel behavior; this is not a named /dev/shm file.
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, 'memfd_create', None)
    if function is None:
        raise OSError('memfd_create is unavailable; configure a local file fallback')
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(b'aitk-shared-model', 3)
    if fd < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return fd


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SharedModelError(f"Duplicate checkpoint key {key}")
        result[key] = value
    return result


def checkpoint_layout(path):
    """Read only headers; return raw source ranges, preserving expert banks."""
    path = Path(path).resolve(strict=True)
    if path.is_dir():
        index = json.loads((path / "model.safetensors.index.json").read_text(), object_pairs_hook=_unique)
        mapping = index["weight_map"]
        files = sorted({path / name for name in mapping.values()})
    else:
        mapping, files = None, [path]
    table, metadata, provenance = {}, {}, []
    for source in files:
        source = source.resolve(strict=True)
        stat = source.stat()
        with source.open("rb", buffering=0) as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise SharedModelError("Incomplete safetensors header")
            size, = struct.unpack("<Q", prefix)
            if size > MAX_HEADER or size + 8 > stat.st_size:
                raise SharedModelError("Invalid safetensors header size")
            header = json.loads(handle.read(size), object_pairs_hook=_unique)
        metadata.update(header.pop("__metadata__", {}))
        for name, item in header.items():
            if mapping is not None and mapping.get(name) != os.path.relpath(source, path):
                # Resolve symlinked shard names as well.
                mapped = mapping.get(name)
                if mapped is None or (path / mapped).resolve() != source:
                    raise SharedModelError(f"Index disagrees with shard for {name}")
            if name in table:
                raise SharedModelError(f"Duplicate tensor {name}")
            dtype, shape = item["dtype"], item["shape"]
            if dtype not in DTYPES or any(type(n) is not int or n < 0 for n in shape):
                raise SharedModelError(f"Unsupported storage descriptor {name}")
            begin, end = item["data_offsets"]
            if type(begin) is not int or type(end) is not int or begin < 0 or end < begin or end + size + 8 > stat.st_size or end - begin != math.prod(shape) * DTYPES[dtype]:
                raise SharedModelError(f"Invalid source extent for {name}")
            table[name] = {"path": str(source), "source_offset": 8 + size + begin,
                           "dtype": dtype, "shape": shape, "length": end - begin}
        provenance.append({"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "device": stat.st_dev, "inode": stat.st_ino})
    if mapping is not None and set(mapping) != set(table):
        raise SharedModelError("Incomplete shard index")
    # Reject overlapping source ranges too (malformed headers cannot alias silently).
    for source in files:
        ranges = sorted((v["source_offset"], v["source_offset"] + v["length"]) for v in table.values() if v["path"] == str(source.resolve()))
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise SharedModelError("Overlapping checkpoint ranges")
    return table, metadata, provenance


def source_identity(path, config, variant, quantization):
    _, metadata, provenance = checkpoint_layout(path)
    # Lookup identity only. Published content identity is computed from all stored bytes.
    return digest_json({"sources": provenance, "config": config, "variant": variant,
                        "quantization": quantization, "layout": LAYOUT_VERSION}), metadata


def build_checkpoint(path, config, variant="instruct", quantization="int8_tensorwise",
                     reserve_bytes=12 * 1024**3, fallback_directory=None,
                     skip_prefixes=("vae.", "vision_model.", "lm_head.", "model.ln_f.")):
    if variant != "instruct" or quantization != "int8_tensorwise":
        raise SharedModelError("Initial shared store supports Instruct int8 only")
    sources, metadata, provenance = checkpoint_layout(path)
    tensors, total = {}, 0
    for name, source in sorted(sources.items()):
        if name.startswith(skip_prefixes) or name.endswith(".comfy_attention.config"):
            continue
        total = (total + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        shape, stride = source["shape"], []
        running = 1
        for n in reversed(shape):
            stride.insert(0, running)
            running *= n
        tensors[name] = {k: source[k] for k in ("dtype", "shape", "length")}
        tensors[name].update(offset=total, strides=stride)
        total += source["length"]
    total = max(ALIGNMENT, (total + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT)
    if memory_available() < total + reserve_bytes:
        raise SharedModelError("Shared arena would violate MemAvailable reserve; no private fallback")
    # cgroup v2 can impose a tighter bound than the machine's free memory.
    cgroup = Path("/proc/self/cgroup").read_text().splitlines()
    for line in cgroup:
        if line.startswith("0::"):
            group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            try:
                limit = (group / "memory.max").read_text().strip()
                current = int((group / "memory.current").read_text())
                if limit != "max" and int(limit) - current < total + reserve_bytes:
                    raise SharedModelError("Shared arena exceeds cgroup memory budget")
            except FileNotFoundError:
                pass
    temp_path = None
    try:
        try:
            fd = create_memfd()
            backing = "sealed_memfd"
        except OSError:
            if fallback_directory is None:
                raise
            directory = Path(fallback_directory).resolve(strict=True)
            if str(directory).startswith("/dev/shm"):
                raise SharedModelError("/dev/shm is not a full-backbone fallback")
            if os.statvfs(directory).f_bavail * os.statvfs(directory).f_frsize < total:
                raise SharedModelError("Insufficient fallback disk space")
            fd, temp_path = tempfile.mkstemp(prefix=".aitk-arena-", dir=directory)
            backing = "readonly_file"
        os.ftruncate(fd, total)
        digest, markers, reads = hashlib.sha256(), {}, 0
        for name, desc in tensors.items():
            source = sources[name]
            digest.update(json.dumps([name, desc["dtype"], desc["shape"]], separators=(",", ":")).encode())
            with open(source["path"], "rb", buffering=0) as handle:
                handle.seek(source["source_offset"])
                copied = 0
                marker = bytearray() if name.endswith(".comfy_quant") else None
                if marker is not None and desc["length"] > 65536:
                    raise SharedModelError("Oversized quantization marker")
                while copied < desc["length"]:
                    chunk = handle.read(min(CHUNK_BYTES, desc["length"] - copied))
                    if not chunk:
                        raise SharedModelError(f"Interrupted checkpoint read: {name}")
                    digest.update(chunk)
                    if marker is not None:
                        marker.extend(chunk)
                    written = 0
                    while written < len(chunk):
                        n = os.pwrite(fd, chunk[written:], desc["offset"] + copied + written)
                        if n <= 0:
                            raise SharedModelError("Incomplete arena write")
                        written += n
                    copied += len(chunk)
                    reads += len(chunk)
                if marker is not None:
                    conf = json.loads(marker)
                    if not isinstance(conf, dict) or conf.get("format") != quantization:
                        raise SharedModelError(f"{name}: unsupported quantization marker")
                    markers[name[:-len(".comfy_quant")]] = conf
        for item in provenance:
            stat = os.stat(item["path"])
            if [stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino] != [item[k] for k in ("size", "mtime_ns", "device", "inode")]:
                raise SharedModelError("Checkpoint changed during arena build")
        if not markers:
            raise SharedModelError("Shared Instruct store requires prequantized int8 weights")
        manifest = {"protocol_version": PROTOCOL_VERSION, "layout_version": LAYOUT_VERSION,
                    "endianness": "little", "alignment": ALIGNMENT, "arena_uuid": str(uuid.uuid4()),
                    "content_digest": digest.hexdigest(), "config_digest": digest_json(config), "config": config,
                    "variant": variant, "quantization": quantization, "quantization_markers": markers,
                    "arena_bytes": total, "tensors": tensors, "source_metadata": metadata,
                    "provenance": provenance, "backing": backing, "checkpoint_bytes_read": reads}
        validate_manifest(manifest, total)
        if backing == "sealed_memfd":
            fcntl.fcntl(fd, F_ADD_SEALS, SEALS)
        else:
            os.fsync(fd)
            os.fchmod(fd, 0o400)
        readonly = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
        os.close(fd)
        fd = None
        if temp_path:
            os.unlink(temp_path)
            temp_path = None
        return manifest, readonly
    except BaseException:
        if "fd" in locals() and fd is not None:
            os.close(fd)
        if temp_path:
            os.unlink(temp_path)
        raise


class Arena:
    """Read-only mapping; tensor views retain the owner independently of attributes."""
    def __init__(self, manifest, fd):
        validate_manifest(manifest, os.fstat(fd).st_size)
        if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
            raise SharedModelError("Arena descriptor must be read-only")
        if manifest.get("backing") == "sealed_memfd" and fcntl.fcntl(fd, F_GET_SEALS) & SEALS != SEALS:
            raise SharedModelError("Arena lacks immutable seals")
        self.manifest = manifest
        self.fd = fd
        self.mapping = mmap.mmap(fd, manifest["arena_bytes"], access=mmap.ACCESS_READ)
        self._tensor_owner = None

    def tensor(self, name, slice_spec=None):
        from .tensors import tensor_view
        return tensor_view(self, name, slice_spec)

    def close(self):
        # Explicit close is refused while any exported tensor exists. Keeping a cached
        # model attached is legitimate; closing its socket must not invalidate its views.
        if self._tensor_owner is not None and self._tensor_owner.has_views():
            raise SharedModelError("Arena still has live tensor views")
        self.mapping.close()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __del__(self):
        try:
            if self.fd is not None:
                os.close(self.fd)
        except (AttributeError, OSError):
            pass

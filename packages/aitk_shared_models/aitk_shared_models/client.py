"""Synchronous local client. Waiting is polled and cancellable; timeout never grants GPU."""
import contextlib
import os
import socket
import threading
import time
import uuid

from .arena import Arena, build_checkpoint, source_identity
from .protocol import PROTOCOL_VERSION, SharedModelError, default_socket, process_identity, recv_frame, require_compatible, send_frame, normalize_device_uuid


class Client:
    def __init__(self, socket_path=None, role="observer", timeout=10):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.lock = threading.RLock()
        self.role = role
        self.leases = {}
        self.arenas = {}
        try:
            self.socket.connect(socket_path or default_socket())
            response, _ = self.call("hello", protocol_version=PROTOCOL_VERSION,
                                    identity=process_identity(), role=role)
            self.client_id = response["client_id"]
        except BaseException:
            self.socket.close()
            raise

    def call(self, op, fd=None, **fields):
        if 'device_uuid' in fields:
            fields['device_uuid'] = normalize_device_uuid(fields['device_uuid'])
        with self.lock:
            send_frame(self.socket, dict(fields, op=op), fd)
            response, received = recv_frame(self.socket)
            if not response.get("ok"):
                if received is not None:
                    os.close(received)
                raise SharedModelError(response.get("error", "Broker refused request"))
            return response, received

    def status(self):
        return self.call("status")[0]

    def open_store(self, identity, expected=None):
        if identity in self.arenas:
            arena = self.arenas[identity]
            require_compatible(arena.manifest, expected or {})
            return arena
        response, fd = self.call("open_store", identity=identity)
        if fd is None:
            raise SharedModelError("Store response lacks arena descriptor")
        try:
            require_compatible(response["manifest"], expected or {})
            arena = Arena(response["manifest"], fd)
            self.arenas[identity] = arena
            return arena
        except BaseException:
            os.close(fd)
            raise

    def load_checkpoint(self, path, config, variant="instruct", identity=None,
                        reserve_bytes=12 * 1024**3, fallback_directory=None,
                        cancel_check=None, timeout=3600):
        calculated, _ = source_identity(path, config, variant, "int8_tensorwise")
        # Configured names are handles, not replacements for content/config checks.
        identity = identity or calculated
        deadline = time.monotonic() + timeout
        while True:
            if cancel_check and cancel_check():
                raise InterruptedError("Shared store wait cancelled")
            response, _ = self.call("begin_store", identity=identity)
            state = response["state"]
            if state == "READY":
                break
            if state == "ERROR":
                raise SharedModelError(response["error"])
            if state == "BUILD":
                try:
                    manifest, fd = build_checkpoint(path, config, variant, reserve_bytes=reserve_bytes,
                                                    fallback_directory=fallback_directory)
                    try:
                        manifest["source_lookup_identity"] = calculated
                        self.call("publish_store", identity=identity, manifest=manifest, fd=fd)
                    finally:
                        os.close(fd)
                except BaseException as exc:
                    try:
                        self.call("fail_store", identity=identity, error=str(exc))
                    except (OSError, SharedModelError):
                        pass
                    raise
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Shared store construction wait timed out; no independent load")
            time.sleep(0.1)
        from .protocol import digest_json
        arena = self.open_store(identity, {"variant": variant, "quantization": "int8_tensorwise",
                                           "config_digest": digest_json(config), "source_lookup_identity": calculated})
        return identity, arena

    def request_gpu(self, device_uuid, request_id):
        device_uuid = normalize_device_uuid(device_uuid)
        response, _ = self.call("acquire_gpu", device_uuid=device_uuid, request_id=request_id)
        if response["granted"]:
            self.leases[device_uuid] = {"request_id": request_id, "epoch": response["epoch"]}
        return response

    def acquire_gpu(self, device_uuid, request_id=None, cancel_check=None, timeout=None):
        request_id = request_id or str(uuid.uuid4())
        start = time.monotonic()
        try:
            while True:
                if cancel_check and cancel_check():
                    raise InterruptedError("GPU lease wait cancelled")
                response = self.request_gpu(device_uuid, request_id)
                if response["granted"]:
                    return response["epoch"]
                if timeout is not None and time.monotonic() - start >= timeout:
                    raise TimeoutError("GPU lease wait timed out; owner remains exclusive")
                time.sleep(0.1)
        except BaseException:
            self.call("cancel_request", device_uuid=device_uuid, request_id=request_id)
            raise

    def poll_gpu(self, device_uuid):
        return self.call("poll_gpu", device_uuid=device_uuid)[0]

    def ack_quiescent(self, device_uuid, epoch=None):
        device_uuid = normalize_device_uuid(device_uuid)
        current = self.leases.get(device_uuid)
        epoch = current["epoch"] if epoch is None and current else epoch
        self.call("ack_quiescent", device_uuid=device_uuid, epoch=epoch)
        self.leases.pop(device_uuid, None)

    def publish_adapter(self, metadata):
        return self.call("publish_adapter", metadata=metadata)[0]["snapshot_id"]

    def open_adapter(self, snapshot_id, store_identity):
        return self.call("open_adapter", snapshot_id=snapshot_id, store_identity=store_identity)[0]["metadata"]

    def release_adapter(self, snapshot_id):
        self.call("release_adapter", snapshot_id=snapshot_id)

    def release_client(self):
        # Mapping owners remain valid after socket close. An active GPU lease is
        # intentionally NOT acknowledged here; its caller must prove quiescence.
        self.socket.close()
        self.arenas.clear()

    close = release_client

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release_client()


class SharedTensorSource:
    def __init__(self, arena):
        self.arena = arena
        self.keys = set(arena.manifest["tensors"])
        self.consumed = set()
        self.shared = True

    def shape(self, key):
        return list(self.arena.manifest["tensors"][key]["shape"])

    def tensor(self, key, expert=None):
        self.consumed.add(key)
        return self.arena.tensor(key, expert)

    def metadata(self):
        return self.arena.manifest.get("source_metadata", {})

    def close(self):
        # The model, Parameter aliases and transfer owner retain the mapping.
        pass

"""Owner-only local store/lease broker. No torch, model or CUDA imports."""
import argparse
import collections
import fcntl
import os
import signal
import socket
import struct
import threading
import uuid
from pathlib import Path

from .arena import SEALS, F_GET_SEALS
from .protocol import (PROTOCOL_VERSION, SharedModelError, default_socket, process_identity,
                       recv_frame, send_frame, same_process, validate_manifest, normalize_device_uuid)


class Broker:
    def __init__(self, socket_path=None):
        self.socket_path = socket_path or default_socket()
        self.lock = threading.RLock()
        self.clients = {}
        self.stores = {}
        self.contents = {}
        self.devices = {}
        self.snapshots = {}
        self.listener = None
        self.stopping = threading.Event()
        self.threads = []

    def serve_forever(self):
        path = Path(self.socket_path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        stat = path.parent.stat()
        if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise SharedModelError("Broker directory must be owner-only")
        # Never unlink an existing socket: it could be a surviving live broker.
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        os.chmod(path, 0o600)
        self.listener.listen(16)
        self.listener.settimeout(0.2)
        try:
            while not self.stopping.is_set():
                try:
                    connection, _ = self.listener.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(target=self._serve_client, args=(connection,), daemon=True)
                self.threads.append(thread)
                thread.start()
        finally:
            self.listener.close()
            path.unlink(missing_ok=True)

    def stop(self):
        self.stopping.set()
        with self.lock:
            for client in list(self.clients.values()):
                try:
                    client["socket"].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        self.stop()
        for thread in self.threads:
            thread.join(timeout=1)
        with self.lock:
            for store in self.contents.values():
                os.close(store["fd"])
            self.contents.clear()

    def _serve_client(self, connection):
        client_id = None
        try:
            pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != os.getuid():
                raise SharedModelError("Peer UID differs from broker owner")
            hello, fd = recv_frame(connection)
            if fd is not None:
                os.close(fd)
                raise SharedModelError("Unexpected hello descriptor")
            if hello.get("op") != "hello" or hello.get("protocol_version") != PROTOCOL_VERSION or hello.get("identity") != process_identity(pid):
                raise SharedModelError("Peer protocol/process identity mismatch")
            if hello.get("role") not in ("trainer", "comfy", "observer"):
                raise SharedModelError("Unknown client role")
            client_id = str(uuid.uuid4())
            with self.lock:
                self.clients[client_id] = {"identity": hello["identity"], "role": hello["role"],
                                           "socket": connection, "stores": set(), "snapshots": set()}
            send_frame(connection, {"ok": True, "client_id": client_id, "protocol_version": PROTOCOL_VERSION})
            while not self.stopping.is_set():
                request, fd = recv_frame(connection)
                try:
                    with self.lock:
                        response, outgoing = self.dispatch(client_id, request, fd)
                    send_frame(connection, dict(response, ok=True), outgoing)
                except (SharedModelError, ValueError, KeyError, OSError) as exc:
                    send_frame(connection, {"ok": False, "error": str(exc)})
                finally:
                    if fd is not None:
                        os.close(fd)
        except (EOFError, OSError, SharedModelError):
            pass
        finally:
            with self.lock:
                if client_id is not None:
                    self._disconnect(client_id)
            connection.close()

    def _disconnect(self, client_id):
        client = self.clients.pop(client_id, None)
        if client is None:
            return
        for store in self.stores.values():
            if store.get("builder") == client_id and store["state"] == "BUILDING":
                store.update(state="ERROR", error="Builder disconnected before READY")
            store.get("clients", set()).discard(client_id)
        for device in self.devices.values():
            device["queue"] = collections.deque(r for r in device["queue"] if r["client"] != client_id)
            if device["owner"] and device["owner"]["client"] == client_id:
                # Disconnect/timeout proves nothing about outstanding CUDA work.
                device["fault"] = dict(device["owner"], identity=client["identity"], reason="GPU owner disconnected without quiescence")
                device["owner"] = None
        for snapshot in self.snapshots.values():
            snapshot["clients"].discard(client_id)

    def _device(self, device_uuid):
        device_uuid = normalize_device_uuid(device_uuid)
        return self.devices.setdefault(device_uuid, {"owner": None, "epoch": 0, "queue": collections.deque(), "fault": None})

    def dispatch(self, client_id, request, fd=None):
        op, client = request.get("op"), self.clients[client_id]
        if fd is not None and op != "publish_store":
            raise SharedModelError("Descriptor is allowed only for store publication")
        if op == "status":
            from .diagnostics import memory_available, process_memory
            memory = {}
            for key, value in self.clients.items():
                try:
                    memory[key] = process_memory(value['identity']['pid'])
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    memory[key] = {'unavailable': True}
            return {"stores": {key: {"state": value["state"], "identity": value.get("content"),
                                      "clients": sorted(value.get("clients", ())),
                                      "shared_bytes": value.get("manifest", {}).get("arena_bytes", 0),
                                      "committed_bytes": 0 if value.get('fd') is None else os.fstat(value['fd']).st_blocks * 512} for key, value in self.stores.items()},
                    'mem_available': memory_available(), 'process_memory': memory,
                    "clients": {key: {"role": value["role"], "identity": value["identity"]} for key, value in self.clients.items()},
                    "devices": {key: dict(value, queue=list(value["queue"])) for key, value in self.devices.items()},
                    "snapshots": {key: {"step": value["metadata"]["step"], "hash": value["metadata"]["content_hash"],
                                         "refs": len(value["clients"])} for key, value in self.snapshots.items()}}, None
        if op == "begin_store":
            identity = request["identity"]
            store = self.stores.get(identity)
            if store is None:
                if client["role"] != "trainer":
                    raise SharedModelError("Only a trainer can build a shared backbone")
                store = self.stores[identity] = {"state": "BUILDING", "builder": client_id, "clients": set()}
                return {"state": "BUILD", "identity": identity}, None
            return {"state": store["state"], "error": store.get("error")}, None
        if op == "fail_store":
            store = self.stores[request["identity"]]
            if store.get("builder") != client_id or store["state"] != "BUILDING":
                raise SharedModelError("Client is not the active builder")
            store.update(state="ERROR", error=str(request.get("error", "Arena build failed"))[:2048])
            return {}, None
        if op == "publish_store":
            store = self.stores[request["identity"]]
            if client_id != store.get("builder") or store["state"] != "BUILDING" or fd is None:
                raise SharedModelError("Invalid store publication")
            manifest = validate_manifest(request["manifest"], os.fstat(fd).st_size)
            if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
                raise SharedModelError("Publication must use a read-only descriptor")
            if manifest.get("backing") == "sealed_memfd":
                if fcntl.fcntl(fd, F_GET_SEALS) & SEALS != SEALS:
                    raise SharedModelError("Unsealed arena publication")
            elif manifest.get("backing") != "readonly_file" or os.fstat(fd).st_mode & 0o222:
                raise SharedModelError("Unsupported immutable arena backing")
            content = manifest["content_digest"] + ":" + manifest["config_digest"]
            shared = self.contents.get(content)
            if shared is None:
                shared = self.contents[content] = {"fd": os.dup(fd), "manifest": manifest}
            elif shared["manifest"]["tensors"] != manifest["tensors"] or shared["manifest"]["variant"] != manifest["variant"] or shared["manifest"]["quantization"] != manifest["quantization"]:
                raise SharedModelError("Content identity has incompatible layout")
            per_lookup = dict(shared['manifest'])
            per_lookup['source_lookup_identity'] = manifest.get('source_lookup_identity')
            per_lookup['provenance'] = manifest.get('provenance')
            store.update(state="READY", content=content, fd=shared['fd'], manifest=per_lookup)
            return {"content_identity": content}, None
        if op == "open_store":
            store = self.stores.get(request["identity"])
            if store is None or store["state"] != "READY":
                raise SharedModelError("Shared store is not READY")
            store["clients"].add(client_id)
            client["stores"].add(request["identity"])
            return {"manifest": store["manifest"]}, store["fd"]
        if op in ("acquire_gpu", "poll_gpu", "ack_quiescent", "cancel_request"):
            if client["role"] == "observer":
                raise SharedModelError("Observer cannot own GPU")
            device = self._device(request["device_uuid"])
            if device["fault"]:
                raise SharedModelError("GPU ownership uncertain; explicit recovery required")
            owner = device["owner"]
            if op == "poll_gpu":
                return {"owner": owner, "yield_requested": bool(owner and owner["client"] == client_id and device["queue"]), "epoch": device["epoch"]}, None
            if op == "cancel_request":
                device["queue"] = collections.deque(r for r in device["queue"] if not (r["client"] == client_id and r["request_id"] == request["request_id"]))
                return {}, None
            if op == "ack_quiescent":
                if not owner or owner["client"] != client_id or owner["epoch"] != request["epoch"]:
                    raise SharedModelError("Stale/non-owner quiescent acknowledgment")
                device["owner"] = None
                return {}, None
            request_id = request["request_id"]
            if owner and owner["client"] == client_id:
                if owner["request_id"] != request_id:
                    raise SharedModelError("Nested lease must use the active request ID")
                return {"granted": True, "epoch": owner["epoch"]}, None
            if not any(r["client"] == client_id and r["request_id"] == request_id for r in device["queue"]):
                device["queue"].append({"client": client_id, "request_id": request_id})
            if owner is None and device["queue"][0] == {"client": client_id, "request_id": request_id}:
                device["queue"].popleft()
                device["epoch"] += 1
                device["owner"] = {"client": client_id, "request_id": request_id, "epoch": device["epoch"], "role": client["role"]}
                return {"granted": True, "epoch": device["epoch"]}, None
            return {"granted": False, "owner": owner}, None
        if op == "recover_gpu":
            if client["role"] != "observer":
                raise SharedModelError("Recovery requires an explicit operator/observer")
            device = self._device(request["device_uuid"])
            fault = device["fault"]
            if not fault or request.get("identity") != fault["identity"] or same_process(fault["identity"]) or request.get("cuda_resources_verified_released") is not True:
                raise SharedModelError("Recovery requires dead process identity and separately verified CUDA resource release")
            device["fault"] = None
            return {}, None
        if op == "publish_adapter":
            if client["role"] != "trainer":
                raise SharedModelError("Only trainer publishes adapters")
            metadata = request["metadata"]
            from .snapshot import validate_snapshot
            validate_snapshot(metadata)
            snapshot_id = metadata["snapshot_id"]
            if snapshot_id in self.snapshots and self.snapshots[snapshot_id]["metadata"] != metadata:
                raise SharedModelError("Snapshot generation is immutable")
            self.snapshots.setdefault(snapshot_id, {"metadata": metadata, "clients": set()})
            return {"snapshot_id": snapshot_id}, None
        if op == "open_adapter":
            snapshot_id = request["snapshot_id"]
            if snapshot_id == "latest":
                candidates = [v for v in self.snapshots.values() if v["metadata"]["store_identity"] == request["store_identity"]]
                if not candidates:
                    raise SharedModelError("No adapter snapshot is available")
                snapshot_id = max(candidates, key=lambda v: (v["metadata"]["step"], v["metadata"]["created_ns"]))["metadata"]["snapshot_id"]
            snapshot = self.snapshots[snapshot_id]
            if snapshot["metadata"]["store_identity"] != request["store_identity"]:
                raise SharedModelError("Adapter/base identity mismatch")
            snapshot["clients"].add(client_id)
            client["snapshots"].add(snapshot_id)
            return {"metadata": snapshot["metadata"]}, None
        if op == "release_adapter":
            self.snapshots[request["snapshot_id"]]["clients"].discard(client_id)
            client["snapshots"].discard(request["snapshot_id"])
            return {}, None
        if op == "prune_adapters":
            if client["role"] != "trainer":
                raise SharedModelError("Only trainer prunes snapshots")
            candidates = sorted((v for v in self.snapshots.values() if v["metadata"]["store_identity"] == request["store_identity"]), key=lambda v: v["metadata"]["created_ns"], reverse=True)
            removed = []
            for snapshot in candidates[max(1, int(request.get("keep", 3))):]:
                if not snapshot["clients"]:
                    metadata = snapshot["metadata"]
                    del self.snapshots[metadata["snapshot_id"]]
                    removed.append(metadata)
            return {"removed": removed}, None
        raise SharedModelError(f"Unknown broker operation {op!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=default_socket())
    args = parser.parse_args()
    broker = Broker(args.socket)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: broker.stop())
    try:
        broker.serve_forever()
    finally:
        broker.close()


if __name__ == "__main__":
    main()

Experimental Linux-only CPU arenas and cooperative GPU leases. The broker imports
only the standard library; clients import their own Torch runtimes lazily.

No running process is migrated. Enable this only for a new frozen-base Hunyuan
Instruct int8 LoRA job after the GPU qualification described in
[the validation report](../../docs/comfy_shared_model_memory_validation.md). Runtime qualification is incomplete.

Install this directory into both environments during an authorized idle window:

```sh
conda run -n ai-toolkit python -m pip install ./packages/aitk_shared_models
conda run -n comfyui python -m pip install ./packages/aitk_shared_models
conda run -n ai-toolkit python -m aitk_shared_models.broker
```

Default socket: `$XDG_RUNTIME_DIR/aitk-shared-models/broker.sock`, with owner-only
directory/socket permissions and Linux peer credentials. No TCP generation service.
An existing socket is never silently removed. If a broker dies, surviving clients
keep their mappings but cannot resume GPU work without explicit reconciliation.

Sealed memfd arenas use bounded 8 MiB checkpoint reads. The Conda Python on this
machine lacks `os.memfd_create`, so the equivalent libc syscall wrapper is used.
Explicit local-file fallback is supported when memfd creation is unavailable;
`/dev/shm` is rejected as the full-backbone fallback. Build budgets reserve at least
12 GiB for configured training and check cgroup v2 limits before allocation.

Torch has no native read-only tensor flag. `SharedTensor` guards write schemas,
cloning, pinning, CPU dtype conversions and concatenation; CPU mappings are also
OS-read-only. Parameter/slice aliases and read-only accounting proxies retain their
arena owner. NumPy/raw writable-storage exports are rejected. Trusted runtime
internals can still misuse a raw pointer: this is an experimental integration,
not a security boundary against arbitrary Python running under the same UID.

Two private pinned staging buffers are capped at 256 MiB each per arena owner.
`cudaHostRegister(ReadOnly)` is an opt-in capability prototype, not the default
transfer policy. Registration success alone does not establish asynchronous H2D
correctness. Bounded CPU conditioning casts are capped at 64 MiB per projection
and recorded in `TensorOwner.cpu_temporary_peak_bytes`.

Never infer GPU release from a timeout/disconnect. A disconnected owner blocks
reassignment. Operator recovery requires the dead process's exact identity and
separate verification that CUDA resources have been released. There is no automatic
process killing, broker restart, service restart or live model export.

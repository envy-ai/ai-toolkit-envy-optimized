# Queue job launch

`processQueue` may run in more than one UI worker. Both workers can read the
same queued job before either starts it. `startJob` first acquires the shared
GPU/render gate, then atomically changes that job from `queued` to `running`.
Only the worker whose conditional update changes one row launches a trainer.
The other worker releases the gate and returns.

The gate also coordinates launches with saved ComfyUI renders. The database
claim protects the job identity after the short launch gate is released; the
trainer continues independently of its UI worker.

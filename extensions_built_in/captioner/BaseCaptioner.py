"""Model-backed captioning setup; shared caption IO and status live in CaptionerProcess."""
import torch

from toolkit.train_tools import get_torch_dtype

from .CaptionerProcess import CaptionerProcess, CaptionConfig


class BaseCaptioner(CaptionerProcess):
    def __init__(self, process_id, job, config, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.model = None
        self.lora_stack = None
        self.processor = None
        self.model2 = None
        self.processor2 = None
        self.device_torch = torch.device(self.caption_config.device)
        self.torch_dtype = get_torch_dtype(self.caption_config.dtype)

    def run(self):
        super(BaseCaptioner, self).run()
        with torch.no_grad():
            self.start_stop_watcher()
            self.update_status("running", "Loading Model")
            self.load_model()
            self.load_loras()
            self.maybe_compile_models()
            self.update_status("running", "Looking for files")
            self.find_files()
            self.update_db_key("total_steps", len(self.file_paths))
            self.update_step()
            self.update_status("running", f"Captioning {len(self.file_paths)} files")
            self.run_caption_loop()
            self.update_status("completed", "Captioning completed")
            print("")

            print("****************************************************")
            print("Captioning complete")
            print("****************************************************")

    def load_model(self):
        raise NotImplementedError("Model loading not implemented for this captioner")

    def load_loras(self):
        """Attach the configured LoRAs to self.model as forward-hook sidechains
        (the inference engine's hook mode): the low-rank branch runs alongside
        the layer in the model's compute dtype and the base weights are never
        touched, so quantized weights stay on their grid and nothing is merged."""
        specs = [
            l for l in self.caption_config.loras if isinstance(l, dict) and l.get("path")
        ]
        if not specs or self.model is None:
            return
        from toolkit.inference_lora import LoRAStack

        # self is the holder: LoRAStack reads .model (and .text_encoder, unused here)
        stack = LoRAStack(self, mode="hook").load(
            specs, status_fn=self.print_and_status_update
        )
        stack.apply(status_fn=self.print_and_status_update)
        self.lora_stack = stack
        self.print_and_status_update(
            f"Applied {len(stack.loras)} LoRA(s) as sidechains"
        )

    def maybe_compile_models(self):
        if not self.caption_config.compile:
            return
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            print(
                "[AITK] compile requested but triton is not installed, skipping compilation."
            )
            return
        try:
            # compilation happens lazily on first forward, so fall back to
            # eager there too if the backend fails (e.g. broken triton install)
            torch._dynamo.config.suppress_errors = True
            for model in [self.model, self.model2]:
                if model is not None and isinstance(model, torch.nn.Module):
                    # compile per transformer block instead of the whole model:
                    # small graphs compile far faster and identical blocks hit
                    # the inductor cache, vs many minutes tracing one huge graph
                    compiled_blocks = self._compile_blocks(model)
                    if compiled_blocks == 0:
                        # no repeated block lists found; compile the whole model
                        # dynamic=True avoids recompiling for every new image/token shape
                        model.compile(dynamic=True)
            print(
                "[AITK] Model compilation enabled. The first few items will be slow while the model compiles."
            )
        except Exception as e:
            print(f"[AITK] Failed to compile model, continuing without compile: {e}")

    def _compile_blocks(self, model: torch.nn.Module) -> int:
        """Compile the repeated transformer blocks individually, leaving one-off
        modules (embeddings, mergers, lm_head) eager. Returns the number of
        blocks compiled."""
        # candidate lists: ModuleLists of >= 2 blocks that all share one class
        # and have submodules of their own (i.e. real transformer blocks, not
        # lists of leaf layers)
        candidates = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
                continue
            classes = {type(b) for b in module}
            if len(classes) != 1:
                continue
            if next(module[0].children(), None) is None:
                continue
            candidates.append(name)
        # skip lists nested inside another candidate list
        candidates = [
            name
            for name in candidates
            if not any(
                name != other and name.startswith(other + ".") for other in candidates
            )
        ]
        count = 0
        for name in candidates:
            block_list = model.get_submodule(name)
            for i, block in enumerate(block_list):
                block_list[i] = torch.compile(block, dynamic=True)
                count += 1
        return count

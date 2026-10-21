import os
import tqdm
from .BaseCaptioner import BaseCaptioner, CaptionConfig
from .api_caption import DEFAULT_PROMPT, IMAGE_EXTENSIONS, generate_api_caption, validate_api_settings


class APICaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        kwargs = {**kwargs, "device": "cpu", "compile": False, "quantize": False, "loras": [],
                  "extensions": kwargs.get("extensions", sorted(IMAGE_EXTENSIONS))}
        validate_api_settings(kwargs)
        super().__init__(**kwargs)
        self.api_settings = kwargs


class APICaptioner(BaseCaptioner):
    caption_config_class = APICaptionConfig

    def load_model(self):
        """Remote inference only; no weights, CUDA context, or compilation."""

    def run(self):
        self.start_stop_watcher()
        self.update_status("running", "Looking for images (API captioner)")
        self.find_files()
        self.update_db_key("total_steps", len(self.file_paths))
        self.update_step()
        self.update_status("running", f"API captioning {len(self.file_paths)} images")
        for filename in tqdm.tqdm(self.file_paths, desc="API captioning", unit="image"):
            self.maybe_stop()
            # Unlike model batch captioners, transport/auth/truncation failures stop
            # the job and reach BaseCaptioner.on_error; never report false completion.
            caption = self.get_caption_for_file(filename)
            self.maybe_stop()
            self.save_caption_for_file(filename, caption)
            self.step_num += 1
            self.update_step()
        self.update_status("completed", "API captioning completed")

    def get_caption_for_file(self, file_path):
        settings = {**self.caption_config.api_settings}
        settings["caption_prompt"] = self.resolve_prompt(settings.get("caption_prompt") or DEFAULT_PROMPT, file_path)
        return generate_api_caption(file_path, settings, os.environ.get("AITK_API_CAPTION_KEY", ""))

"""Offline contract tests for the OpenAI-compatible caption transport."""

import base64
import hashlib
import importlib.util
import io
import json
import math
import os
import runpy
import ssl
import subprocess
import sys
import tempfile
import threading
import types
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "extensions_built_in/captioner/api_caption.py"
API = runpy.run_path(str(MODULE_PATH))
generate_api_caption = API["generate_api_caption"]
validate_api_settings = API["validate_api_settings"]
apply_regex_replace = API["apply_regex_replace"]
validate_regex_replace = API["validate_regex_replace"]


class CaptionServer:
    def __init__(self, *, status=200, response=None, location=None):
        self.requests = []
        self.status = status
        self.response = response if response is not None else {
            "choices": [{"finish_reason": "stop", "message": {"content": "  a green glass vase  "}}]
        }
        self.location = location
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                owner.requests.append({
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "payload": json.loads(self.rfile.read(size)),
                })
                self.send_response(owner.status)
                if owner.location:
                    self.send_header("Location", owner.location)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                body = owner.response if isinstance(owner.response, bytes) else json.dumps(owner.response).encode()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def image_fixture(tmp_path):
    path = tmp_path / "edge-fixture.png"
    Image.new("RGBA", (11, 7), (20, 100, 220, 127)).save(path)
    return path


def decode_sent_image(server):
    data_url = server.requests[0]["payload"]["messages"][-1]["content"][1]["image_url"]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def config(server, **overrides):
    result = {
        "api_url": server.url,
        "model_name_or_path": "fixture-model",
        "caption_prompt": "Describe the blue fixture.",
        "system_prompt": "Return one concise caption.",
        "max_new_tokens": 4096,
        "api_timeout": 3,
        "api_max_tokens_field": "max_tokens",
        # Existing transport fixtures are intentionally not resized. Tests for
        # the API's missing-field default explicitly omit this key.
        "image_megapixels": None,
    }
    result.update(overrides)
    return result


def regex_replace(pattern, replacement="$1", **flags):
    return {
        "enabled": flags.get("enabled", True),
        "pattern": pattern,
        "replacement": replacement,
        "ignore_case": flags.get("ignore_case", False),
        "multiline": flags.get("multiline", False),
        "dotall": flags.get("dotall", False),
        "verbose": flags.get("verbose", False),
        "global": flags.get("global", True),
    }


def test_openai_compatible_request_has_image_prompts_auth_and_selected_limit(tmp_path):
    with CaptionServer() as server:
        caption = generate_api_caption(
            image_fixture(tmp_path),
            config(server, api_max_tokens_field="max_completion_tokens", max_new_tokens=16384, temperature=0.25,
                   top_p=0.8, frequency_penalty=-0.2, presence_penalty=0.3, seed=7,
                   stop=["END"]),
            api_key="fixture-secret",
        )

    assert caption == "a green glass vase"
    assert len(server.requests) == 1
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer fixture-secret"
    payload = request["payload"]
    assert payload["model"] == "fixture-model"
    assert payload["stream"] is False
    assert payload["max_completion_tokens"] == 16384
    assert "max_tokens" not in payload
    assert {key: payload[key] for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop")} == {
        "temperature": 0.25, "top_p": 0.8, "frequency_penalty": -0.2,
        "presence_penalty": 0.3, "seed": 7, "stop": ["END"],
    }
    assert payload["messages"] == [
        {"role": "system", "content": "Return one concise caption."},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe the blue fixture."},
            {"type": "image_url", "image_url": {"url": payload["messages"][1]["content"][1]["image_url"]["url"]}},
        ]},
    ]
    data_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as sent_image:
        assert sent_image.format == "PNG"
        assert sent_image.size == (11, 7)
        assert sent_image.mode == "RGBA"


def test_blank_key_omits_auth_and_unset_sampling_options_are_not_injected(tmp_path):
    with CaptionServer() as server:
        generate_api_caption(image_fixture(tmp_path), config(server), api_key="")

    request = server.requests[0]
    assert request["authorization"] is None
    payload = request["payload"]
    assert payload["max_tokens"] == 4096
    assert not ({"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"} & payload.keys())


@pytest.mark.parametrize("model", [None, "", " \t\n "])
def test_missing_or_blank_model_is_omitted_from_http_payload(tmp_path, model):
    with CaptionServer() as server:
        settings = config(server)
        if model is None:
            settings.pop("model_name_or_path")
        else:
            settings["model_name_or_path"] = model
        generate_api_caption(image_fixture(tmp_path), settings)

    assert "model" not in server.requests[0]["payload"]


@pytest.mark.parametrize(("source_size", "limit", "expected_size"), [
    ((1600, 900), 0.25, (667, 375)),
    ((10, 6), 0.001, (41, 24)),
])
def test_image_megapixel_limit_resizes_up_or_down_preserving_aspect_and_alpha(tmp_path, source_size, limit, expected_size):
    filename = tmp_path / "resize-fixture.png"
    Image.new("RGBA", source_size, (20, 100, 220, 127)).save(filename)
    original = hashlib.sha256(filename.read_bytes()).digest()

    with CaptionServer() as server:
        generate_api_caption(filename, config(server, image_megapixels=limit))

    with decode_sent_image(server) as sent:
        assert sent.size == expected_size
        assert sent.mode == "RGBA"
        assert abs(sent.getpixel((sent.width // 2, sent.height // 2))[3] - 127) <= 1
    assert hashlib.sha256(filename.read_bytes()).digest() == original
    payload = server.requests[0]["payload"]
    assert "image_megapixels" not in payload
    scale = math.sqrt(limit * 1_000_000 / (source_size[0] * source_size[1]))
    assert abs(expected_size[0] - source_size[0] * scale) <= 0.5
    assert abs(expected_size[1] - source_size[1] * scale) <= 0.5


def test_missing_image_megapixels_defaults_to_one_megapixel_and_null_disables_resize(tmp_path):
    filename = tmp_path / "default-limit.png"
    Image.new("RGBA", (2000, 1000), (50, 120, 210, 96)).save(filename)
    source_hash = hashlib.sha256(filename.read_bytes()).digest()

    with CaptionServer() as server:
        settings = config(server)
        settings.pop("image_megapixels")
        generate_api_caption(filename, settings)
    with decode_sent_image(server) as sent:
        assert sent.size == (1414, 707)
        assert sent.width * sent.height <= 1_000_000
        assert sent.mode == "RGBA"
        assert sent.getpixel((sent.width // 2, sent.height // 2))[3] == 96

    with CaptionServer() as server:
        generate_api_caption(filename, config(server, image_megapixels=None))
    with decode_sent_image(server) as sent:
        assert sent.size == (2000, 1000)
        assert sent.mode == "RGBA"
    assert hashlib.sha256(filename.read_bytes()).digest() == source_hash


def test_megapixel_resize_applies_exif_orientation_before_calculating_target_size(tmp_path):
    filename = tmp_path / "orientation-fixture.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (1600, 900), (100, 120, 140)).save(filename, exif=exif, quality=95)
    source_hash = hashlib.sha256(filename.read_bytes()).digest()

    with CaptionServer() as server:
        generate_api_caption(filename, config(server, image_megapixels=1))

    with decode_sent_image(server) as sent:
        assert sent.size == (750, 1333)
        assert sent.height > sent.width
        assert sent.width * sent.height <= 1_000_000
    with Image.open(filename) as original:
        assert original.size == (1600, 900)
        assert original.getexif()[274] == 6
    assert hashlib.sha256(filename.read_bytes()).digest() == source_hash


@pytest.mark.parametrize("value", [0, -0.5, True, False, float("inf"), float("nan"), "1"])
def test_invalid_image_megapixel_limits_fail_before_opening_the_image_or_requesting_provider(tmp_path, value):
    with CaptionServer() as server:
        with pytest.raises(ValueError, match="megapixel"):
            generate_api_caption(tmp_path / "missing.png", config(server, image_megapixels=value))
        assert server.requests == []


@pytest.mark.parametrize("value", [1e-12, 1e308])
def test_unrepresentable_normalized_image_dimensions_fail_without_provider_request(tmp_path, value):
    with CaptionServer() as server:
        with pytest.raises(ValueError, match="image|pixel|dimension|megapixel"):
            generate_api_caption(image_fixture(tmp_path), config(server, image_megapixels=value))
        assert server.requests == []


def test_user_xml_caption_pattern_handles_case_multiline_verbose_and_escaped_regex_slash():
    flags = dict(ignore_case=True, multiline=True, dotall=True, verbose=True)
    pattern = r"""
        ^ .* <caption> (.*) <\/caption> .* $
    """
    settings = regex_replace(pattern, **flags)
    plain_xml = "prefix\n<CAPTION>blue\nsky</caption>\nsuffix"

    assert apply_regex_replace(plain_xml, settings) == "blue\nsky"
    # Escaping `/` is valid regex spelling in Python and matches the same XML
    # closing tag; it is not interpreted as an extra literal backslash.
    assert validate_regex_replace(settings) is not None


def test_regex_replace_preserves_literal_escaped_slash_in_provider_text_when_pattern_requests_it():
    source = "before <caption>green\nsky<\\/caption> after"
    settings = regex_replace(r".*<caption>(.*)<\\/caption>.*", dotall=True)
    assert apply_regex_replace(source, settings) == "green\nsky"


def test_regex_flags_and_global_switch_have_their_standard_independent_meanings():
    multiline_pattern = r"^ caption: \s* (.*) $"
    text = "prefix\nCAPTION: blue sky\nsuffix"
    multiline = regex_replace(multiline_pattern, ignore_case=True, multiline=True, verbose=True)
    assert apply_regex_replace(text, multiline) == "prefix\nblue sky\nsuffix"
    no_multiline = regex_replace(multiline_pattern, ignore_case=True, verbose=True)
    assert apply_regex_replace(text, no_multiline) == text

    repeated = "pre<caption>one</caption> mid <caption>two</caption>end"
    all_matches = regex_replace(r"<caption>(.*?)</caption>")
    first_match = regex_replace(r"<caption>(.*?)</caption>", **{"global": False})
    assert apply_regex_replace(repeated, all_matches) == "preone mid twoend"
    assert apply_regex_replace(repeated, first_match) == "preone mid <caption>two</caption>end"


def test_regex_replacement_dollar_tokens_and_literal_backslashes_are_not_python_evaluated():
    source = "cat-7 cat-8"
    settings = regex_replace(
        r"(?P<word>\w+)-(\d+)",
        r"$<word> $2 $& $$ \1",
    )
    assert apply_regex_replace(source, settings) == r"cat 7 cat-7 $ \1 cat 8 cat-8 $ \1"


def test_absent_disabled_and_no_match_cleanup_preserve_caption_and_disabled_draft_is_not_compiled():
    source = "Keep this caption <caption>literal</caption>"
    assert apply_regex_replace(source, None) == source
    assert apply_regex_replace(source, regex_replace("(", "$99", enabled=False)) == source
    assert apply_regex_replace(source, regex_replace("missing", "replacement")) == source


@pytest.mark.parametrize("pattern,replacement", [
    ("(", "$1"),
    ("(x)", "$2"),
    ("(x)", "$0"),
    ("(?P<word>x)", "$<missing>"),
    ("(x)", "$<unfinished"),
])
def test_invalid_regex_or_capture_reference_fails_before_provider_or_image_io(tmp_path, pattern, replacement):
    with CaptionServer() as server:
        invalid = config(
            server,
            regex_replace=regex_replace(pattern, replacement),
        )
        with pytest.raises(ValueError, match="regex"):
            # If validation happens after request construction, this missing
            # path would raise FileNotFoundError instead and the server would
            # observe a request.
            generate_api_caption(tmp_path / "missing.png", invalid)
        assert server.requests == []


def test_cleanup_that_removes_the_entire_caption_is_a_hard_error():
    with pytest.raises(ValueError, match="empty caption; caption not saved"):
        apply_regex_replace("<caption> \n </caption>", regex_replace(r".*<caption>(.*)</caption>.*", dotall=True))


@pytest.mark.parametrize("finish_reason,message,pattern", [
    ("length", {"content": "partial"}, "truncated"),
    ("stop", {"content": "  "}, "empty"),
    ("content_filter", {"content": "refused"}, "did not complete"),
    ("stop", {"content": "ignored", "refusal": "blocked"}, "refused"),
])
def test_incomplete_or_missing_caption_is_a_hard_error(tmp_path, finish_reason, message, pattern):
    with CaptionServer(response={"choices": [{"finish_reason": finish_reason, "message": message}]}) as server:
        with pytest.raises(RuntimeError, match=pattern):
            generate_api_caption(image_fixture(tmp_path), config(server))


@pytest.mark.parametrize("response", [
    {"choices": []},
    {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}},
                  {"finish_reason": "stop", "message": {"content": "extra"}}]},
    {"error": {"message": "provider echoed private detail"}},
    b"not-json",
])
def test_malformed_provider_responses_fail_closed(tmp_path, response):
    with CaptionServer(response=response) as server:
        with pytest.raises(RuntimeError):
            generate_api_caption(image_fixture(tmp_path), config(server))


def test_http_error_body_is_not_echoed_and_redirect_is_not_followed(tmp_path):
    secret = "fixture-secret-do-not-log"
    with CaptionServer(status=401, response={"error": secret}) as server:
        with pytest.raises(RuntimeError) as error:
            generate_api_caption(image_fixture(tmp_path), config(server), api_key=secret)
        assert secret not in str(error.value)
        assert len(server.requests) == 1
    with CaptionServer(status=302, location="http://127.0.0.1:1/stolen") as server:
        with pytest.raises(RuntimeError, match="redirects are disabled"):
            generate_api_caption(image_fixture(tmp_path), config(server), api_key=secret)
        assert len(server.requests) == 1


def test_caption_text_that_echoes_the_api_key_is_withheld(tmp_path):
    secret = "caption-key-fixture"
    with CaptionServer(response={"choices": [{
        "finish_reason": "stop", "message": {"content": f"caption containing {secret}"},
    }]}) as server:
        with pytest.raises(RuntimeError, match="echoed its credential") as error:
            generate_api_caption(image_fixture(tmp_path), config(server), api_key=secret)
        assert secret not in str(error.value)


def test_cleanup_cannot_hide_a_provider_caption_that_echoes_the_api_key(tmp_path):
    secret = "caption-key-hidden-by-regex-fixture"
    response = {"choices": [{
        "finish_reason": "stop",
        "message": {"content": f"private wrapper {secret} <caption>safe text</caption>"},
    }]}
    settings = regex_replace(r".*<caption>(.*)</caption>.*", dotall=True)
    with CaptionServer(response=response) as server:
        with pytest.raises(RuntimeError, match="echoed its credential") as error:
            generate_api_caption(
                image_fixture(tmp_path), config(server, regex_replace=settings), api_key=secret
            )
    assert secret not in str(error.value)
    assert len(server.requests) == 1


def test_cleanup_cannot_create_a_caption_that_echoes_the_api_key(tmp_path):
    secret = "fixturesecret"
    response = {"choices": [{
        "finish_reason": "stop",
        "message": {"content": "fixture-secret"},
    }]}
    settings = regex_replace(r"(fixture)-(secret)", "$1$2")
    with CaptionServer(response=response) as server:
        with pytest.raises(RuntimeError, match="cleanup contains the API credential") as error:
            generate_api_caption(
                image_fixture(tmp_path), config(server, regex_replace=settings), api_key=secret
            )
    assert secret not in str(error.value)
    assert len(server.requests) == 1


def test_https_uses_python_default_certificate_verification():
    handlers = API["build_opener"](API["NoCaptionRedirects"]()).handlers
    https = next(handler for handler in handlers if handler.__class__.__name__ == "HTTPSHandler")
    assert https._context is None, "urllib should create its normal verified TLS context"
    default_context = ssl._create_default_https_context()
    assert default_context.verify_mode == ssl.CERT_REQUIRED
    assert default_context.check_hostname is True


@pytest.mark.parametrize("bad_url", [
    "https://user:pass@example.test/v1", "https://example.test/v1?key=x",
    "https://example.test/v1#fragment", "file:///tmp/model",
])
def test_invalid_api_urls_are_rejected_before_network_io(tmp_path, bad_url):
    with pytest.raises(ValueError):
        validate_api_settings(config(type("Server", (), {"url": bad_url})()))


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 9_007_199_254_740_992])
def test_response_token_limit_must_be_a_positive_safe_integer(limit):
    with pytest.raises(ValueError, match="safe integer"):
        validate_api_settings({
            "api_url": "http://127.0.0.1:12345/v1", "model_name_or_path": "fixture",
            "max_new_tokens": limit,
        })


@pytest.mark.parametrize("model", [None, 42, True, [], {}])
def test_malformed_supplied_model_type_is_rejected(model):
    settings = {"api_url": "http://127.0.0.1:12345/v1", "max_new_tokens": 4096,
                "model_name_or_path": model}
    with pytest.raises(ValueError, match="model"):
        validate_api_settings(settings)


def test_response_token_limit_over_default_is_valid_and_kept_exact():
    settings = {"api_url": "http://127.0.0.1:12345/v1", "model_name_or_path": "fixture", "max_new_tokens": 16384}
    assert validate_api_settings(settings) == "http://127.0.0.1:12345/v1/chat/completions"


def _load_api_captioner_with_test_base():
    """Load the real run loop while stubbing BaseCaptioner to avoid model/toolkit startup."""
    package_name = "_api_captioner_test_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[package_name] = package
    base = types.ModuleType(package_name + ".BaseCaptioner")

    class CaptionConfig:
        def __init__(self, **kwargs):
            self.recaption = kwargs.get("recaption", False)

    class BaseCaptioner:
        caption_config_class = CaptionConfig

    base.CaptionConfig = CaptionConfig
    base.BaseCaptioner = BaseCaptioner
    sys.modules[base.__name__] = base

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(package_name + ".api_caption", MODULE_PATH)
    return load(package_name + ".APICaptioner", MODULE_PATH.parent / "APICaptioner.py")


def _fake_run_loop(captioner, *, should_stop=False, fail=None):
    statuses, saved, called = [], [], []
    captioner.start_stop_watcher = lambda: None
    captioner.update_status = lambda status, _message: statuses.append(status)
    captioner.find_files = lambda: None
    captioner.file_paths = ["synthetic-image.png"]
    captioner.update_db_key = lambda *_args: None
    captioner.update_step = lambda: None
    captioner.step_num = 0
    captioner.maybe_stop = lambda: (_ for _ in ()).throw(KeyboardInterrupt()) if should_stop else None

    def get_caption(filename):
        called.append(filename)
        if fail:
            raise fail
        return "caption"

    captioner.get_caption_for_file = get_caption
    captioner.save_caption_for_file = lambda *_args: saved.append(True)
    return statuses, saved, called


def test_api_captioner_failure_and_stop_never_write_completion_status():
    module = _load_api_captioner_with_test_base()
    captioner = module.APICaptioner.__new__(module.APICaptioner)
    statuses, saved, called = _fake_run_loop(captioner, fail=RuntimeError("synthetic 401"))
    with pytest.raises(RuntimeError, match="synthetic 401"):
        captioner.run()
    assert "completed" not in statuses
    assert saved == []
    assert called == ["synthetic-image.png"]

    captioner = module.APICaptioner.__new__(module.APICaptioner)
    statuses, saved, called = _fake_run_loop(captioner, should_stop=True)
    with pytest.raises(KeyboardInterrupt):
        captioner.run()
    assert "completed" not in statuses
    assert saved == []
    assert called == []


def test_api_captioner_defaults_to_preserving_existing_sidecars_and_recaptions_when_enabled(tmp_path):
    from extensions_built_in.captioner.APICaptioner import APICaptionConfig, APICaptioner

    image = tmp_path / "already-captioned.png"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(image)
    image.with_suffix(".txt").write_text("existing caption", encoding="utf-8")
    common = {
        "path_to_caption": str(tmp_path), "extensions": ["png"], "caption_extension": "txt",
        "model_name_or_path": "fixture-model", "api_url": "http://127.0.0.1:12345/v1",
        "caption_prompt": "fixture", "max_new_tokens": 4096,
    }

    default_config = APICaptionConfig(**common)
    assert default_config.recaption is False
    default_run = APICaptioner.__new__(APICaptioner)
    default_run.caption_config = default_config
    default_run.file_paths = []
    default_run.find_files()
    assert default_run.file_paths == []

    overwrite_config = APICaptionConfig(**common, recaption=True)
    overwrite_run = APICaptioner.__new__(APICaptioner)
    overwrite_run.caption_config = overwrite_config
    overwrite_run.file_paths = []
    overwrite_run.find_files()
    assert overwrite_run.file_paths == [str(image)]
    overwrite_run.save_caption_for_file(str(image), "fresh API caption")
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "fresh API caption"


@pytest.mark.parametrize("model", [None, "fixture-model"])
def test_run_py_api_captioner_works_without_cuda_or_eager_extension_imports(model):
    """Exercise the real CLI bootstrap while keeping inference local and synthetic."""
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    (repo_root / "tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="api-caption-cli-", dir=repo_root / "tmp") as work_dir:
        work = Path(work_dir)
        image_dir = work / "images"
        image_dir.mkdir()
        Image.new("RGBA", (9, 5), (30, 90, 180, 123)).save(image_dir / "fixture.png")
        config_path = work / "job.yaml"

        with CaptionServer(response={"choices": [{
            "finish_reason": "stop",
            "message": {"content": "prefix\n<CAPTION>blue\nvase</caption>\nsuffix"},
        }]}) as server:
            config_path.write_text(yaml.safe_dump({
                "job": "extension",
                "config": {
                    "name": "offline-api-caption-cli-regression",
                    "process": [{
                        "type": "APICaptioner",
                        "sqlite_db_path": str(work / "job.sqlite"),
                        "caption": {
                            "path_to_caption": str(image_dir),
                            "extensions": ["png"],
                            "caption_extension": "txt",
                            "api_url": server.url,
                            "system_prompt": "Return one concise caption.",
                            "caption_prompt": "Describe the synthetic fixture.",
                            "max_new_tokens": 48,
                            "api_timeout": 5,
                            "regex_replace": regex_replace(
                                r".*<caption>(.*)<\/caption>.*",
                                ignore_case=True,
                                multiline=True,
                                dotall=True,
                                verbose=True,
                            ),
                        },
                    }],
                },
            }), encoding="utf-8")

            if model is not None:
                config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                config_data["config"]["process"][0]["caption"]["model_name_or_path"] = model
                config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["AITK_API_CAPTION_KEY"] = ""
            # Prevent a developer shell's provider credentials from reaching this test.
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AITK_JOB_ID", "SEED"):
                env.pop(key, None)
            result = subprocess.run(
                ["conda", "run", "-n", "ai-toolkit", "python", "run.py", str(config_path)],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=150,
                check=False,
            )

        assert result.returncode == 0, (
            f"run.py exited {result.returncode}; stderr tail:\n{result.stderr[-5000:]}\n"
            f"stdout tail:\n{result.stdout[-5000:]}"
        )
        assert (image_dir / "fixture.txt").read_text(encoding="utf-8") == "blue\nvase"
        assert len(server.requests) == 1
        request = server.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["authorization"] is None
        if model is None:
            assert "model" not in request["payload"]
        else:
            assert request["payload"]["model"] == model
        assert request["payload"]["max_tokens"] == 48
        assert "regex_replace" not in request["payload"]
        assert request["payload"]["messages"][-1]["content"][0]["text"] == "Describe the synthetic fixture."


def test_empty_regex_cleanup_keeps_existing_caption_unchanged(tmp_path):
    module = _load_api_captioner_with_test_base()
    image = image_fixture(tmp_path)
    sidecar = image.with_suffix(".txt")
    sidecar.write_text("existing caption must survive", encoding="utf-8")
    server_response = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "<caption> \n\t </caption>"},
        }]
    }

    with CaptionServer(response=server_response) as server:
        captioner = module.APICaptioner.__new__(module.APICaptioner)
        captioner.caption_config = SimpleNamespace(api_settings=config(
            server,
            regex_replace=regex_replace(r".*<caption>(.*)</caption>.*", dotall=True),
        ))
        captioner.resolve_prompt = lambda prompt, _filename: prompt
        captioner.start_stop_watcher = lambda: None
        captioner.update_status = lambda *_args: None
        captioner.find_files = lambda: None
        captioner.file_paths = [str(image)]
        captioner.update_db_key = lambda *_args: None
        captioner.update_step = lambda: None
        captioner.step_num = 0
        captioner.maybe_stop = lambda: None
        captioner.save_caption_for_file = lambda filename, caption: Path(filename).with_suffix(".txt").write_text(
            caption, encoding="utf-8"
        )

        with pytest.raises(ValueError, match="empty caption; caption not saved"):
            captioner.run()

    assert len(server.requests) == 1
    assert sidecar.read_text(encoding="utf-8") == "existing caption must survive"


def test_extension_job_only_bypasses_global_discovery_for_api_captioner(monkeypatch):
    import importlib
    from collections import OrderedDict

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module = importlib.import_module("jobs.ExtensionJob")
    discovered = []
    loaded = []
    api_process = object()
    global_process = object()

    def global_discovery():
        discovered.append(True)
        return {"APICaptioner": global_process, "OtherExtension": global_process}

    def capture_load_processes(self, process_dict):
        loaded.append(process_dict)

    monkeypatch.setattr(module, "get_all_extensions_process_dict", global_discovery)
    monkeypatch.setattr(module.BaseJob, "load_processes", capture_load_processes)
    from extensions_built_in.captioner import APICaptionerExtension
    monkeypatch.setattr(APICaptionerExtension, "get_process", classmethod(lambda cls: api_process))

    def make_job(processes):
        return module.ExtensionJob(OrderedDict([
            ("job", "extension"),
            ("config", OrderedDict([("name", "resolver-fixture"), ("process", processes)])),
        ]))

    make_job([{"type": "APICaptioner"}, {"type": "APICaptioner"}])
    assert loaded[-1] == {"APICaptioner": api_process}
    assert discovered == []

    make_job([{"type": "APICaptioner"}, {"type": "OtherExtension"}])
    assert loaded[-1] == {"APICaptioner": global_process, "OtherExtension": global_process}
    make_job([{"type": "OtherExtension"}])
    assert loaded[-1] == {"APICaptioner": global_process, "OtherExtension": global_process}
    assert discovered == [True, True]

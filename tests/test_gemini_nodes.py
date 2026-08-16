import hashlib
import threading
from io import BytesIO
from types import SimpleNamespace

import google.auth
import pytest
import torch
from PIL import Image

from custom_nodes.ComfyUI_Gemini_Expanded_API import gemini_nodes


def _execute_kwargs(config):
    return {
        "config": config,
        "prompt": "test prompt",
        "system_instruction": "test system instruction",
        "model": "gemini-2.0-flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 128,
        "include_images": False,
        "aspect_ratio": "None",
        "bypass_mode": "None",
        "thinking_budget": 0,
    }


def _config(**overrides):
    config = {
        "api_key": "",
        "api_version": "v1",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
    }
    config.update(overrides)
    return config


def _install_success_client(monkeypatch, captured):
    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            part = SimpleNamespace(text="ok", inline_data=None)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()


def test_timeout_fallback_is_last_input():
    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()

    assert schema.inputs[-1].id == "timeout_fallback_text"


def test_vertex_config_socket_ids_remain_stable():
    schema = gemini_nodes.SSL_GeminiAPIKeyConfig.define_schema()

    assert [input_.id for input_ in schema.inputs] == [
        "api_key",
        "api_version",
        "use_vertexai_env",
        "vertexai_express",
        "vertexai_project",
        "vertexai_location",
        "google_application_credentials",
    ]


def test_image_inputs_use_clean_autogrow_socket_ids():
    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()
    image_inputs = next(input_ for input_ in schema.inputs if input_.id == "image_inputs")

    assert image_inputs.template.min == 0
    assert image_inputs.template.names[:3] == [
        "image_1",
        "image_2",
        "image_3",
    ]
    assert image_inputs.template.names[-1] == "image_100"
    assert not {"input_image", "input_image_2"} & {input_.id for input_ in schema.inputs}


def test_image_inputs_flatten_in_numeric_socket_and_batch_order():
    first_batch = torch.stack(
        [
            torch.full((2, 2, 3), 0.1),
            torch.full((2, 2, 3), 0.2),
            torch.full((2, 2, 3), 0.3),
            torch.full((2, 2, 3), 0.4),
            torch.full((2, 2, 3), 0.5),
        ]
    )
    second_socket = torch.full((1, 2, 2, 3), 0.6)

    frames, batch_counts = gemini_nodes.SSL_GeminiTextPrompt._ordered_image_frames(
        {
            "image_2": second_socket,
            "image_1": first_batch,
        }
    )

    assert [(socket, index) for socket, index, _ in frames] == [
        ("image_1", 0),
        ("image_1", 1),
        ("image_1", 2),
        ("image_1", 3),
        ("image_1", 4),
        ("image_2", 0),
    ]
    assert [round(float(frame[0, 0, 0]), 1) for _, _, frame in frames] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
    ]
    assert batch_counts == [("image_1", 5), ("image_2", 1)]


def test_image_inputs_reject_non_bhwc_tensor_with_socket_name():
    with pytest.raises(ValueError, match=r"Gemini image_2 must be a BHWC image tensor"):
        gemini_nodes.SSL_GeminiTextPrompt._ordered_image_frames(
            {"image_2": torch.zeros((2, 2, 3))}
        )


def test_all_ordered_images_are_sent_to_gemini_2_and_3_before_prompt(monkeypatch):
    captured = {"call_count": 0}

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["call_count"] += 1
            captured.update(kwargs)
            part = SimpleNamespace(text="ok", inline_data=None)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    config = {
        "api_key": "test-key",
        "api_version": "v1",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
    }
    kwargs = _execute_kwargs(config)
    kwargs["image_inputs"] = {
        "image_2": torch.full((1, 2, 2, 3), 0.6),
        "image_1": torch.stack(
            [
                torch.full((2, 2, 3), 0.1),
                torch.full((2, 2, 3), 0.2),
                torch.full((2, 2, 3), 0.3),
                torch.full((2, 2, 3), 0.4),
                torch.full((2, 2, 3), 0.5),
            ]
        ),
    }

    def assert_ordered_parts():
        assert captured["call_count"] == 1
        assert len(captured["contents"]) == 1
        parts = captured["contents"][0].parts
        assert len(parts) == 7
        assert parts[-1].text == "test prompt"
        pixels = [
            Image.open(BytesIO(part.inline_data.data)).getpixel((0, 0))[0]
            for part in parts[:-1]
        ]
        assert pixels == [25, 51, 76, 102, 127, 153]

    for model in ("gemini-2.0-flash", "gemini-3-flash-preview"):
        captured.clear()
        captured["call_count"] = 0
        kwargs["model"] = model
        output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

        assert output[0] == "ok"
        assert_ordered_parts()

    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    kwargs["config"] = config | {"use_vertexai_env": True, "api_key": ""}
    kwargs["model"] = "gemini-2.5-flash"
    captured.clear()
    captured["call_count"] = 0

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert output[0] == "ok"
    assert_ordered_parts()


def test_fingerprint_hashes_every_ordered_image():
    common = {
        "config": {"api_key": "key", "api_version": "v1"},
        "prompt": "prompt",
        "system_instruction": "system",
        "model": "gemini-2.0-flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 128,
        "include_images": False,
        "aspect_ratio": "None",
        "bypass_mode": "None",
        "thinking_budget": 0,
        "use_seed": True,
        "seed": 1,
    }
    first = torch.zeros((1, 2, 2, 3))
    second = torch.ones((1, 2, 2, 3))

    fingerprint_a, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        image_inputs={"image_1": first, "image_2": second},
    )
    fingerprint_b, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        image_inputs={"image_1": second, "image_2": first},
    )

    assert fingerprint_a != fingerprint_b


def test_api_key_is_hashed_in_fingerprint():
    api_key = "secret-api-key"
    fingerprint, cached = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        config={"api_key": api_key, "api_version": "v1"},
        prompt="prompt",
        system_instruction="system",
        model="gemini-2.0-flash",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=128,
        include_images=False,
        aspect_ratio="None",
        bypass_mode="None",
        thinking_budget=0,
        use_seed=True,
        seed=1,
    )

    assert cached is None
    assert api_key not in fingerprint[0]
    assert hashlib.sha256(api_key.encode("utf-8")).hexdigest() in fingerprint[0]


def test_false_vertex_environment_value_is_rejected(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setattr(
        gemini_nodes.genai,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Client must not be created")),
    )
    config = {
        "api_key": "",
        "api_version": "v1",
        "use_vertexai_env": True,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
    }

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(config))

    assert output[0] == (
        "Invalid Enterprise/Vertex AI configuration: GOOGLE_GENAI_USE_ENTERPRISE or "
        "GOOGLE_GENAI_USE_VERTEXAI must be enabled when use_vertexai_env is true"
    )


def test_adc_path_is_loaded_and_passed_to_vertex_client(monkeypatch, tmp_path):
    credentials_path = tmp_path / "application_default_credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    credentials = object()
    captured = {}

    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setattr(
        google.auth,
        "load_credentials_from_file",
        lambda path, scopes: (credentials, "credentials-project"),
    )

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            part = SimpleNamespace(text="ok", inline_data=None)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    config = {
        "api_key": "",
        "api_version": "v1",
        "use_vertexai_env": True,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": str(credentials_path),
    }

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(config))

    assert output[0] == "ok"
    assert captured["enterprise"] is True
    assert "vertexai" not in captured
    assert captured["credentials"] is credentials
    assert captured["project"] == "credentials-project"


def test_enterprise_environment_takes_precedence_over_legacy_vertex(monkeypatch):
    captured = {}
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    _install_success_client(monkeypatch, captured)

    with pytest.warns(UserWarning, match="GOOGLE_GENAI_USE_ENTERPRISE takes precedence"):
        output = gemini_nodes.SSL_GeminiTextPrompt.execute(
            **_execute_kwargs(_config(use_vertexai_env=True))
        )

    assert output[0] == "ok"
    assert captured["enterprise"] is True
    assert captured["project"] == "test-project"
    assert captured["location"] == "global"


def test_vertex_environment_mode_defaults_to_enterprise_when_selector_is_absent(monkeypatch):
    captured = {}
    monkeypatch.delenv("GOOGLE_GENAI_USE_ENTERPRISE", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    _install_success_client(monkeypatch, captured)

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(
        **_execute_kwargs(_config(use_vertexai_env=True))
    )

    assert output[0] == "ok"
    assert captured["enterprise"] is True


def test_malformed_enterprise_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "yes")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setattr(
        gemini_nodes.genai,
        "Client",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Client must not be created")),
    )

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(
        **_execute_kwargs(_config(use_vertexai_env=True))
    )

    assert output[0] == (
        "Invalid Enterprise/Vertex AI configuration: "
        "GOOGLE_GENAI_USE_ENTERPRISE must be true, false, 1, or 0"
    )


def test_enterprise_express_requires_only_api_key(monkeypatch):
    captured = {}
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    _install_success_client(monkeypatch, captured)

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(
        **_execute_kwargs(_config(api_key="express-key", vertexai_express=True))
    )

    assert output[0] == "ok"
    assert captured["enterprise"] is True
    assert captured["api_key"] == "express-key"
    assert "project" not in captured
    assert "location" not in captured


def test_proxy_uses_sdk_http_options_without_mutating_environment(monkeypatch):
    captured = {}
    monkeypatch.setenv("HTTP_PROXY", "http://existing-proxy:9000")
    _install_success_client(monkeypatch, captured)

    kwargs = _execute_kwargs(_config(api_key="test-key"))
    kwargs.update(use_proxy=True, proxy_host="proxy.example", proxy_port=8080)
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert output[0] == "ok"
    assert captured["http_options"].client_args == {
        "proxy": "http://proxy.example:8080"
    }
    assert gemini_nodes.os.environ["HTTP_PROXY"] == "http://existing-proxy:9000"


def test_confirmed_current_models_are_visible_without_removing_legacy_ids():
    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()
    model_input = next(input_ for input_ in schema.inputs if input_.id == "model")

    for model in (
        "gemini-3.5-flash",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image",
        "gemini-2.5-flash-image-preview",
    ):
        assert model in model_input.options


def test_gemini_4_placeholder_matches_3_6_without_ui_exposure():
    placeholder = gemini_nodes.SSL_GeminiTextPrompt.GEMINI_4_FLASH_PREVIEW
    reference = "gemini-3.6-flash"

    for capability_list in (
        gemini_nodes.SSL_GeminiTextPrompt.THINKING_MODELS,
        gemini_nodes.SSL_GeminiTextPrompt.GEN3_THINKING_MODELS,
        gemini_nodes.SSL_GeminiTextPrompt.IMAGE_MODELS,
        gemini_nodes.SSL_GeminiTextPrompt.MEDIA_RES_MODELS,
    ):
        assert (placeholder in capability_list) == (reference in capability_list)

    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()
    model_input = next(input_ for input_ in schema.inputs if input_.id == "model")
    assert placeholder not in model_input.options


def test_timeout_returns_custom_fallback_without_caching(monkeypatch):
    release_request = threading.Event()

    class BlockingModels:
        @staticmethod
        def generate_content(**kwargs):
            release_request.wait(timeout=1)
            raise RuntimeError("request released")

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = BlockingModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._cache.clear()
    config = {
        "api_key": "test-key",
        "api_version": "v1",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
    }
    kwargs = _execute_kwargs(config)
    kwargs.update(
        use_seed=True,
        seed=123,
        timeout=0.01,
        timeout_fallback_text="configured fallback",
    )

    try:
        output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)
    finally:
        release_request.set()

    assert output[0] == "configured fallback"
    assert gemini_nodes.SSL_GeminiTextPrompt._cache == {}

import hashlib
import threading
from io import BytesIO
from types import SimpleNamespace

import google.auth
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


def test_timeout_fallback_is_last_input():
    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()

    assert schema.inputs[-1].id == "timeout_fallback_text"


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
        ]
    )
    third_socket = torch.full((1, 2, 2, 3), 0.3)

    images = gemini_nodes.SSL_GeminiTextPrompt._flatten_image_inputs(
        {
            "image_3": third_socket,
            "image_1": first_batch,
        }
    )

    assert [round(float(image[0, 0, 0, 0]), 1) for image in images] == [0.1, 0.2, 0.3]


def test_all_ordered_images_are_sent_to_gemini_2_and_3_before_prompt(monkeypatch):
    captured = {}

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
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
        "image_3": torch.full((1, 2, 2, 3), 0.3),
        "image_1": torch.stack(
            [
                torch.full((2, 2, 3), 0.1),
                torch.full((2, 2, 3), 0.2),
            ]
        ),
    }

    def assert_ordered_parts():
        assert len(captured["contents"]) == 4
        assert captured["contents"][-1] == {"text": "test prompt"}
        pixels = [
            Image.open(BytesIO(part["inline_data"]["data"])).getpixel((0, 0))[0]
            for part in captured["contents"][:-1]
        ]
        assert pixels == [25, 51, 76]

    for model in ("gemini-2.0-flash", "gemini-3-flash-preview"):
        captured.clear()
        kwargs["model"] = model
        output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

        assert output[0] == "ok"
        assert_ordered_parts()

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    kwargs["config"] = config | {"use_vertexai_env": True, "api_key": ""}
    kwargs["model"] = "gemini-2.5-flash"
    captured.clear()

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
        "Invalid Vertex AI configuration: GOOGLE_GENAI_USE_VERTEXAI must be enabled "
        "when use_vertexai_env is true"
    )


def test_adc_path_is_loaded_and_passed_to_vertex_client(monkeypatch, tmp_path):
    credentials_path = tmp_path / "application_default_credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    credentials = object()
    captured = {}

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
    assert captured["vertexai"] is True
    assert captured["credentials"] is credentials
    assert captured["project"] == "credentials-project"


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

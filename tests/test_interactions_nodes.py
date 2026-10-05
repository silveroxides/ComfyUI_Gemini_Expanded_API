import base64
import hashlib
import json
import threading
from io import BytesIO
from types import SimpleNamespace

import google.auth
import pytest
import torch
from PIL import Image

from custom_nodes.ComfyUI_Gemini_Expanded_API import interactions_nodes
from custom_nodes.ComfyUI_Gemini_Expanded_API.response_schema import SSL_GeminiResponseSchema, default_state, default_entry


@pytest.fixture
def mock_interactions_client(monkeypatch):
    node = interactions_nodes.SSL_GeminiInteractionsTextPrompt
    for cache in (node._cache, node._client_cache, node._seed_map_cache):
        cache.clear()

    captured = SimpleNamespace(calls=[], clients=[], responses=[], on_client=None)

    class FakeInteractions:
        def create(self, **kwargs):
            captured.calls.append(kwargs)
            resp = captured.responses.pop(0) if captured.responses else {"text": "default response", "id": "int_default_123"}
            if callable(resp):
                resp = resp()
            if isinstance(resp, Exception):
                raise resp

            steps = []
            if "thoughts" in resp:
                steps.append(SimpleNamespace(
                    type="thought",
                    summary=[SimpleNamespace(type="text", text=resp["thoughts"])],
                    text="",
                ))
            if "steps" in resp:
                steps.extend(resp["steps"])
            elif "text" in resp:
                steps.append(SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text=resp["text"])],
                ))

            out_img = None
            if "image_data" in resp:
                out_img = SimpleNamespace(
                    type="image",
                    data=resp["image_data"],
                    mime_type=resp.get("image_mime", "image/png"),
                )

            return SimpleNamespace(
                id=resp.get("id", "int_123"),
                status=resp.get("status", "completed"),
                errors=resp.get("errors", None),
                output_text=resp.get("text", ""),
                output_image=out_img,
                steps=steps,
                usage=SimpleNamespace(total_cached_tokens=resp.get("cached_tokens", 0)) if "cached_tokens" in resp else None,
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured.clients.append(kwargs)
            if captured.on_client:
                captured.on_client()
            self.interactions = FakeInteractions()

    monkeypatch.setattr(interactions_nodes.genai, "Client", FakeClient)
    yield captured
    for cache in (node._cache, node._client_cache, node._seed_map_cache):
        cache.clear()


def _config(**overrides):
    config = {
        "api_key": "test_api_key",
        "api_version": "v1beta",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
    }
    config.update(overrides)
    return config


def _execute_kwargs(config=None, **overrides):
    kwargs = {
        "config": config or _config(),
        "prompt": "test prompt",
        "system_instruction": "test system instruction",
        "model": "gemini-2.5-flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 128,
        "include_images": False,
        "aspect_ratio": "None",
        "bypass_mode": "None",
        "thinking_budget": 0,
        "use_proxy": False,
        "proxy_host": "127.0.0.1",
        "proxy_port": 7890,
        "use_seed": False,
        "seed": 0,
        "timeout": 30,
        "include_thoughts": False,
        "thinking_level": "None",
        "media_resolution": "unspecified",
        "retry_pattern": "",
        "max_retries": 3,
        "timeout_fallback_text": "",
        "image_size": "None",
        "store": True,
        "previous_interaction_id": "",
        "video": None,
        "image_inputs": None,
        "response_schema": None,
    }
    kwargs.update(overrides)
    return kwargs


class FakeVideo:
    def __init__(self, data):
        self.data = data
        self.save_calls = []

    def save_to(self, buffer, format, codec):
        self.save_calls.append((format, codec))
        buffer.write(self.data)


def test_interactions_define_schema():
    schema = interactions_nodes.SSL_GeminiInteractionsTextPrompt.define_schema()
    assert schema.node_id == "SSL_GeminiInteractionsTextPrompt"
    assert schema.category == "API/Gemini/Interactions"
    input_names = [i.id for i in schema.inputs]
    assert "store" in input_names
    assert "previous_interaction_id" in input_names
    assert "response_schema" in input_names
    assert "image_inputs" in input_names

    output_names = [o.id for o in schema.outputs]
    assert output_names == ["text", "image", "final_actual_seed", "structured_output", "thoughts", "interaction_id"]


def test_interactions_api_key_config_node():
    schema = interactions_nodes.SSL_GeminiInteractionsAPIKeyConfig.define_schema()
    assert schema.node_id == "SSL_GeminiInteractionsAPIKeyConfig"
    assert schema.category == "API/Gemini/Interactions"

    output = interactions_nodes.SSL_GeminiInteractionsAPIKeyConfig.execute(
        api_key="secret123",
        api_version="v1beta",
        use_vertexai_env=False,
        vertexai_express=True,
    )
    config = output[0]
    assert config["api_key"] == "secret123"
    assert config["api_version"] == "v1beta"
    assert config["vertexai_express"] is True


def test_basic_text_generation(mock_interactions_client):
    mock_interactions_client.responses = [{"text": "Hello world!", "id": "int_abc_123"}]
    args = _execute_kwargs()
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Hello world!"  # text
    assert output[5] == "int_abc_123"   # interaction_id
    assert len(mock_interactions_client.calls) == 1

    call = mock_interactions_client.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["input"] == "test prompt"
    assert call["system_instruction"] == "test system instruction"
    assert call["store"] is True
    assert "safety_settings" in call
    assert len(call["safety_settings"]) == 10
    assert {s["type"] for s in call["safety_settings"]} == set(interactions_nodes.SSL_GeminiInteractionsTextPrompt.ALL_HARM_CATEGORIES)
    assert all(s["threshold"] == "block_none" for s in call["safety_settings"])


def test_multimodal_image_input(mock_interactions_client):
    mock_interactions_client.responses = [{"text": "Found a square in image", "id": "int_img_1"}]
    image_tensor = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    args = _execute_kwargs(
        image_inputs={"image_1": image_tensor},
        prompt="Describe this image",
        media_resolution="high",
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Found a square in image"
    call = mock_interactions_client.calls[0]
    assert isinstance(call["input"], list)
    assert len(call["input"]) == 2
    img_item = call["input"][0]
    assert img_item["type"] == "image"
    assert img_item["mime_type"] == "image/png"
    assert "data" in img_item
    assert img_item["resolution"] == "high"
    text_item = call["input"][1]
    assert text_item["type"] == "text"
    assert text_item["text"] == "Describe this image"


def test_multimodal_video_input(mock_interactions_client):
    mock_interactions_client.responses = [{"text": "Found motion in video", "id": "int_vid_1"}]
    video_data = b"fake mp4 video bytes"
    args = _execute_kwargs(
        video={"video": FakeVideo(video_data), "fps": 2, "pad_at_start": False, "duration_aware_padding": False},
        prompt="What happened in the video?",
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Found motion in video"
    call = mock_interactions_client.calls[0]
    assert isinstance(call["input"], list)
    assert len(call["input"]) == 2
    vid_item = call["input"][0]
    assert vid_item["type"] == "video"
    assert vid_item["mime_type"] == "video/mp4"
    assert base64.b64decode(vid_item["data"]) == video_data


def test_structured_output_with_schema(mock_interactions_client):
    schema = {
        "type": "object",
        "properties": {"animal": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["animal", "count"],
    }
    mock_interactions_client.responses = [{"text": '{"animal": "cat", "count": 3}', "id": "int_struct_1"}]
    args = _execute_kwargs(response_schema=schema)
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == '{"animal": "cat", "count": 3}'
    assert output[3] == '{"animal": "cat", "count": 3}'  # structured_output
    call = mock_interactions_client.calls[0]
    assert "response_format" in call
    assert call["response_format"] == [
        {"type": "text", "mime_type": "application/json", "schema": schema}
    ]


def test_image_generation(mock_interactions_client):
    # Create small valid 16x16 PNG image
    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64_png = base64.b64encode(buf.getvalue()).decode("utf-8")

    mock_interactions_client.responses = [{
        "text": "Generated an image",
        "image_data": b64_png,
        "image_mime": "image/png",
        "id": "int_img_gen_1",
    }]

    args = _execute_kwargs(
        model="gemini-3.1-flash-image",
        include_images=True,
        aspect_ratio="16:9",
        image_size="1K",
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    image_tensor = output[1]
    assert torch.is_tensor(image_tensor)
    assert image_tensor.shape == (1, 16, 16, 3)

    call = mock_interactions_client.calls[0]
    assert call["response_format"] == [
        {"type": "image", "aspect_ratio": "16:9", "image_size": "1K"}
    ]


def test_thinking_and_thoughts_extraction(mock_interactions_client):
    mock_interactions_client.responses = [{
        "text": "42 is the answer",
        "thoughts": "I need to calculate the answer carefully.",
        "id": "int_think_1",
    }]

    args = _execute_kwargs(
        model="gemini-3.8-flash",
        thinking_level="high",
        include_thoughts=True,
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "42 is the answer"
    assert output[4] == "I need to calculate the answer carefully."

    call = mock_interactions_client.calls[0]
    assert call["generation_config"]["thinking_level"] == "high"
    assert call["generation_config"]["thinking_summaries"] == "auto"


def test_multi_turn_chaining(mock_interactions_client):
    mock_interactions_client.responses = [
        {"text": "Turn 1 answer", "id": "int_turn_1"},
        {"text": "Turn 2 answer", "id": "int_turn_2"},
    ]

    # First turn
    args1 = _execute_kwargs(prompt="First message", store=True)
    out1 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args1)
    id1 = out1[5]
    assert id1 == "int_turn_1"
    assert mock_interactions_client.calls[0].get("previous_interaction_id") is None

    # Second turn continuing first
    args2 = _execute_kwargs(prompt="Follow-up message", previous_interaction_id=id1, store=True)
    out2 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args2)
    assert out2[0] == "Turn 2 answer"
    assert out2[5] == "int_turn_2"
    assert mock_interactions_client.calls[1]["previous_interaction_id"] == "int_turn_1"


def test_vertexai_enterprise_auth(mock_interactions_client, monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    config = _config(use_vertexai_env=True)
    args = _execute_kwargs(config=config)
    interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    client_args = mock_interactions_client.clients[0]
    assert client_args.get("enterprise") is True
    assert client_args.get("project") == "test-project"
    assert client_args.get("location") == "us-central1"


def test_vertexai_express_auth(mock_interactions_client):
    config = _config(vertexai_express=True, api_key="my_express_key")
    args = _execute_kwargs(config=config)
    interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    client_args = mock_interactions_client.clients[0]
    assert client_args.get("enterprise") is True
    assert client_args.get("api_key") == "my_express_key"


def test_local_cache_hit_on_same_seed(mock_interactions_client):
    mock_interactions_client.responses = [
        {"text": "Cached response", "id": "int_cache_1"},
    ]

    args = _execute_kwargs(use_seed=True, seed=42)
    out1 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)
    assert out1[0] == "Cached response"
    assert len(mock_interactions_client.calls) == 1

    # Second call with same seed and params should hit local cache
    out2 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)
    assert out2[0] == "Cached response"
    assert len(mock_interactions_client.calls) == 1


def test_retry_pattern_triggers_retry(mock_interactions_client):
    mock_interactions_client.responses = [
        {"text": "Please retry me", "id": "int_r1"},
        {"text": "Final good answer", "id": "int_r2"},
    ]

    args = _execute_kwargs(
        retry_pattern="retry me",
        max_retries=2,
        use_seed=True,
        seed=100,
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Final good answer"
    assert len(mock_interactions_client.calls) == 2


def test_timeout_fallback(mock_interactions_client, monkeypatch):
    def slow_create(**kwargs):
        import time
        time.sleep(1.0)
        return SimpleNamespace(id="slow", output_text="slow")

    class SlowInteractions:
        def create(self, **kwargs):
            return slow_create(**kwargs)

    class SlowClient:
        def __init__(self, **kwargs):
            self.interactions = SlowInteractions()

    monkeypatch.setattr(interactions_nodes.genai, "Client", SlowClient)
    args = _execute_kwargs(timeout=0.1, timeout_fallback_text="Custom timeout message")
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Custom timeout message"


def test_interactions_api_key_config_with_cache():
    output = interactions_nodes.SSL_GeminiInteractionsAPIKeyConfig.execute(
        api_key="secret123",
        api_version="v1beta",
        use_vertexai_env=False,
        vertexai_express=False,
        use_cache=True,
        cache_ttl_minutes=120,
        cache_seed=999,
    )
    config = output[0]
    assert config["use_cache"] is True
    assert config["cache_ttl_minutes"] == 120
    assert config["cache_seed"] == 999


def test_caching_with_cache_seed_and_usage(mock_interactions_client):
    mock_interactions_client.responses = [
        {"text": "Cached response", "id": "int_cache_hit_1", "cached_tokens": 4096},
    ]

    config = _config(use_cache=True, cache_seed=777)
    args = _execute_kwargs(config=config, use_seed=True, seed=123)
    out1 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert out1[0] == "Cached response"
    assert out1[2] == 777  # actual_seed matches cache_seed when use_cache=True
    assert len(mock_interactions_client.calls) == 1
    assert mock_interactions_client.calls[0]["generation_config"]["seed"] == 777

    # Second call with identical params hits local result cache
    out2 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)
    assert out2[0] == "Cached response"
    assert out2[2] == 777
    assert len(mock_interactions_client.calls) == 1


def test_safety_config_node():
    schema = interactions_nodes.SSL_GeminiInteractionsSafetyConfig.define_schema()
    assert schema.node_id == "SSL_GeminiInteractionsSafetyConfig"
    assert schema.category == "API/Gemini/Interactions"

    output = interactions_nodes.SSL_GeminiInteractionsSafetyConfig.execute(
        harassment="off",
        hate_speech="block_none",
        jailbreak="block_only_high",
        method="severity",
    )
    settings = output[0]
    assert isinstance(settings, list)
    settings_dict = {s["type"]: s for s in settings}
    assert settings_dict["harassment"]["threshold"] == "off"
    assert settings_dict["harassment"]["method"] == "severity"
    assert settings_dict["hate_speech"]["threshold"] == "block_none"
    assert settings_dict["jailbreak"]["threshold"] == "block_only_high"


def test_safety_preferences_off_and_method(mock_interactions_client):
    mock_interactions_client.responses = [{"text": "Unrestricted response", "id": "int_safe_1"}]
    args = _execute_kwargs(safety_level="off", safety_method="severity")
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Unrestricted response"
    call = mock_interactions_client.calls[0]
    assert "safety_settings" in call
    assert len(call["safety_settings"]) == 10
    assert all(s["threshold"] == "off" for s in call["safety_settings"])
    assert all(s["method"] == "severity" for s in call["safety_settings"])


def test_safety_config_input_override(mock_interactions_client):
    mock_interactions_client.responses = [{"text": "Granular safety response", "id": "int_safe_2"}]
    custom_safety = [
        {"type": "harassment", "threshold": "block_only_high"},
        {"type": "jailbreak", "threshold": "off"},
    ]
    args = _execute_kwargs(
        safety_level="block_none",  # Should be overridden by safety_config
        safety_config=custom_safety,
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Granular safety response"
    call = mock_interactions_client.calls[0]
    assert call["safety_settings"] == custom_safety


def test_safety_block_error_reporting(mock_interactions_client):
    mock_interactions_client.responses = [{
        "text": "",
        "id": "int_blocked",
        "errors": [SimpleNamespace(code="SAFETY", message="Content was blocked by safety policy")],
        "status": "failed",
    }]

    args = _execute_kwargs()
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0].startswith("API call/processing error:")
    assert "Content was blocked by safety policy" in output[0]



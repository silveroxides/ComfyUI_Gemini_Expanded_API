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

    tmp_store = {"last_interaction_id": "", "sessions": []}
    monkeypatch.setattr(interactions_nodes.GeminiInteractionsStoreManager, "load", lambda: json.loads(json.dumps(tmp_store)))
    def fake_save(d):
        tmp_store.clear()
        tmp_store.update(json.loads(json.dumps(d)))
    monkeypatch.setattr(interactions_nodes.GeminiInteractionsStoreManager, "save", fake_save)

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
    assert len(call["safety_settings"]) == 9
    assert {s["type"] for s in call["safety_settings"]} == set(interactions_nodes.SSL_GeminiInteractionsTextPrompt.CORE_HARM_CATEGORIES)
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
    assert isinstance(img_item["data"], BytesIO)
    assert img_item["data"].tell() == 0
    assert len(img_item["data"].getvalue()) > 0
    assert img_item["resolution"] == "high"

    from google.genai._gaos.types.interactions.imagecontent import ImageContent
    validated_img = ImageContent(**img_item)
    assert isinstance(validated_img.data, str)
    assert len(validated_img.data) > 0
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
    assert isinstance(vid_item["data"], BytesIO)
    assert vid_item["data"].getvalue() == video_data

    from google.genai._gaos.types.interactions.videocontent import VideoContent
    validated_vid = VideoContent(**vid_item)
    assert base64.b64decode(validated_vid.data) == video_data


def test_multimodal_image_and_video_ordering(mock_interactions_client):
    mock_interactions_client.responses = [{"text": "Found image and video", "id": "int_both_1"}]
    video_data = b"fake video bytes"
    image_tensor = torch.zeros(1, 16, 16, 3)
    args = _execute_kwargs(
        video={"video": FakeVideo(video_data), "fps": 2, "pad_at_start": False, "duration_aware_padding": False},
        image_inputs={"image_1": image_tensor},
        prompt="Describe image and video",
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)
    assert output[0] == "Found image and video"
    call = mock_interactions_client.calls[0]
    items = call["input"]
    assert len(items) == 3
    assert items[0]["type"] == "image"
    assert items[1]["type"] == "video"
    assert items[2]["type"] == "text"


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


def test_image_generation(mock_interactions_client, monkeypatch):
    called_with = []
    fake_img = torch.zeros((1, 16, 16, 3), dtype=torch.float32)

    def fake_generate(**kwargs):
        called_with.append(kwargs)
        return interactions_nodes.IO.NodeOutput("Generated an image", fake_img, 0, "", "")

    monkeypatch.setattr(interactions_nodes.SSL_GeminiTextPrompt, "execute", fake_generate)

    args = _execute_kwargs(
        model="gemini-3.1-flash-image",
        include_images=True,
        aspect_ratio="16:9",
        image_size="1K",
    )
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Generated an image"
    image_tensor = output[1]
    assert torch.is_tensor(image_tensor)
    assert image_tensor.shape == (1, 16, 16, 3)
    assert output[5] == ""  # interaction_id is empty for rerouted image models
    assert len(called_with) == 1
    assert called_with[0]["model"] == "gemini-3.1-flash-image"
    assert called_with[0]["include_images"] is True
    assert called_with[0]["aspect_ratio"] == "16:9"
    assert called_with[0]["image_size"] == "1K"


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


def test_interactions_store_node():
    schema = interactions_nodes.SSL_GeminiInteractionsStore.define_schema()
    assert schema.node_id == "SSL_GeminiInteractionsStore"
    assert schema.category == "API/Gemini/Interactions"

    # Initially empty
    out = interactions_nodes.SSL_GeminiInteractionsStore.execute(clear_history=False)
    assert "No interactions stored" in out[0]
    assert out[1] == ""

    # Record some sessions
    interactions_nodes.GeminiInteractionsStoreManager.record_interaction("int_test_1", "gemini-2.5-flash")
    interactions_nodes.GeminiInteractionsStoreManager.record_interaction("int_test_1", "gemini-2.5-flash")
    interactions_nodes.GeminiInteractionsStoreManager.record_interaction("int_test_2", "gemini-3.8-flash")

    out2 = interactions_nodes.SSL_GeminiInteractionsStore.execute(clear_history=False, limit=5)
    table, last_id = out2[0], out2[1]
    assert "int_test_2" in table
    assert "gemini-3.8-flash" in table
    assert last_id == "int_test_2"

    # Test clear_history
    out3 = interactions_nodes.SSL_GeminiInteractionsStore.execute(clear_history=True)
    assert out3[0] == "History cleared."
    assert out3[1] == ""
    assert interactions_nodes.GeminiInteractionsStoreManager.get_last_interaction_id() == ""


def test_auto_store_and_continue_interaction(mock_interactions_client):
    mock_interactions_client.responses = [
        {"text": "Turn 1 answer", "id": "int_turn_1"},
        {"text": "Turn 2 answer", "id": "int_turn_2"},
    ]

    # First turn: store=True, previous_interaction_id="" -> creates new interaction
    args1 = _execute_kwargs(prompt="First message", store=True, previous_interaction_id="")
    out1 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args1)
    assert out1[0] == "Turn 1 answer"
    assert out1[5] == "int_turn_1"
    assert "previous_interaction_id" not in mock_interactions_client.calls[0]
    assert interactions_nodes.GeminiInteractionsStoreManager.get_last_interaction_id() == "int_turn_1"

    # Second turn with DIFFERENT prompt: store=True, previous_interaction_id="" -> auto-continues with int_turn_1
    args2 = _execute_kwargs(prompt="Second message", store=True, previous_interaction_id="")
    out2 = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args2)
    assert out2[0] == "Turn 2 answer"
    assert out2[5] == "int_turn_2"
    assert mock_interactions_client.calls[1]["previous_interaction_id"] == "int_turn_1"
    assert interactions_nodes.GeminiInteractionsStoreManager.get_last_interaction_id() == "int_turn_2"


def test_expired_interaction_fallback(mock_interactions_client):
    # First call with expired ID fails with 404/not found, then retry succeeds with new interaction
    mock_interactions_client.responses = [
        RuntimeError("404: Interaction not found or expired"),
        {"text": "Fresh interaction response", "id": "int_fresh_1"},
    ]

    args = _execute_kwargs(prompt="Continue expired chat", store=True, previous_interaction_id="int_old_expired")
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Fresh interaction response"
    assert output[5] == "int_fresh_1"
    assert len(mock_interactions_client.calls) == 2
    # First attempt tried expired ID
    assert mock_interactions_client.calls[0]["previous_interaction_id"] == "int_old_expired"
    # Fallback attempt cleared previous_interaction_id
    assert "previous_interaction_id" not in mock_interactions_client.calls[1]


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


def test_safety_config_node_and_override(mock_interactions_client):
    schema = interactions_nodes.SSL_GeminiInteractionsSafetyConfig.define_schema()
    assert schema.node_id == "SSL_GeminiInteractionsSafetyConfig"
    assert schema.category == "API/Gemini/Interactions"

    # Default execution of safety config produces off + severity
    safety_out = interactions_nodes.SSL_GeminiInteractionsSafetyConfig.execute()
    custom_safety = safety_out[0]
    assert isinstance(custom_safety, list)
    assert len(custom_safety) == 9  # default jailbreak is 'default' so omitted
    assert all(s["threshold"] == "off" for s in custom_safety)
    assert all(s["method"] == "severity" for s in custom_safety)

    # Wire to API key config
    key_config_out = interactions_nodes.SSL_GeminiInteractionsAPIKeyConfig.execute(
        api_key="test_key",
        api_version="v1beta",
        use_vertexai_env=False,
        vertexai_express=False,
        safety_override=custom_safety,
    )
    config = key_config_out[0]
    assert config["safety_override"] == custom_safety

    # Execute text prompt with this config
    mock_interactions_client.responses = [{"text": "Unrestricted custom response", "id": "int_safe_custom"}]
    args = _execute_kwargs(config=config)
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Unrestricted custom response"
    call = mock_interactions_client.calls[0]
    assert call["safety_settings"] == custom_safety


def test_enterprise_auto_safety_resolution(mock_interactions_client):
    # When vertexai_express or use_vertexai_env is True and no override is given,
    # auto-resolves to 'off' with 'severity'
    mock_interactions_client.responses = [{"text": "Enterprise response", "id": "int_ent_1"}]
    key_config_out = interactions_nodes.SSL_GeminiInteractionsAPIKeyConfig.execute(
        api_key="test_key",
        api_version="v1beta",
        use_vertexai_env=False,
        vertexai_express=True,
    )
    args = _execute_kwargs(config=key_config_out[0])
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Enterprise response"
    call = mock_interactions_client.calls[0]
    assert "safety_settings" in call
    assert len(call["safety_settings"]) == 9
    assert all(s["threshold"] == "off" for s in call["safety_settings"])
    assert all(s["method"] == "severity" for s in call["safety_settings"])


def test_standard_auto_safety_resolution(mock_interactions_client):
    # When using consumer API (non-enterprise) and no override is given,
    # auto-resolves to 'block_none' without forcing 'off' (which Google AI Studio rejects)
    mock_interactions_client.responses = [{"text": "Standard response", "id": "int_std_1"}]
    key_config_out = interactions_nodes.SSL_GeminiInteractionsAPIKeyConfig.execute(
        api_key="test_key",
        api_version="v1beta",
        use_vertexai_env=False,
        vertexai_express=False,
    )
    args = _execute_kwargs(config=key_config_out[0])
    output = interactions_nodes.SSL_GeminiInteractionsTextPrompt.execute(**args)

    assert output[0] == "Standard response"
    call = mock_interactions_client.calls[0]
    assert "safety_settings" in call
    assert len(call["safety_settings"]) == 9
    assert all(s["threshold"] == "block_none" for s in call["safety_settings"])
    assert all("method" not in s for s in call["safety_settings"])



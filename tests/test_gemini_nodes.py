import hashlib
import json
import threading
from io import BytesIO
from types import SimpleNamespace

import google.auth
import pytest
import torch
from PIL import Image

from custom_nodes.ComfyUI_Gemini_Expanded_API import gemini_nodes
from custom_nodes.ComfyUI_Gemini_Expanded_API.response_schema import SSL_GeminiResponseSchema, default_state, default_entry


@pytest.fixture
def structured_client(monkeypatch):
    node = gemini_nodes.SSL_GeminiTextPrompt
    for cache in (node._cache, node._client_cache, node._context_cache, node._seed_map_cache):
        cache.clear()
    captured = SimpleNamespace(calls=[], creations=[], clients=[], responses=[], cache_error=None, on_client=None)

    class Models:
        def generate_content(self, **kwargs):
            captured.calls.append({**kwargs, "config": kwargs["config"].model_copy(deep=True)})
            response = captured.responses.pop(0) if captured.responses else [('final', '{"answer":"ok"}')]
            if callable(response):
                response = response()
            if isinstance(response, Exception):
                raise response
            parts = [SimpleNamespace(text=text, thought=kind == "thought", inline_data=None) for kind, text in response]
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])

    class Caches:
        def create(self, **kwargs):
            captured.creations.append(kwargs)
            if captured.cache_error:
                raise captured.cache_error
            return SimpleNamespace(name=f"cachedContents/{len(captured.creations)}")

    class Client:
        def __init__(self, **kwargs):
            captured.clients.append(kwargs)
            if captured.on_client:
                captured.on_client()
            self.models, self.caches = Models(), Caches()

    monkeypatch.setattr(gemini_nodes.genai, "Client", Client)
    yield captured
    for cache in (node._cache, node._client_cache, node._context_cache, node._seed_map_cache):
        cache.clear()


@pytest.mark.parametrize("model, settings", [
    ("gemini-2.5-flash", {}), ("gemini-3.1-pro-preview", {"thinking_level": "high"}),
    (gemini_nodes.SSL_GeminiTextPrompt.GEMINI_3_7_FLASH, {"thinking_level": "medium"}),
    (gemini_nodes.SSL_GeminiTextPrompt.GEMINI_3_8_FLASH, {"thinking_level": "medium"}),
])
@pytest.mark.parametrize("client_config", [{}, {"vertexai_express": True}, {"use_vertexai_env": True, "vertexai_project": "test", "vertexai_location": "global"}])
def test_builder_schema_reaches_generate_content(structured_client, monkeypatch, model, settings, client_config):
    target = gemini_nodes.SSL_GeminiTextPrompt.define_schema().inputs[-2]
    assert target.get_io_type() == SSL_GeminiResponseSchema.define_schema().outputs[0].get_io_type() == "GEMINI_RESPONSE_SCHEMA"
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    state = default_state()
    state["entries"]["n1"] = default_entry("string")
    state["entries"]["n1"]["name"] = "answer"
    state["entries"]["n0"]["fields"] = ["n1"]
    state["next_id"] = 2
    schema = SSL_GeminiResponseSchema.execute(json.dumps(state))[0]
    args = _execute_kwargs(_config(api_key="test", **client_config))
    args.update(model=model, response_schema=schema, include_thoughts=True, **settings)
    structured_client.responses = [[("thought", "thinking"), ("final", '{"answer":'), ("final", ' "ok"}')]]
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    assert output[0] == output[3] == '{"answer": "ok"}'
    assert output[4] == "thinking"
    sent = structured_client.calls[0]["config"]
    assert sent.response_mime_type == "application/json"
    assert sent.response_json_schema == schema
    assert sent.response_schema is None


@pytest.mark.parametrize("schema", [None, {}])
def test_absent_schema_is_distinct_from_empty_schema(structured_client, schema):
    structured_client.responses = [[("thought", "separate"), ("final", "{}")]]
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(_config()), response_schema=schema)
    assert output[0] == "{}"
    assert output[3] == ("" if schema is None else "{}")
    assert output[4] == "separate"
    assert structured_client.calls[0]["config"].response_mime_type == (None if schema is None else "application/json")


@pytest.mark.parametrize("text", [' {"x": 1}\n', '[1, 2]', '"hello"', '1', '1.2', 'true', 'null'])
def test_structured_json_keeps_original_text(structured_client, text):
    structured_client.responses = [[("final", text)]]
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(_config()), response_schema={})
    assert output[0] == output[3] == text


@pytest.mark.parametrize("text", ["", "{", '```json\n{}\n```', "NaN", "Infinity", "-Infinity", '{"x":NaN}'])
def test_invalid_structured_answer_is_not_cached(structured_client, text):
    structured_client.responses = [[("thought", "must not leak"), ("final", text)]]
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(_config()), response_schema={}, use_seed=True, seed=4)
    assert output[0].startswith("API call/processing error:")
    assert output[3] == output[4] == ""
    assert not gemini_nodes.SSL_GeminiTextPrompt._cache


def test_schema_snapshot_and_result_cache(structured_client):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    structured_client.on_client = lambda: schema["properties"].update(extra={"type": "number"})
    structured_client.responses = [[("thought", "saved thought"), ("final", '{"answer":"ok"}')]]
    args = _execute_kwargs(_config())
    args.update(use_seed=True, seed=7, response_schema=schema)
    first = gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    sent = structured_client.calls[0]["config"].response_json_schema
    assert "extra" not in sent["properties"]
    args["response_schema"] = sent
    second = gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    assert len(structured_client.calls) == 1
    assert second[3] == first[3] and second[4] == "saved thought"
    assert len(next(iter(gemini_nodes.SSL_GeminiTextPrompt._cache.values()))) == 5
    args["response_schema"] = schema
    gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    assert len(structured_client.calls) == 2
    args["response_schema"] = None
    assert gemini_nodes.SSL_GeminiTextPrompt.execute(**args)[3] == ""
    assert len(structured_client.calls) == 3


@pytest.mark.parametrize("failure", [False, True])
def test_schema_changes_reuse_context_and_survive_cache_fallback(structured_client, failure):
    if failure:
        structured_client.cache_error = RuntimeError("cache unavailable")
    args = _execute_kwargs(_config(use_cache=True))
    for schema in ({"type": "object"}, {"type": "object", "description": "changed"}):
        gemini_nodes.SSL_GeminiTextPrompt.execute(**args, response_schema=schema)
        assert structured_client.calls[-1]["config"].response_json_schema == schema
        assert structured_client.calls[-1]["config"].response_mime_type == "application/json"
    assert len(structured_client.creations) == (2 if failure else 1)
    if not failure:
        assert structured_client.calls[0]["config"].cached_content == structured_client.calls[1]["config"].cached_content


def test_expired_context_recreation_keeps_schema(structured_client):
    structured_client.responses = [RuntimeError("cached content expired"), [("final", "{}")]]
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(_config(use_cache=True)), response_schema={})
    assert output[3] == "{}"
    assert len(structured_client.creations) == 2
    assert all(call["config"].response_json_schema == {} for call in structured_client.calls)


def test_regex_retry_uses_only_final_text_and_keeps_latest_thoughts(structured_client):
    structured_client.responses = [
        [("thought", "old"), ("final", '{"answer":"retry"}')],
        [("thought", "retry in thought only"), ("final", '{"answer":"done"}')],
    ]
    args = _execute_kwargs(_config())
    args.update(response_schema={}, use_seed=True, seed=5, retry_pattern="retry", max_retries=2)
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    assert len(structured_client.calls) == 2
    assert output[3] == '{"answer":"done"}'
    assert output[4] == "retry in thought only"
    assert all(call["config"].response_json_schema == {} for call in structured_client.calls)
    key = next(iter(gemini_nodes.SSL_GeminiTextPrompt._seed_map_cache))
    assert key[1][-1] == "{}"
    assert len(key[1]) == len(next(iter(gemini_nodes.SSL_GeminiTextPrompt._cache)))


def test_failed_retry_and_outer_exception_are_not_cached(structured_client, monkeypatch):
    node = gemini_nodes.SSL_GeminiTextPrompt
    structured_client.responses = [[("thought", "old"), ("final", '"retry"')], RuntimeError("failed")]
    args = _execute_kwargs(_config())
    args.update(response_schema={}, use_seed=True, seed=5, retry_pattern="retry", max_retries=1)
    output = node.execute(**args)
    assert output[3] == output[4] == ""
    assert not node._cache and not node._seed_map_cache
    def broken(**_kwargs):
        raise RuntimeError("config failed")
    monkeypatch.setattr(node, "_build_generate_content_config", broken)
    output = node.execute(**args)
    assert output[0].startswith("Unhandled error:")
    assert output[3] == output[4] == ""
    assert not node._cache


@pytest.mark.parametrize("schema", [[], {"bad": float("nan")}, {"bad": object()}])
def test_bad_schema_fails_before_client_creation(structured_client, schema):
    with pytest.raises((ValueError, TypeError)):
        gemini_nodes.SSL_GeminiTextPrompt.execute(**_execute_kwargs(_config()), response_schema=schema)
    assert not structured_client.clients


def test_schema_rejects_effective_image_generation_only(structured_client):
    args = _execute_kwargs(_config())
    args.update(include_images=True, response_schema={}, model="gemini-3-pro-image")
    with pytest.raises(ValueError, match="image generation"):
        gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    assert not structured_client.clients
    args.update(model="gemini-2.5-flash", image_inputs={"image_1": torch.zeros(1, 8, 8, 3)})
    assert gemini_nodes.SSL_GeminiTextPrompt.execute(**args)[3] == '{"answer":"ok"}'


def test_schema_order_and_request_settings_isolate_retry_seed_keys(structured_client):
    node = gemini_nodes.SSL_GeminiTextPrompt
    args = _execute_kwargs(_config())
    args.update(model="gemini-3.1-pro-preview", use_seed=True, seed=7, retry_pattern="never", max_retries=1,
                response_schema={"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}, thinking_level="low")
    node.execute(**args)
    for change in [
        {"response_schema": {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}},
        {"thinking_level": "high"}, {"media_resolution": "high"}, {"max_retries": 2}, {"retry_pattern": "different"},
    ]:
        node.execute(**{**args, **change})
    assert len(node._seed_map_cache) == len(node._cache) == len(structured_client.calls) == 6


def test_success_then_retry_timeout_discards_outputs(structured_client):
    release = threading.Event()
    completed = threading.Event()
    def block():
        try:
            release.wait(timeout=1)
            raise RuntimeError("released")
        finally:
            completed.set()
    structured_client.responses = [[("thought", "first"), ("final", '"retry"')], block]
    args = _execute_kwargs(_config())
    args.update(response_schema={}, use_seed=True, seed=4, retry_pattern="retry", max_retries=1, timeout=0.01,
                timeout_fallback_text="fallback")
    try:
        output = gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    finally:
        release.set()
        completed.wait(timeout=1)
    assert output[0] == "fallback"
    assert output[3] == output[4] == ""
    assert not gemini_nodes.SSL_GeminiTextPrompt._cache
    assert not gemini_nodes.SSL_GeminiTextPrompt._seed_map_cache




def test_cache_seed_explicitly_enables_seed_widget_control():
    inputs = gemini_nodes.SSL_GeminiAPIKeyConfig.INPUT_TYPES()
    kind, options = inputs["optional"]["cache_seed"]
    assert kind == "INT"
    assert options["control_after_generate"] is True
    assert options["min"] == 0
    assert options["max"] == 2147483647


@pytest.mark.parametrize("use_cache", [False, True])
def test_torch_seed_generation_preserves_request_and_fingerprint(structured_client, monkeypatch, use_cache):
    node = gemini_nodes.SSL_GeminiTextPrompt
    def forbidden(*args, **kwargs):
        raise AssertionError("Seed handling must not use global random generators")
    monkeypatch.setattr(gemini_nodes.random, "randint", forbidden)
    monkeypatch.setattr(gemini_nodes.random, "seed", forbidden)
    monkeypatch.setattr(gemini_nodes.np.random, "seed", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    before = torch.get_rng_state().clone()
    generated = node._handle_seed(True, 0)
    assert 0 <= generated < 2147483647
    assert node._handle_seed(False, 7) is None
    args = _execute_kwargs(_config(use_cache=use_cache, cache_seed=123))
    args.update(use_seed=True, seed=456)
    first = node.execute(**args)
    second = node.execute(**args)
    assert first[2] == second[2] == (123 if use_cache else 456)
    assert len(structured_client.calls) == 1
    assert structured_client.calls[0]["config"].seed == first[2]
    node._cache.clear()
    structured_client.calls.clear()
    structured_client.responses = [[("final", "retry")], [("final", "done")]]
    result = node.execute(**args, retry_pattern="retry", max_retries=1)
    assert len(structured_client.calls) == 2
    assert 0 <= result[2] < 2147483647
    assert structured_client.calls[-1]["config"].seed == result[2]
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("value", [130610794845696, 450167232274663, 303772010844099, 854079365259032])
@pytest.mark.parametrize("use_cache", [False, True])
def test_high_precision_seed_is_retained_in_fingerprint_but_mapped_for_api(structured_client, use_cache, value):
    node = gemini_nodes.SSL_GeminiTextPrompt
    args = _execute_kwargs(_config(use_cache=use_cache, cache_seed=value))
    args.update(use_seed=True, seed=value)
    first = node.execute(**args)
    wire_seed = structured_client.calls[-1]["config"].seed
    assert 0 <= wire_seed < 2**31
    assert first[2] == wire_seed
    fingerprint = next(iter(node._cache))
    assert fingerprint[14] == value
    if use_cache:
        assert fingerprint[17] == value
    node.execute(**args)
    assert len(structured_client.calls) == 1
    node._cache.clear()
    assert node.execute(**args)[2] == wire_seed
    assert structured_client.calls[-1]["config"].seed == wire_seed


def test_unseeded_image_branch_also_respects_wire_seed_range(structured_client):
    args = _execute_kwargs(_config())
    args.update(model="gemini-3-pro-image", include_images=True, use_seed=False, seed=854079365259032)
    gemini_nodes.SSL_GeminiTextPrompt.execute(**args)
    assert 0 <= structured_client.calls[-1]["config"].seed < 2**31


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


class FakeVideo:
    def __init__(self, data):
        self.data = data
        self.save_calls = []

    def save_to(self, buffer, format, codec):
        self.save_calls.append((format, codec))
        buffer.write(self.data)


def _video_config(data, fps=1):
    return {"video": FakeVideo(data), "fps": fps}


def test_image_autogrow_is_last_input():
    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()

    assert schema.inputs[-3].id == "video"
    assert schema.inputs[-2].id == "response_schema"
    assert schema.inputs[-2].optional is True
    assert schema.inputs[-1].id == "image_inputs"
    assert "response_schema" in gemini_nodes.SSL_GeminiTextPrompt.INPUT_TYPES()["optional"]
    assert [output.id for output in schema.outputs] == ["text", "image", "final_actual_seed", "structured_output", "thoughts"]


def test_video_config_exposes_native_video_and_integer_fps():
    schema = gemini_nodes.SSL_GeminiVideoConfig.define_schema()
    fps = next(input_ for input_ in schema.inputs if input_.id == "fps")
    pad_at_start = next(input_ for input_ in schema.inputs if input_.id == "pad_at_start")
    duration_aware_padding = next(input_ for input_ in schema.inputs if input_.id == "duration_aware_padding")

    assert [input_.id for input_ in schema.inputs] == ["video", "fps", "pad_at_start", "duration_aware_padding"]
    assert fps.default == 1
    assert fps.min == 1
    assert fps.max == 24
    assert pad_at_start.default is False
    assert duration_aware_padding.default is False
    native_video = object()
    output = gemini_nodes.SSL_GeminiVideoConfig.execute(native_video, 2, True, True)
    assert output[0] == {
        "video": native_video,
        "fps": 2,
        "pad_at_start": True,
        "duration_aware_padding": True,
    }


def test_video_start_padding_matches_resolution_and_half_sampling_interval(monkeypatch):
    images = torch.ones((4, 3, 5, 3))
    waveform = torch.ones((1, 2, 8000))
    video = SimpleNamespace(
        get_components=lambda: gemini_nodes.Types.VideoComponents(
            images=images,
            audio={"waveform": waveform, "sample_rate": 8000},
            frame_rate=4,
        ),
        get_bit_depth=lambda: 8,
        get_color_space=lambda: "sRGB",
    )
    captured = {}

    class CapturingVideo:
        def __init__(self, components, bit_depth, color_space):
            captured.update(components=components, bit_depth=bit_depth, color_space=color_space)

        def save_to(self, buffer, format, codec):
            buffer.write(b"padded-video")

    monkeypatch.setattr(gemini_nodes, "VideoFromComponents", CapturingVideo)

    video_bytes, mime_type = gemini_nodes.SSL_GeminiTextPrompt._serialize_video(video, fps=2, pad_at_start=True)

    assert video_bytes == b"padded-video"
    assert mime_type == "video/mp4"
    assert captured["components"].frame_rate == 4
    assert captured["components"].images.shape == (5, 3, 5, 3)
    assert torch.count_nonzero(captured["components"].images[0]) == 0
    assert torch.equal(captured["components"].images[1], images[0])
    assert captured["components"].audio["waveform"].shape[-1] == 10000
    assert torch.count_nonzero(captured["components"].audio["waveform"][..., :2000]) == 0


def test_duration_aware_padding_preserves_native_fps_and_rounds_down_to_frames(monkeypatch):
    images = torch.ones((422, 1, 1, 3))
    video = SimpleNamespace(
        get_components=lambda: gemini_nodes.Types.VideoComponents(images=images, frame_rate=40),
        get_duration=lambda: 10.559,
        get_bit_depth=lambda: 8,
        get_color_space=lambda: "sRGB",
    )
    captured = {}

    class CapturingVideo:
        def __init__(self, components, bit_depth, color_space):
            captured["components"] = components

        def save_to(self, buffer, format, codec):
            buffer.write(b"duration-aware-video")

    monkeypatch.setattr(gemini_nodes, "VideoFromComponents", CapturingVideo)

    gemini_nodes.SSL_GeminiTextPrompt._serialize_video(
        video,
        fps=4,
        pad_at_start=True,
        duration_aware_padding=True,
    )

    components = captured["components"]
    assert components.frame_rate == 40
    assert components.images.shape[0] == 22 + images.shape[0]
    assert torch.count_nonzero(components.images[:22]) == 0
    assert torch.equal(components.images[22], images[0])


def test_context_cache_reuses_full_input_after_local_fingerprint_miss(monkeypatch):
    captured = {"creates": [], "generations": []}

    class FakeCaches:
        @staticmethod
        def create(**kwargs):
            captured["creates"].append(kwargs)
            return SimpleNamespace(name="cachedContents/test", expire_time=None)

        @staticmethod
        def update(**kwargs):
            raise AssertionError("TTL should not update")

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["generations"].append(kwargs)
            part = SimpleNamespace(text="ok", inline_data=None)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.caches = FakeCaches()
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._context_cache.clear()
    config = {
        "api_key": "test-key",
        "api_version": "v1",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
        "use_cache": True,
        "cache_ttl_minutes": 15,
        "cache_seed": 123,
    }
    kwargs = _execute_kwargs(config)
    kwargs.update(
        use_seed=True,
        seed=1,
        video=_video_config(b"normalized-video", fps=2),
        image_inputs={"image_1": torch.full((1, 2, 2, 3), 0.25)},
    )

    first = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)
    kwargs["seed"] = 2
    second = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)
    third = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert first[0] == second[0] == third[0] == "ok"
    assert len(captured["creates"]) == 1
    create_config = captured["creates"][0]["config"]
    assert create_config.system_instruction == "test system instruction"
    assert create_config.ttl == "900s"
    assert create_config.contents[0].parts[0].inline_data.mime_type == "image/png"
    assert create_config.contents[0].parts[1].inline_data.data == b"normalized-video"
    assert create_config.contents[0].parts[1].inline_data.mime_type == "video/mp4"
    assert create_config.contents[0].parts[1].video_metadata.fps == 2
    assert create_config.contents[0].parts[-1].text == "test prompt"
    assert len(create_config.contents[0].parts) == 3
    assert len(captured["generations"]) == 2
    assert all(call["contents"] == " " for call in captured["generations"])
    assert all(call["config"].cached_content == "cachedContents/test" for call in captured["generations"])
    assert all(call["config"].system_instruction is None for call in captured["generations"])
    assert all(call["config"].seed == 123 for call in captured["generations"])


def test_context_cache_creation_failure_uses_original_request(monkeypatch):
    captured = {}

    class FakeCaches:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("unsupported cache")

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured.update(kwargs)
            part = SimpleNamespace(text="ok", inline_data=None)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.caches = FakeCaches()
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._context_cache.clear()
    config = {
        "api_key": "test-key",
        "api_version": "v1",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
        "use_cache": True,
        "cache_ttl_minutes": 60,
        "cache_seed": 456,
    }
    kwargs = _execute_kwargs(config)
    kwargs.update(
        model="gemini-3-pro-image",
        include_images=True,
        use_seed=True,
        seed=1,
    )

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert output[0] == "ok"
    assert captured["contents"] == "test prompt"
    assert captured["config"].cached_content is None
    assert captured["config"].system_instruction[0].text == "test system instruction"
    assert captured["config"].seed == 456

    kwargs.update(use_seed=False, seed=2)
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert output[0] == "ok"
    assert captured["config"].seed is None

    kwargs["config"] = config | {"use_cache": False}
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert output[0] == "ok"
    assert captured["config"].seed == 2


def test_missing_context_cache_is_recreated_once(monkeypatch):
    captured = {"creates": 0, "generations": 0}

    class FakeCaches:
        @staticmethod
        def create(**kwargs):
            captured["creates"] += 1
            return SimpleNamespace(
                name=f"cachedContents/test-{captured['creates']}",
                expire_time=None,
            )

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["generations"] += 1
            if captured["generations"] == 1:
                raise RuntimeError("cached content not found")
            part = SimpleNamespace(text="ok", inline_data=None)
            content = SimpleNamespace(parts=[part])
            return SimpleNamespace(candidates=[SimpleNamespace(content=content)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.caches = FakeCaches()
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._context_cache.clear()
    config = {
        "api_key": "test-key",
        "api_version": "v1",
        "use_vertexai_env": False,
        "vertexai_express": False,
        "vertexai_project": "",
        "vertexai_location": "",
        "google_application_credentials": "",
        "use_cache": True,
        "cache_ttl_minutes": 60,
        "cache_seed": 456,
    }
    kwargs = _execute_kwargs(config)
    kwargs.update(use_seed=True, seed=1)

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert output[0] == "ok"
    assert captured == {"creates": 2, "generations": 2}


def test_context_cache_ttl_change_updates_existing_resource():
    captured = {"creates": 0, "updates": []}

    class FakeCaches:
        @staticmethod
        def create(**kwargs):
            captured["creates"] += 1
            return SimpleNamespace(name="cachedContents/test", expire_time=None)

        @staticmethod
        def update(**kwargs):
            captured["updates"].append(kwargs)
            return SimpleNamespace(name="cachedContents/test", expire_time=None)

    client = SimpleNamespace(caches=FakeCaches())
    gemini_nodes.SSL_GeminiTextPrompt._context_cache.clear()

    first = gemini_nodes.SSL_GeminiTextPrompt._get_or_create_context_cache(
        client, ("context",), "gemini-2.0-flash", "prompt", "system", 15
    )
    second = gemini_nodes.SSL_GeminiTextPrompt._get_or_create_context_cache(
        client, ("context",), "gemini-2.0-flash", "prompt", "system", 30
    )

    assert first == second == "cachedContents/test"
    assert captured["creates"] == 1
    assert len(captured["updates"]) == 1
    assert captured["updates"][0]["config"].ttl == "1800s"


def test_context_cache_identity_covers_every_cached_input():
    build_key = gemini_nodes.SSL_GeminiTextPrompt._build_context_cache_key
    base = build_key(("standard", "key-a"), "model-a", "system-a", "prompt-a", "video-a", "video/mp4", 1, ["image-a"], "low")
    variants = {
        build_key(("standard", "key-b"), "model-a", "system-a", "prompt-a", "video-a", "video/mp4", 1, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-b", "system-a", "prompt-a", "video-a", "video/mp4", 1, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-b", "prompt-a", "video-a", "video/mp4", 1, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-a", "prompt-b", "video-a", "video/mp4", 1, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-a", "prompt-a", "video-b", "video/mp4", 1, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-a", "prompt-a", "video-a", "video/webm", 1, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-a", "prompt-a", "video-a", "video/mp4", 2, ["image-a"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-a", "prompt-a", "video-a", "video/mp4", 1, ["image-b"], "low"),
        build_key(("standard", "key-a"), "model-a", "system-a", "prompt-a", "video-a", "video/mp4", 1, ["image-a"], "high"),
    }

    assert base not in variants
    assert len(variants) == 9


def test_video_is_normalized_sent_first_and_locally_fingerprinted(monkeypatch):
    captured = {"generations": []}

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["generations"].append(kwargs)
            part = SimpleNamespace(text="ok", inline_data=None)
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    gemini_nodes.SSL_GeminiTextPrompt._cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    kwargs = _execute_kwargs(_config(api_key="test-key"))
    kwargs.update(
        model="gemini-3-flash-preview",
        media_resolution="low",
        use_seed=True,
        seed=7,
        video=_video_config(b"video-a", fps=2),
    )

    first = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)
    kwargs["video"] = _video_config(b"video-a", fps=2)
    second = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)
    kwargs["video"] = _video_config(b"video-b", fps=2)
    third = gemini_nodes.SSL_GeminiTextPrompt.execute(**kwargs)

    assert first[0] == second[0] == third[0] == "ok"
    assert len(captured["generations"]) == 2
    parts = captured["generations"][0]["contents"][0].parts
    assert parts[0].inline_data.data == b"video-a"
    assert parts[0].inline_data.mime_type == "video/mp4"
    assert parts[0].media_resolution.level == "MEDIA_RESOLUTION_LOW"
    assert parts[-1].text == "test prompt"
    assert kwargs["video"]["video"].save_calls == [
        (gemini_nodes.Types.VideoContainer.MP4, gemini_nodes.Types.VideoCodec.H264)
    ]


def test_video_at_inline_limit_returns_clear_error(monkeypatch):
    monkeypatch.setattr(gemini_nodes.SSL_GeminiTextPrompt, "VIDEO_INLINE_LIMIT_BYTES", 4)
    output = gemini_nodes.SSL_GeminiTextPrompt.execute(
        **_execute_kwargs(_config()),
        video=_video_config(b"1234"),
    )

    assert output[0] == "Error processing input video: Gemini video input must be smaller than 100 MB after MP4/H.264 normalization."


def test_vertex_video_over_cache_blob_limit_uses_full_request(monkeypatch):
    captured = {"cache_creates": 0, "generations": []}

    class FakeCaches:
        @staticmethod
        def create(**kwargs):
            captured["cache_creates"] += 1
            raise AssertionError("Vertex cache creation must be skipped")

    class FakeModels:
        @staticmethod
        def generate_content(**kwargs):
            captured["generations"].append(kwargs)
            part = SimpleNamespace(text="ok", inline_data=None)
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

    class FakeClient:
        def __init__(self, **kwargs):
            self.caches = FakeCaches()
            self.models = FakeModels()

    monkeypatch.setattr(gemini_nodes.genai, "Client", FakeClient)
    monkeypatch.setattr(gemini_nodes.SSL_GeminiTextPrompt, "VERTEX_CACHE_INLINE_LIMIT_BYTES", 1)
    gemini_nodes.SSL_GeminiTextPrompt._cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._client_cache.clear()
    gemini_nodes.SSL_GeminiTextPrompt._context_cache.clear()
    config = _config(api_key="test-key", vertexai_express=True, use_cache=True, cache_seed=11)

    output = gemini_nodes.SSL_GeminiTextPrompt.execute(
        **_execute_kwargs(config),
        use_seed=True,
        seed=2,
        video=_video_config(b"12"),
    )

    assert output[0] == "ok"
    assert captured["cache_creates"] == 0
    assert len(captured["generations"]) == 1
    assert captured["generations"][0]["contents"] != " "
    assert captured["generations"][0]["config"].seed == 11


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
        "use_cache",
        "cache_ttl_minutes",
        "cache_seed",
    ]
    assert all(input_.optional for input_ in schema.inputs[-3:])


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


def test_confirmed_current_models_are_visible_without_deprecated_image_aliases():
    schema = gemini_nodes.SSL_GeminiTextPrompt.define_schema()
    model_input = next(input_ for input_ in schema.inputs if input_.id == "model")
    input_ids = [input_.id for input_ in schema.inputs]
    aspect_ratio = next(input_ for input_ in schema.inputs if input_.id == "aspect_ratio")
    image_size = next(input_ for input_ in schema.inputs if input_.id == "image_size")

    for model in (
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image",
    ):
        assert model in model_input.options

    assert "gemini-2.5-flash-image-preview" not in model_input.options
    assert "nano-banana-pro-preview" not in model_input.options
    assert input_ids[input_ids.index("include_images") + 1] == "aspect_ratio"
    assert aspect_ratio.options == ["None", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"]
    assert aspect_ratio.default == "None"
    assert input_ids[input_ids.index("timeout_fallback_text") + 1] == "image_size"
    assert image_size.options == ["None", "512", "1K", "2K", "4K"]
    assert image_size.default == "None"


def test_image_config_omits_none_aspect_ratio():
    config = gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
        model="gemini-3.1-flash-image",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        seed=7,
        include_images=True,
        response_modalities=["TEXT", "IMAGE"],
        aspect_ratio="None",
        padded_system_instruction="system",
        thinking_level="None",
        thinking_budget=0,
        include_thoughts=False,
        media_resolution="unspecified",
    )

    assert config.response_modalities == ["TEXT", "IMAGE"]
    assert config.image_config is None


def test_image_config_preserves_concrete_aspect_ratio():
    config = gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
        model="gemini-3-pro-image",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        seed=7,
        include_images=True,
        response_modalities=["IMAGE", "TEXT"],
        aspect_ratio="16:9",
        padded_system_instruction="system",
        thinking_level="None",
        thinking_budget=0,
        include_thoughts=False,
        media_resolution="unspecified",
    )

    assert config.image_config.aspect_ratio == "16:9"


def test_image_config_preserves_requested_image_size():
    config = gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
        model="gemini-3.1-flash-image",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        seed=7,
        include_images=True,
        response_modalities=["IMAGE", "TEXT"],
        aspect_ratio="None",
        padded_system_instruction="system",
        thinking_level="None",
        thinking_budget=0,
        include_thoughts=False,
        media_resolution="unspecified",
        image_size="2K",
    )

    assert config.image_config.aspect_ratio is None
    assert config.image_config.image_size == "2K"


@pytest.mark.parametrize(
    ("model", "image_size", "allowed"),
    [
        ("gemini-3.1-flash-lite-image", "2K", "1K"),
        ("gemini-3-pro-image", "512", "1K, 2K, 4K"),
    ],
)
def test_image_config_rejects_unsupported_image_size(model, image_size, allowed):
    with pytest.raises(ValueError, match=rf"^{model} image_size must be one of: {allowed}\.$"):
        gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
            model=model,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            seed=7,
            include_images=True,
            response_modalities=["IMAGE", "TEXT"],
            aspect_ratio="None",
            padded_system_instruction="system",
            thinking_level="None",
            thinking_budget=0,
            include_thoughts=False,
            media_resolution="unspecified",
            image_size=image_size,
        )


def test_enterprise_image_config_allows_all_people():
    config = gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
        model="gemini-3.1-flash-image",
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        seed=7,
        include_images=True,
        response_modalities=["IMAGE", "TEXT"],
        aspect_ratio="None",
        padded_system_instruction="system",
        thinking_level="None",
        thinking_budget=0,
        include_thoughts=False,
        media_resolution="unspecified",
        allow_all_people=True,
    )

    assert config.image_config.aspect_ratio is None
    assert config.image_config.person_generation == "ALLOW_ALL"


@pytest.mark.parametrize(
    ("selected_level", "expected_level"),
    [("None", "medium"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_gemini_3_7_uses_supported_thinking_levels(selected_level, expected_level):
    config = gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
        model="gemini-3.7-flash",
        temperature=0.4,
        top_p=0.8,
        top_k=20,
        max_output_tokens=1024,
        seed=7,
        include_images=False,
        response_modalities=["TEXT"],
        aspect_ratio="None",
        padded_system_instruction="system",
        thinking_level=selected_level,
        thinking_budget=4096,
        include_thoughts=True,
        media_resolution="unspecified",
    )

    assert config.thinking_config.thinking_level.value == expected_level.upper()
    assert config.thinking_config.thinking_budget is None
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None


def test_gemini_3_8_uses_gemini_3_thinking_config():
    config = gemini_nodes.SSL_GeminiTextPrompt._build_generate_content_config(
        model="gemini-3.8-flash",
        temperature=0.4,
        top_p=0.8,
        top_k=20,
        max_output_tokens=1024,
        seed=7,
        include_images=False,
        response_modalities=["TEXT"],
        aspect_ratio="None",
        padded_system_instruction="system",
        thinking_level="None",
        thinking_budget=4096,
        include_thoughts=True,
        media_resolution="unspecified",
    )

    assert config.thinking_config.thinking_level.value == "MEDIUM"
    assert config.thinking_config.thinking_budget is None
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None


def test_gemini_3_8_rejects_minimal_thinking_level():
    with pytest.raises(ValueError, match="gemini-3.8-flash thinking_level"):
        gemini_nodes.SSL_GeminiTextPrompt._resolve_gemini_3_7_thinking_level(
            "minimal", gemini_nodes.SSL_GeminiTextPrompt.GEMINI_3_8_FLASH
        )


def test_gemini_3_7_fingerprint_matches_effective_request_controls():
    common = {
        "config": {"api_key": "key", "api_version": "v1"},
        "prompt": "prompt",
        "system_instruction": "system",
        "model": "gemini-3.7-flash",
        "max_output_tokens": 1024,
        "include_images": False,
        "aspect_ratio": "None",
        "bypass_mode": "None",
        "use_seed": False,
        "seed": 0,
        "include_thoughts": True,
    }

    default_fingerprint, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        temperature=0.4,
        top_p=0.8,
        top_k=20,
        thinking_budget=0,
        thinking_level="None",
    )
    explicit_medium_fingerprint, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        thinking_budget=8192,
        thinking_level="medium",
    )
    high_fingerprint, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        thinking_budget=8192,
        thinking_level="high",
    )
    minimal_fingerprint, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        thinking_budget=8192,
        thinking_level="minimal",
    )
    low_fingerprint, _ = gemini_nodes.SSL_GeminiTextPrompt._compute_fingerprint_and_check_cache(
        **common,
        temperature=0.2,
        top_p=0.7,
        top_k=10,
        thinking_budget=0,
        thinking_level="low",
    )

    assert default_fingerprint == explicit_medium_fingerprint
    assert high_fingerprint != default_fingerprint
    assert minimal_fingerprint == low_fingerprint


def test_gemini_3_7_coerces_minimal_to_low_with_warning(capsys):
    resolved = gemini_nodes.SSL_GeminiTextPrompt._resolve_gemini_3_7_thinking_level(
        "minimal"
    )

    assert resolved == "low"
    assert "does not support minimal thinking; using low instead" in capsys.readouterr().out


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
    assert output[3] == output[4] == ""
    assert gemini_nodes.SSL_GeminiTextPrompt._cache == {}

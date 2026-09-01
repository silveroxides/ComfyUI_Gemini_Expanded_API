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

    assert schema.inputs[-2].id == "video"
    assert schema.inputs[-2].optional is True
    assert schema.inputs[-1].id == "image_inputs"


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
    assert create_config.contents[0].parts[0].inline_data.data == b"normalized-video"
    assert create_config.contents[0].parts[0].inline_data.mime_type == "video/mp4"
    assert create_config.contents[0].parts[0].video_metadata.fps == 2
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

    for model in (
        "gemini-3.7-flash",
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
    assert gemini_nodes.SSL_GeminiTextPrompt._cache == {}

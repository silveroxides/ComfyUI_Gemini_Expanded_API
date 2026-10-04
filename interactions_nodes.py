import os
import json
import uuid
import re
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
import folder_paths  # type: ignore[reportMissingImports]
from comfy_api.latest import ComfyExtension, UI, IO, Types, VideoFromComponents  # type: ignore[reportMissingImports]
from google import genai
from google.genai import errors, types
import time
import traceback
import threading
import queue
import sys
import subprocess
import random
import hashlib
import warnings
from typing import Any, Tuple

from .gemini_nodes import (
    GetKeyAPI,
    SSL_GeminiAPIKeyConfig,
    SSL_GeminiVideoConfig,
)


class SSL_GeminiInteractionsAPIKeyConfig(IO.ComfyNode):
    GemConfig = IO.Custom("GEMINI_CONFIG")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SSL_GeminiInteractionsAPIKeyConfig",
            display_name="Configure Gemini Interactions API Key",
            category="API/Gemini/Interactions",
            inputs=[
                IO.String.Input("api_key", multiline=False, default=""),
                IO.Combo.Input("api_version", options=["v1beta", "v1alpha", "v1beta1"], default="v1beta", tooltip="Select API version to use for Interactions API. Default is v1beta."),
                IO.Boolean.Input("use_vertexai_env", default=False, tooltip="Uses Gemini Enterprise Agent Platform (formerly Vertex AI) with GOOGLE_GENAI_USE_ENTERPRISE or legacy GOOGLE_GENAI_USE_VERTEXAI environment variable."),
                IO.Boolean.Input("vertexai_express", default=False, tooltip="Uses Vertex AI Express mode with an API key."),
                IO.String.Input("vertexai_project", optional=True, tooltip="Google Cloud Project ID."),
                IO.String.Input("vertexai_location", optional=True, tooltip="Google Cloud Location/Region (e.g. us-central1)."),
                IO.String.Input("google_application_credentials", default="", optional=True, multiline=False, tooltip="Optional absolute path to an application default credentials JSON file for Vertex AI."),
            ],
            outputs=[
                cls.GemConfig.Output("config")
            ]
        )

    @classmethod
    def execute(cls, api_key: str, api_version: str, use_vertexai_env: bool, vertexai_express: bool,
                vertexai_project: str | None = "", vertexai_location: str | None = "",
                google_application_credentials: str | None = "") -> IO.NodeOutput:
        config = {
            "api_key": api_key,
            "api_version": api_version,
            "use_vertexai_env": use_vertexai_env,
            "vertexai_express": vertexai_express,
            "vertexai_project": vertexai_project,
            "vertexai_location": vertexai_location,
            "google_application_credentials": google_application_credentials,
        }
        return IO.NodeOutput(config)


class SSL_GeminiInteractionsTextPrompt(IO.ComfyNode):
    GemConfig = IO.Custom("GEMINI_CONFIG")
    ResponseSchema = IO.Custom("GEMINI_RESPONSE_SCHEMA")
    GeminiVideoConfig = IO.Custom("GEMINI_VIDEO_CONFIG")
    _cache: dict = {}
    _seed_map_cache: dict = {}  # Maps (input_seed, fingerprint) -> successful_gemini_seed
    _client_cache: dict = {}  # Maps client_key tuple -> genai.Client instance
    GEMINI_3_7_FLASH = "gemini-3.7-flash"
    GEMINI_3_8_FLASH = "gemini-3.8-flash"
    VIDEO_MIME_TYPE = "video/mp4"
    VIDEO_INLINE_LIMIT_BYTES = 100 * 1024 * 1024

    GEMINI_4_FLASH_PREVIEW = "gemini-4-flash-preview"

    THINKING_MODELS = [
        "gemini-1.5-pro-002", "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25",
        "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", GEMINI_3_7_FLASH, GEMINI_3_8_FLASH, GEMINI_4_FLASH_PREVIEW, "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"
    ]
    GEN3_THINKING_MODELS = [
        "gemini-pro-latest", "gemini-flash-latest", "gemini-3.1-pro-preview",
        "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", GEMINI_3_7_FLASH, GEMINI_3_8_FLASH, GEMINI_4_FLASH_PREVIEW
    ]
    IMAGE_MODELS = [
        "gemini-2.5-flash-image",
        "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image-preview", "gemini-3-pro-image",
    ]
    IMAGE_SIZE_BY_SELECTABLE_MODEL = {
        "gemini-3.1-flash-image": ("512", "1K", "2K", "4K"),
        "gemini-3.1-flash-lite-image": ("1K",),
        "gemini-3-pro-image": ("1K", "2K", "4K"),
    }
    MEDIA_RES_MODELS = [
        "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.1-pro-preview",
        "gemini-3.5-flash", "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"
    ]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        image_names = [f"image_{index}" for index in range(1, 101)]
        image_template = IO.Autogrow.TemplateNames(
            IO.Image.Input("image_1", optional=True),
            names=image_names,
            min=0,
        )
        return IO.Schema(
            node_id="SSL_GeminiInteractionsTextPrompt",
            display_name="Expanded Gemini Interactions (Text/Image)",
            category="API/Gemini/Interactions",
            inputs=[
                cls.GemConfig.Input("config"),
                IO.String.Input("prompt", multiline=True),
                IO.String.Input("system_instruction", default="You are a helpful AI assistant.", multiline=True),
                IO.Combo.Input("model", options=["gemini-1.5-pro-002", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", cls.GEMINI_3_7_FLASH, cls.GEMINI_3_8_FLASH, "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image", "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"], default="gemini-2.5-flash"),
                IO.Float.Input("temperature", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Ignored in Interactions API"),
                IO.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01, tooltip="Ignored in Interactions API"),
                IO.Int.Input("top_k", default=40, min=1, max=100, step=1, tooltip="Ignored in Interactions API"),
                IO.Int.Input("max_output_tokens", default=8192, min=1, max=65536, step=1),
                IO.Boolean.Input("include_images", default=False),
                IO.Combo.Input("aspect_ratio", options=["None", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9", "1:8", "8:1", "1:4", "4:1"], default="None"),
                IO.Combo.Input("bypass_mode", options=["None", "system_instruction", "prompt", "both"], default="None"),
                IO.Int.Input("thinking_budget", default=0, min=-1, max=24576, step=1, tooltip="Ignored in Interactions API. Use thinking_level instead."),
                IO.Boolean.Input("use_proxy", default=False),
                IO.String.Input("proxy_host", default="127.0.0.1"),
                IO.Int.Input("proxy_port", default=7890, min=1, max=65535),
                IO.Boolean.Input("use_seed", default=True),
                IO.Int.Input("seed", default=0, min=0, max=2147483647),
                IO.Int.Input("timeout", default=30, min=15, max=300, step=15),
                IO.Boolean.Input("include_thoughts", default=False),
                IO.Combo.Input("thinking_level", options=["None", "minimal", "low", "medium", "high"], default="None", tooltip="Thinking level for reasoning models."),
                IO.Combo.Input("media_resolution", options=["unspecified", "low", "medium", "high", "ultra_high"], default="unspecified", tooltip="Set input media resolution for image, video and pdf."),
                IO.String.Input("retry_pattern", default="", optional=True, multiline=False, tooltip="Regex pattern to match in response text. If matched, retry with new seed. Leave empty to disable."),
                IO.Int.Input("max_retries", default=3, min=0, max=10, step=1, tooltip="Maximum number of retry attempts when pattern matches. 0 disables retry."),
                IO.String.Input("timeout_fallback_text", default="", optional=True, multiline=True, tooltip="Text returned when the Gemini request times out. Leave empty to return the standard timeout message."),
                IO.Combo.Input("image_size", options=["None", "512", "1K", "2K", "4K"], default="None", optional=True, tooltip="Generated image resolution. Gemini 3.1 Flash Lite Image supports only 1K; 512 is supported only by Gemini 3.1 Flash Image."),
                IO.Boolean.Input("store", default=True, optional=True, tooltip="Store interaction on server for multi-turn conversations."),
                IO.String.Input("previous_interaction_id", default="", optional=True, multiline=False, tooltip="Optional ID of previous interaction to continue multi-turn conversation."),
                cls.GeminiVideoConfig.Input("video", optional=True, tooltip="Optional configured Gemini video input with embedded audio and sampling FPS."),
                cls.ResponseSchema.Input("response_schema", display_name="Response format", optional=True,
                                         tooltip="Connect an answer format to request the fields you defined. Leave disconnected for a normal answer."),
                IO.Autogrow.Input(
                    "image_inputs",
                    template=image_template,
                    optional=True,
                    tooltip=(
                        "Ordered Gemini image parts growing from image_1 through image_100. Images inside a batch "
                        "are sent consecutively before the next socket. Provider request-size and model-specific "
                        "reference limits still apply."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output("text"),
                IO.Image.Output("image"),
                IO.Int.Output("final_actual_seed"),
                IO.String.Output("structured_output", display_name="Structured output"),
                IO.String.Output("thoughts", display_name="Thoughts"),
                IO.String.Output("interaction_id", display_name="Interaction ID"),
            ]
        )

    @classmethod
    def _pad_text_with_joiners(cls, text: str) -> str:
        if not text:
            return ""

        patternperiod = r"\."
        patternspace = r"\s"
        patterncomma = r","
        patterndash = r"\-"
        patternsingq = r"\'"
        patterndoubq = r'\"'
        patternword = r"(.)(?=.)"

        replperiod = r"。"
        replspace = r"﻿"
        replcomma = r"、"
        repldash = r"‐"
        replsingq = r"ʼ"
        repldoubq = r"ˮ"
        replword = r"⁠\1⁠﻿"

        joined_textperiod = re.sub(patternperiod, replperiod, text)
        joined_textspace = re.sub(patternspace, replspace, joined_textperiod)
        joined_textcomma = re.sub(patterncomma, replcomma, joined_textspace)
        joined_textdash = re.sub(patterndash, repldash, joined_textcomma)
        joined_textsingq = re.sub(patternsingq, replsingq, joined_textdash)
        joined_textdoubq = re.sub(patterndoubq, repldoubq, joined_textsingq)
        joined_textfinal = re.sub(patternword, replword, joined_textdoubq)

        return joined_textfinal

    @classmethod
    def save_binary_file(cls, data: bytes, mime_type: str) -> str:
        ext = ".bin"
        if mime_type == "image/png":
            ext = ".png"
        elif mime_type == "image/jpeg":
            ext = ".jpg"

        output_dir = folder_paths.get_output_directory()
        gemini_dir = os.path.join(output_dir, "gemini_outputs")
        os.makedirs(gemini_dir, exist_ok=True)

        file_name = os.path.join(gemini_dir, f"gemini_output_{uuid.uuid4()}{ext}")

        with open(file_name, "wb") as f:
            f.write(data)

        return file_name

    @classmethod
    def generate_empty_image(cls, width=64, height=64) -> torch.Tensor:
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0)
        return tensor

    @classmethod
    def _ordered_image_frames(cls, image_inputs: IO.Autogrow.Type | None):
        if not image_inputs:
            return [], []

        def image_index(name):
            match = re.fullmatch(r"image_(\d+)", name)
            return int(match.group(1)) if match else 101

        frames = []
        batch_counts = []
        for socket_name, image_batch in sorted(
            image_inputs.items(), key=lambda item: (image_index(item[0]), item[0])
        ):
            if image_batch is None:
                continue
            if not torch.is_tensor(image_batch) or image_batch.ndim != 4:
                raise ValueError(
                    f"Gemini {socket_name} must be a BHWC image tensor."
                )
            batch_size = int(image_batch.shape[0])
            batch_counts.append((socket_name, batch_size))
            frames.extend(
                (socket_name, index, image_batch[index])
                for index in range(batch_size)
            )
        return frames, batch_counts

    @classmethod
    def _serialize_video(cls, video: IO.Video.Type | None, fps: int | None = None,
                         pad_at_start: bool = False, duration_aware_padding: bool = False):
        if video is None:
            return None, None

        video_to_save = video
        if pad_at_start:
            if fps is None:
                raise ValueError("Gemini video FPS is required when padding at start.")
            components = video.get_components()
            native_frame_rate = components.frame_rate
            source_frame_rate = float(native_frame_rate)
            source_images = components.images
            source_frame_count = source_images.shape[0]
            if duration_aware_padding:
                source_duration = video.get_duration()
                target_frame_rate = native_frame_rate
                padding_frame_count = int((source_duration % 1.0) * source_frame_rate)
                resampled_images = source_images
            else:
                source_duration = source_frame_count / source_frame_rate
                target_frame_rate = fps * 2
                padding_frame_count = 1
                target_frame_count = max(1, round(source_duration * target_frame_rate))
                source_indices = (
                    torch.arange(target_frame_count, device=source_images.device, dtype=torch.float64)
                    * source_frame_rate
                    / target_frame_rate
                ).floor().long().clamp(max=source_frame_count - 1)
                resampled_images = source_images.index_select(0, source_indices)
            black_frames = torch.zeros_like(resampled_images[:1]).expand(padding_frame_count, -1, -1, -1)
            padded_images = torch.cat((black_frames, resampled_images), dim=0)

            padded_audio = components.audio
            if padded_audio:
                sample_rate = int(padded_audio["sample_rate"])
                waveform = padded_audio["waveform"]
                silence_samples = round(sample_rate * padding_frame_count / target_frame_rate)
                silence = torch.zeros(
                    (*waveform.shape[:-1], silence_samples),
                    dtype=waveform.dtype,
                    device=waveform.device,
                )
                padded_audio = {
                    **padded_audio,
                    "waveform": torch.cat((silence, waveform), dim=-1),
                }

            video_to_save = VideoFromComponents(
                Types.VideoComponents(
                    images=padded_images,
                    audio=padded_audio,
                    frame_rate=target_frame_rate,
                ),
                bit_depth=video.get_bit_depth(),
                color_space=video.get_color_space(),
            )

        buffer = BytesIO()
        video_to_save.save_to(
            buffer,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
        )
        video_bytes = buffer.getvalue()
        if len(video_bytes) >= cls.VIDEO_INLINE_LIMIT_BYTES:
            raise ValueError("Gemini video input must be smaller than 100 MB after MP4/H.264 normalization.")
        return video_bytes, cls.VIDEO_MIME_TYPE

    @classmethod
    def _resolve_gemini_3_7_thinking_level(cls, thinking_level, model_name=None):
        if thinking_level is None or thinking_level == "None":
            return "medium"
        if thinking_level == "minimal":
            if model_name == cls.GEMINI_3_8_FLASH:
                raise ValueError(
                    "gemini-3.8-flash thinking_level must be low, medium, or high."
                )
            print(
                f"[WARNING] {model_name or 'Gemini 3.7 Flash'} does not support minimal "
                "thinking; using low instead."
            )
            return "low"
        if thinking_level not in {"low", "medium", "high"}:
            raise ValueError(
                f"{model_name or 'Gemini 3.7 Flash'} thinking_level must be low, medium, or high."
            )
        return thinking_level

    @classmethod
    def _compute_fingerprint_and_check_cache(cls, config, prompt, system_instruction, model, max_output_tokens,
                                             include_images, aspect_ratio, bypass_mode, use_seed, seed,
                                             video_hash=None, video_mime_type=None, video_fps=None,
                                             image_inputs: IO.Autogrow.Type | None = None,
                                             use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, timeout=30,
                                             include_thoughts=False, thinking_level=None, media_resolution=None,
                                             retry_pattern="", max_retries=3, image_size="None",
                                             store=True, previous_interaction_id="", schema_identity=None):

        def get_tensor_hash(tensor):
            if tensor is None:
                return "None"
            try:
                return hashlib.sha256(np.ascontiguousarray(tensor.cpu().numpy()).tobytes()).hexdigest()
            except Exception as e:
                print(f"[WARNING] Cache hashing failed for image: {e}")
                return "Error"

        image_frames, _ = cls._ordered_image_frames(image_inputs)
        image_hashes = tuple(get_tensor_hash(frame) for _, _, frame in image_frames)

        eff_include_images = False
        eff_aspect_ratio = "None"
        eff_image_size = "None"
        eff_thinking_level = "None"
        eff_include_thoughts = False

        if include_images and model in cls.IMAGE_MODELS:
            eff_include_images = True
            eff_aspect_ratio = str(aspect_ratio)
            eff_image_size = "None" if image_size in (None, "None") else str(image_size)

        elif model in (cls.GEMINI_3_7_FLASH, cls.GEMINI_3_8_FLASH):
            eff_thinking_level = cls._resolve_gemini_3_7_thinking_level(thinking_level, model)
            eff_include_thoughts = include_thoughts

        elif model in cls.GEN3_THINKING_MODELS and thinking_level not in (None, "None"):
            eff_thinking_level = thinking_level
            eff_include_thoughts = include_thoughts

        elif thinking_level not in (None, "None"):
            eff_thinking_level = thinking_level
            eff_include_thoughts = include_thoughts

        sanitized_config = dict(config)
        api_key = sanitized_config.get("api_key")
        if api_key:
            sanitized_config["api_key"] = f"sha256:{hashlib.sha256(str(api_key).encode('utf-8')).hexdigest()}"

        fingerprint = (
            str(sanitized_config),
            prompt,
            system_instruction,
            model,
            int(max_output_tokens),
            eff_include_images,
            eff_aspect_ratio,
            eff_image_size,
            str(bypass_mode),
            use_seed,
            int(seed) if use_seed else 0,
            video_hash,
            video_mime_type,
            video_fps,
            image_hashes,
            bool(use_proxy),
            str(proxy_host),
            int(proxy_port),
            int(timeout),
            eff_include_thoughts,
            eff_thinking_level,
            str(media_resolution),
            str(retry_pattern),
            int(max_retries),
            bool(store),
            str(previous_interaction_id),
            schema_identity,
        )

        cached = cls._cache.get(fingerprint)
        if use_seed and cached is not None:
            return fingerprint, cached
        return fingerprint, None

    @classmethod
    def _handle_seed(cls, use_seed, seed):
        if not use_seed:
            print("[INFO] Seed not used")
            return None
        if 0 < seed < 2**31:
            print(f"[INFO] Using specified seed: {seed}")
            return seed

        generator = torch.Generator(device="cpu")
        if seed == 0:
            generator.seed()
        else:
            generator.manual_seed(seed)
        actual_seed = torch.randint(
            0, 2147483647, (), generator=generator, device="cpu", dtype=torch.int64
        ).item()
        print(f"[INFO] Generated API seed: {actual_seed}")
        return actual_seed

    @classmethod
    def _build_proxy_url(cls, proxy_host, proxy_port):
        if not proxy_host.startswith(('http://', 'https://')):
            proxy_url = f"http://{proxy_host}:{proxy_port}"
        else:
            proxy_url = f"{proxy_host}:{proxy_port}"

        print(f"[INFO] Proxy enabled: {proxy_url}")
        return proxy_url

    @staticmethod
    def _parse_boolean_environment_variable(name: str) -> bool | None:
        value = os.environ.get(name)
        if value is None:
            return None

        normalized = value.strip().lower()
        if normalized not in {"true", "1", "false", "0"}:
            raise ValueError(f"{name} must be true, false, 1, or 0")
        return normalized in {"true", "1"}

    @classmethod
    def _resolve_enterprise_environment(cls) -> bool:
        enterprise = cls._parse_boolean_environment_variable("GOOGLE_GENAI_USE_ENTERPRISE")
        legacy_vertex = cls._parse_boolean_environment_variable("GOOGLE_GENAI_USE_VERTEXAI")

        if enterprise is not None and legacy_vertex is not None and enterprise != legacy_vertex:
            warnings.warn(
                "GOOGLE_GENAI_USE_ENTERPRISE and GOOGLE_GENAI_USE_VERTEXAI conflict; "
                "GOOGLE_GENAI_USE_ENTERPRISE takes precedence.",
                UserWarning,
                stacklevel=2,
            )

        if enterprise is not None:
            return enterprise
        if legacy_vertex is not None:
            return legacy_vertex
        return True

    @staticmethod
    def _build_http_options(api_version: str, proxy_url: str | None):
        client_args = {"proxy": proxy_url} if proxy_url else None
        return types.HttpOptions(api_version=api_version, client_args=client_args)

    @staticmethod
    def _reject_json_constant(value):
        raise ValueError(f"Structured response contains invalid JSON constant: {value}.")

    @classmethod
    def execute(cls, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
                include_images, aspect_ratio, bypass_mode, thinking_budget,
                use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, use_seed=False, seed=0, timeout=30,
                include_thoughts=False, thinking_level=None, media_resolution=None,
                retry_pattern="", max_retries=3, timeout_fallback_text="",
                image_size="None", store=True, previous_interaction_id="",
                video: GeminiVideoConfig.Type | None = None,
                image_inputs: IO.Autogrow.Type | None = None,
                response_schema: ResponseSchema.Type | None = None) -> IO.NodeOutput:

        print(f"[INFO] SSL_GeminiInteractionsTextPrompt execute called, model: {model}")
        schema_identity = None
        schema_snapshot = None
        if response_schema is not None:
            if not isinstance(response_schema, dict):
                raise ValueError("Response format must be a schema dictionary from the schema output.")
            if include_images and model in cls.IMAGE_MODELS:
                raise ValueError("Response format cannot be combined with image generation. Disable include_images.")
            schema_identity = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            schema_snapshot = json.loads(schema_identity)

        video_input = video.get("video") if video is not None else None
        video_fps = int(video.get("fps", 1)) if video is not None else None
        video_pad_at_start = bool(video.get("pad_at_start", False)) if video is not None else False
        video_duration_aware_padding = bool(video.get("duration_aware_padding", False)) if video is not None else False
        try:
            video_bytes, video_mime_type = cls._serialize_video(
                video_input,
                video_fps,
                video_pad_at_start,
                video_duration_aware_padding,
            )
        except Exception as e:
            print(f"[ERROR] Error processing input video: {e}")
            output_seed = seed if use_seed else 0
            return IO.NodeOutput(f"Error processing input video: {e}", cls.generate_empty_image(), output_seed, "", "", "")

        video_hash = hashlib.sha256(video_bytes).hexdigest() if video_bytes is not None else None
        fingerprint, cached = cls._compute_fingerprint_and_check_cache(
            config, prompt, system_instruction, model, max_output_tokens,
            include_images, aspect_ratio, bypass_mode, use_seed, seed,
            video_hash, video_mime_type, video_fps,
            image_inputs,
            use_proxy, proxy_host, proxy_port, timeout,
            include_thoughts, thinking_level, media_resolution,
            retry_pattern, max_retries, image_size=image_size,
            store=store, previous_interaction_id=previous_interaction_id,
            schema_identity=schema_identity,
        )

        if cached is not None:
            print(f"[INFO] Returning cached result for fingerprint {fingerprint}")
            return IO.NodeOutput(*cached)

        print(f"[INFO] Starting interaction, model: {model}")

        padded_prompt = prompt
        padded_system_instruction = system_instruction

        if bypass_mode == "prompt" or bypass_mode == "both":
            padded_prompt = cls._pad_text_with_joiners(prompt)
        if bypass_mode == "system_instruction" or bypass_mode == "both":
            padded_system_instruction = cls._pad_text_with_joiners(system_instruction)

        actual_seed = cls._handle_seed(use_seed, seed)
        input_seed = seed

        seed_cache_key = (input_seed, fingerprint)
        if use_seed and retry_pattern and max_retries > 0:
            cached_gemini_seed = cls._seed_map_cache.get(seed_cache_key)
            if cached_gemini_seed is not None:
                print(f"[INFO] Using cached successful gemini seed {cached_gemini_seed} for input seed {input_seed}")
                actual_seed = cls._handle_seed(True, cached_gemini_seed)

        text_output = ""
        thoughts_output = ""
        structured_output = ""
        interaction_id_output = ""
        is_success = False
        image_tensor = cls.generate_empty_image()
        proxy_url: str | None = None

        try:
            if use_proxy:
                proxy_url = cls._build_proxy_url(proxy_host, proxy_port)

            try:
                vertexai_express = config.get("vertexai_express", False)
                use_vertexai_env = config.get("use_vertexai_env", False)
                api_version = config.get("api_version") or "v1beta"
                project = config.get("vertexai_project")
                location = config.get("vertexai_location")
                credentials_path = config.get("google_application_credentials")
                http_options = cls._build_http_options(api_version, proxy_url)

                if use_vertexai_env:
                    try:
                        env_use = cls._resolve_enterprise_environment()
                        if not env_use:
                            raise ValueError(
                                "GOOGLE_GENAI_USE_ENTERPRISE or GOOGLE_GENAI_USE_VERTEXAI "
                                "must be enabled when use_vertexai_env is true"
                            )

                        credentials = None
                        credentials_cache_key = None
                        credential_project = None
                        if credentials_path:
                            import google.auth

                            absolute_credentials_path = os.path.abspath(os.path.expanduser(credentials_path))
                            if not os.path.isfile(absolute_credentials_path):
                                raise ValueError(f"Application default credentials file not found: {absolute_credentials_path}")
                            credentials, credential_project = google.auth.load_credentials_from_file(
                                absolute_credentials_path,
                                scopes=["https://www.googleapis.com/auth/cloud-platform"],
                            )
                            credentials_cache_key = absolute_credentials_path

                        env_proj = os.environ["GOOGLE_CLOUD_PROJECT"].strip() if "GOOGLE_CLOUD_PROJECT" in os.environ else (project or credential_project)
                        assert env_proj, "GOOGLE_CLOUD_PROJECT is empty"

                        env_loc = os.environ["GOOGLE_CLOUD_LOCATION"].strip() if "GOOGLE_CLOUD_LOCATION" in os.environ else location
                        assert env_loc, "GOOGLE_CLOUD_LOCATION is empty"

                        client_key = ("vertexai_env", env_use, env_proj, env_loc, api_version, proxy_url, credentials_cache_key)
                        if client_key not in cls._client_cache:
                            cls._client_cache[client_key] = genai.Client(
                                enterprise=env_use,
                                credentials=credentials,
                                project=env_proj,
                                location=env_loc,
                                http_options=http_options,
                            )
                            print(f"[INFO] Created new genai.Client (vertexai_env)")
                            is_new_client = True
                        else:
                            print(f"[INFO] Reusing cached genai.Client (vertexai_env)")
                            is_new_client = False
                        client = cls._client_cache[client_key]

                    except KeyError as e:
                        print(f"Missing required environment variable: {e}")
                        return IO.NodeOutput(f"Missing environment variable: {e}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0, "", "", "")

                    except AssertionError as e:
                        print(f"Error: {e}")
                        return IO.NodeOutput(f"Invalid environment variable: {e}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0, "", "", "")

                    except ValueError as e:
                        print(f"Error: {e}")
                        return IO.NodeOutput(f"Invalid Enterprise/Vertex AI configuration: {e}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0, "", "", "")

                elif vertexai_express:
                    api_key = config.get("api_key")
                    if not api_key:
                        return IO.NodeOutput(
                            "Invalid Enterprise express configuration: API key is required",
                            cls.generate_empty_image(),
                            actual_seed if actual_seed is not None else 0,
                            "", "", "",
                        )
                    api_key_hash = hashlib.sha256(str(api_key).encode("utf-8")).hexdigest() if api_key else None
                    client_key = ("vertexai_express", api_key_hash, api_version, proxy_url)
                    if client_key not in cls._client_cache:
                        cls._client_cache[client_key] = genai.Client(
                            enterprise=True,
                            api_key=api_key,
                            http_options=http_options,
                        )
                        print(f"[INFO] Created new genai.Client (vertexai_express)")
                        is_new_client = True
                    else:
                        print(f"[INFO] Reusing cached genai.Client (vertexai_express)")
                        is_new_client = False
                    client = cls._client_cache[client_key]

                else:
                    api_key = config.get("api_key")
                    api_key_hash = hashlib.sha256(str(api_key).encode("utf-8")).hexdigest() if api_key else None
                    client_key = ("standard", api_key_hash, api_version, proxy_url)
                    if client_key not in cls._client_cache:
                        cls._client_cache[client_key] = genai.Client(
                            api_key=api_key,
                            http_options=http_options,
                        )
                        print(f"[INFO] Created new genai.Client (standard)")
                        is_new_client = True
                    else:
                        print(f"[INFO] Reusing cached genai.Client (standard)")
                        is_new_client = False
                    client = cls._client_cache[client_key]

                if is_new_client:
                    try:
                        if hasattr(client, '_api_client') and hasattr(client._api_client, '_access_token'):
                            print("[INFO] Pre-fetching auth token to avoid timeout interference...")
                            client._api_client._access_token()
                    except Exception as auth_e:
                        print(f"[WARNING] Pre-auth check failed (will attempt during generation): {auth_e}")

            except Exception as e:
                print(f"[ERROR] Gemini client initialization failed: {str(e)}")
                return IO.NodeOutput(f"Gemini client initialization failed: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0, "", "", "")

            # Prepare multimodal input items
            image_frames, image_batch_counts = cls._ordered_image_frames(image_inputs)
            media_items = []

            if video_bytes is not None:
                video_b64 = base64.b64encode(video_bytes).decode("utf-8")
                v_item = {
                    "type": "video",
                    "data": video_b64,
                    "mime_type": video_mime_type,
                }
                if media_resolution and media_resolution != "unspecified":
                    v_item["resolution"] = media_resolution.lower()
                media_items.append(v_item)

            if image_frames:
                try:
                    for _, _, image_frame in image_frames:
                        img_array = image_frame.cpu().numpy()
                        img_array = (img_array * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array)
                        img_byte_arr = BytesIO()
                        pil_img.save(img_byte_arr, format='PNG')
                        img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                        img_item = {
                            "type": "image",
                            "data": img_b64,
                            "mime_type": "image/png",
                        }
                        if media_resolution and media_resolution != "unspecified":
                            img_item["resolution"] = media_resolution.lower()
                        media_items.append(img_item)
                    batch_summary = ", ".join(
                        f"{socket_name}={batch_size}"
                        for socket_name, batch_size in image_batch_counts
                    )
                    print(
                        f"[INFO] Prepared Gemini image inputs: {batch_summary}, "
                        f"total={len(media_items) - (1 if video_bytes is not None else 0)}"
                    )
                except Exception as e:
                    print(f"[ERROR] Error processing input image: {str(e)}")
                    return IO.NodeOutput(f"Error processing input image: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0, "", "", "")

            text_item = {"type": "text", "text": padded_prompt}
            if media_items:
                input_contents = [*media_items, text_item]
            else:
                input_contents = padded_prompt

            # Assemble response_format
            response_formats = []
            if include_images and model in cls.IMAGE_MODELS:
                img_fmt: dict[str, Any] = {"type": "image"}
                if aspect_ratio not in (None, "None"):
                    img_fmt["aspect_ratio"] = aspect_ratio
                if image_size not in (None, "None"):
                    supported_image_sizes = cls.IMAGE_SIZE_BY_SELECTABLE_MODEL.get(model)
                    if supported_image_sizes is not None and image_size not in supported_image_sizes:
                        allowed = ", ".join(supported_image_sizes)
                        raise ValueError(f"{model} image_size must be one of: {allowed}.")
                    img_fmt["image_size"] = image_size
                response_formats.append(img_fmt)

            if schema_snapshot is not None:
                response_formats.append({
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema_snapshot,
                })

            # Assemble generation_config
            effective_thinking_level = None
            if model in (cls.GEMINI_3_7_FLASH, cls.GEMINI_3_8_FLASH):
                effective_thinking_level = cls._resolve_gemini_3_7_thinking_level(thinking_level, model)
            elif model in cls.GEN3_THINKING_MODELS and thinking_level not in (None, "None"):
                effective_thinking_level = thinking_level
            elif thinking_level not in (None, "None"):
                effective_thinking_level = thinking_level

            generation_config: dict[str, Any] = {}
            if max_output_tokens:
                generation_config["max_output_tokens"] = int(max_output_tokens)
            if use_seed and actual_seed is not None:
                generation_config["seed"] = int(actual_seed)
            if effective_thinking_level not in (None, "None"):
                generation_config["thinking_level"] = effective_thinking_level
            if include_thoughts:
                generation_config["thinking_summaries"] = "auto"

            # Safety settings
            safety_settings = [
                {"type": "harassment", "threshold": "block_none"},
                {"type": "hate_speech", "threshold": "block_none"},
                {"type": "sexually_explicit", "threshold": "block_none"},
                {"type": "dangerous_content", "threshold": "block_none"},
                {"type": "civic_integrity", "threshold": "block_none"},
            ]

            # Background worker queue for API call
            start_time = time.time()
            result_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

            def create_interaction_once(current_seed):
                call_gen_config = dict(generation_config)
                if use_seed and current_seed is not None:
                    call_gen_config["seed"] = int(current_seed)

                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": input_contents,
                    "safety_settings": safety_settings,
                }
                if padded_system_instruction:
                    kwargs["system_instruction"] = padded_system_instruction
                if call_gen_config:
                    kwargs["generation_config"] = call_gen_config
                if response_formats:
                    kwargs["response_format"] = response_formats
                if store is not None:
                    kwargs["store"] = bool(store)
                if previous_interaction_id and str(previous_interaction_id).strip():
                    kwargs["previous_interaction_id"] = str(previous_interaction_id).strip()

                return client.interactions.create(**kwargs)

            def api_call():
                last_api_exception = None
                api_response = None
                max_attempts = 1
                for attempt in range(max_attempts):
                    try:
                        current_seed = (actual_seed + attempt) if (use_seed and actual_seed is not None) else None
                        response = create_interaction_once(current_seed)
                        api_response = response
                        break
                    except Exception as e:
                        last_api_exception = e
                if api_response is None:
                    result_queue.put(("error", last_api_exception))
                    return

                try:
                    current_text_output = getattr(api_response, "output_text", "") or ""
                    current_thoughts_output = ""
                    current_image_tensor = None
                    current_interaction_id = getattr(api_response, "id", "") or ""

                    # Extract output text / thoughts / image from steps if needed
                    steps = getattr(api_response, "steps", None) or []
                    model_output_texts = []
                    for step in steps:
                        stype = getattr(step, "type", None)
                        if stype == "thought":
                            summary = getattr(step, "summary", None) or []
                            for item in summary:
                                itext = getattr(item, "text", "")
                                if itext:
                                    current_thoughts_output += itext
                            stext = getattr(step, "text", "")
                            if stext:
                                current_thoughts_output += stext
                        elif stype == "model_output":
                            content = getattr(step, "content", None) or []
                            for item in content:
                                itype = getattr(item, "type", None)
                                if itype == "text":
                                    t = getattr(item, "text", "")
                                    if t:
                                        model_output_texts.append(t)
                                elif itype == "image" and current_image_tensor is None:
                                    try:
                                        idata = getattr(item, "data", None)
                                        imime = getattr(item, "mime_type", "image/png")
                                        if idata:
                                            raw_bytes = base64.b64decode(idata) if isinstance(idata, str) else bytes(idata)
                                            image_path = cls.save_binary_file(raw_bytes, imime)
                                            img = Image.open(image_path)
                                            if img.mode != 'RGB':
                                                img = img.convert('RGB')
                                            img_array = np.array(img).astype(np.float32) / 255.0
                                            current_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                                    except Exception as img_err:
                                        print(f"[WARNING] Failed to parse step image: {img_err}")

                    if not current_text_output and model_output_texts:
                        current_text_output = "".join(model_output_texts)

                    # Also check output_image convenience property
                    if current_image_tensor is None and hasattr(api_response, "output_image"):
                        out_img = getattr(api_response, "output_image", None)
                        if out_img is not None:
                            try:
                                idata = getattr(out_img, "data", None)
                                imime = getattr(out_img, "mime_type", "image/png")
                                if idata:
                                    raw_bytes = base64.b64decode(idata) if isinstance(idata, str) else bytes(idata)
                                    image_path = cls.save_binary_file(raw_bytes, imime)
                                    img = Image.open(image_path)
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    img_array = np.array(img).astype(np.float32) / 255.0
                                    current_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            except Exception as img_err:
                                print(f"[WARNING] Failed to parse output_image: {img_err}")

                    if current_image_tensor is None:
                        current_image_tensor = cls.generate_empty_image()

                    if schema_snapshot is not None:
                        json.loads(current_text_output, parse_constant=cls._reject_json_constant)

                    result_queue.put(("success", (current_text_output, current_image_tensor, current_thoughts_output, current_interaction_id)))
                except Exception as e_proc:
                    result_queue.put(("error", e_proc))

            api_thread = threading.Thread(target=api_call)
            api_thread.daemon = True
            api_thread.start()

            retry_attempt = 0
            retry_needed = True
            while retry_needed:
                retry_needed = False
                is_success = False
                structured_output = ""
                thoughts_output = ""
                interaction_id_output = ""

                try:
                    status, result = result_queue.get(timeout=timeout)
                    if status == "success":
                        text_output, image_tensor, thoughts_output, interaction_id_output = result
                        structured_output = text_output if schema_snapshot is not None else ""
                        is_success = True

                        if retry_pattern and max_retries > 0 and retry_attempt < max_retries:
                            try:
                                compiled_pattern = re.compile(retry_pattern, re.IGNORECASE)
                                if compiled_pattern.search(text_output):
                                    retry_attempt += 1
                                    print(f"[INFO] Retry pattern matched in response. Retry attempt {retry_attempt}/{max_retries}")
                                    actual_seed = cls._handle_seed(True, 0)
                                    print(f"[INFO] Retrying with new gemini seed: {actual_seed}")

                                    result_queue = queue.Queue()
                                    start_time = time.time()
                                    api_thread = threading.Thread(target=api_call)
                                    api_thread.daemon = True
                                    api_thread.start()
                                    retry_needed = True
                                    continue
                            except re.error as regex_err:
                                print(f"[WARNING] Invalid retry regex pattern: {regex_err}")
                    else:
                        error_exception = result
                        text_output = f"API call/processing error: {str(error_exception)}"

                        if retry_pattern and max_retries > 0 and retry_attempt < max_retries:
                            try:
                                compiled_pattern = re.compile(retry_pattern, re.IGNORECASE)
                                if compiled_pattern.search(text_output):
                                    retry_attempt += 1
                                    print(f"[INFO] Retry pattern matched in error. Retry attempt {retry_attempt}/{max_retries}")
                                    actual_seed = cls._handle_seed(True, 0)
                                    print(f"[INFO] Retrying with new gemini seed: {actual_seed}")

                                    result_queue = queue.Queue()
                                    start_time = time.time()
                                    api_thread = threading.Thread(target=api_call)
                                    api_thread.daemon = True
                                    api_thread.start()
                                    retry_needed = True
                                    continue
                            except re.error as regex_err:
                                print(f"[WARNING] Invalid retry regex pattern: {regex_err}")

                except queue.Empty:
                    text_output = timeout_fallback_text or f"Gemini API request/processing timed out, waited {timeout} seconds."

        except Exception as e:
            is_success = False
            print(f"[ERROR] Unhandled error in generate method: {str(e)}")
            text_output = f"Unhandled error: {str(e)}"
            if image_tensor is None:
                image_tensor = cls.generate_empty_image()

        final_actual_seed = actual_seed if actual_seed is not None else 0

        if not is_success:
            structured_output = ""
            thoughts_output = ""
            interaction_id_output = ""

        # Cache the result if seed used and successful
        if use_seed and is_success:
            try:
                cls._cache[fingerprint] = (text_output, image_tensor, final_actual_seed, structured_output, thoughts_output, interaction_id_output)
            except Exception:
                pass

            if retry_pattern and max_retries > 0:
                cls._seed_map_cache[seed_cache_key] = final_actual_seed
                print(f"[INFO] Cached successful gemini seed {final_actual_seed} for input seed {input_seed}")

        return IO.NodeOutput(text_output, image_tensor, final_actual_seed, structured_output, thoughts_output, interaction_id_output)

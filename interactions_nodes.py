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
import comfy.model_management  # type: ignore[reportMissingImports]
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
from datetime import datetime

from .gemini_nodes import (
    GetKeyAPI,
    SSL_GeminiAPIKeyConfig,
    SSL_GeminiVideoConfig,
    SSL_GeminiTextPrompt,
)


class GeminiInteractionsStoreManager:
    _lock = threading.Lock()

    @classmethod
    def get_path(cls) -> str:
        try:
            user_dir = folder_paths.get_user_directory()
            if user_dir and os.path.isdir(user_dir):
                return os.path.join(user_dir, "gemini_interactions.json")
        except Exception:
            pass
        output_dir = folder_paths.get_output_directory()
        return os.path.join(output_dir, "gemini_interactions.json")

    @classmethod
    def load(cls) -> dict[str, Any]:
        path = cls.get_path()
        if not os.path.exists(path):
            return {"last_interaction_id": "", "sessions": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[WARNING] Could not read Gemini interactions store: {e}")
        return {"last_interaction_id": "", "sessions": []}

    @classmethod
    def save(cls, data: dict[str, Any]) -> None:
        path = cls.get_path()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Could not save Gemini interactions store: {e}")

    @classmethod
    def get_last_interaction_id(cls) -> str:
        with cls._lock:
            data = cls.load()
            return str(data.get("last_interaction_id") or "").strip()

    @classmethod
    def record_interaction(cls, interaction_id: str, model: str) -> None:
        if not interaction_id or not str(interaction_id).strip():
            return
        interaction_id = str(interaction_id).strip()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with cls._lock:
            data = cls.load()
            sessions = data.get("sessions")
            if not isinstance(sessions, list):
                sessions = []

            existing = None
            for s in sessions:
                if isinstance(s, dict) and s.get("id") == interaction_id:
                    existing = s
                    break

            if existing is not None:
                existing["updated_at"] = now_str
                existing["turn_count"] = int(existing.get("turn_count", 1)) + 1
                if model:
                    existing["model"] = model
            else:
                new_session = {
                    "id": interaction_id,
                    "created_at": now_str,
                    "updated_at": now_str,
                    "model": model or "unknown",
                    "turn_count": 1,
                }
                sessions.append(new_session)

            data["sessions"] = sessions
            data["last_interaction_id"] = interaction_id
            cls.save(data)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls.save({"last_interaction_id": "", "sessions": []})

    @classmethod
    def format_session_table(cls, limit: int = 10) -> str:
        with cls._lock:
            data = cls.load()
            sessions = data.get("sessions") or []
            if not sessions:
                return "No interactions stored."

            sorted_sessions = sorted(
                sessions,
                key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""),
                reverse=True
            )[:limit]

            lines = [
                f"{'ID':<45} | {'Model':<24} | {'Turns':<5} | {'Created':<19} | {'Last Updated':<19}",
                f"{'-'*45}-+-{'-'*24}-+-{'-'*5}-+-{'-'*19}-+-{'-'*19}",
            ]
            for s in sorted_sessions:
                sid = str(s.get("id") or "")[:45]
                smod = str(s.get("model") or "")[:24]
                turns = str(s.get("turn_count") or 1)[:5]
                created = str(s.get("created_at") or "")[:19]
                updated = str(s.get("updated_at") or "")[:19]
                lines.append(f"{sid:<45} | {smod:<24} | {turns:<5} | {created:<19} | {updated:<19}")
            return "\n".join(lines)


class SSL_GeminiInteractionsSafetyConfig(IO.ComfyNode):
    GeminiSafetyConfig = IO.Custom("GEMINI_SAFETY_CONFIG")
    THRESHOLDS = ["off", "block_none", "block_only_high", "block_medium_and_above", "block_low_and_above", "default"]

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SSL_GeminiInteractionsSafetyConfig",
            display_name="Configure Gemini Interactions Safety",
            category="API/Gemini/Interactions",
            inputs=[
                IO.Combo.Input("harassment", options=cls.THRESHOLDS, default="off", tooltip="Negative or harmful comments targeting identity."),
                IO.Combo.Input("hate_speech", options=cls.THRESHOLDS, default="off", tooltip="Content that promotes violence or incites hatred."),
                IO.Combo.Input("sexually_explicit", options=cls.THRESHOLDS, default="off", tooltip="Contains references to sexual acts or other lewd content."),
                IO.Combo.Input("dangerous_content", options=cls.THRESHOLDS, default="off", tooltip="Promotes, facilitates, or encourages harmful acts."),
                IO.Combo.Input("civic_integrity", options=cls.THRESHOLDS, default="off", tooltip="Content that may mislead or influence elections."),
                IO.Combo.Input("image_harassment", options=cls.THRESHOLDS, default="off", tooltip="Images depicting harassment."),
                IO.Combo.Input("image_hate", options=cls.THRESHOLDS, default="off", tooltip="Images depicting hate speech."),
                IO.Combo.Input("image_sexually_explicit", options=cls.THRESHOLDS, default="off", tooltip="Images containing sexually explicit content."),
                IO.Combo.Input("image_dangerous_content", options=cls.THRESHOLDS, default="off", tooltip="Images depicting dangerous content."),
                IO.Combo.Input("jailbreak", options=cls.THRESHOLDS, default="default", tooltip="Prompts designed to bypass safety filters. 'default' keeps filter disabled."),
                IO.Combo.Input("method", options=["default", "severity", "probability"], default="severity", tooltip="Method for blocking content: severity vs probability score."),
            ],
            outputs=[
                cls.GeminiSafetyConfig.Output("safety_config")
            ]
        )

    @classmethod
    def execute(cls, harassment: str = "off", hate_speech: str = "off",
                sexually_explicit: str = "off", dangerous_content: str = "off",
                civic_integrity: str = "off", image_harassment: str = "off",
                image_hate: str = "off", image_sexually_explicit: str = "off",
                image_dangerous_content: str = "off", jailbreak: str = "default",
                method: str = "severity") -> IO.NodeOutput:
        categories = {
            "harassment": harassment,
            "hate_speech": hate_speech,
            "sexually_explicit": sexually_explicit,
            "dangerous_content": dangerous_content,
            "civic_integrity": civic_integrity,
            "image_harassment": image_harassment,
            "image_hate": image_hate,
            "image_sexually_explicit": image_sexually_explicit,
            "image_dangerous_content": image_dangerous_content,
            "jailbreak": jailbreak,
        }
        settings = []
        for cat, threshold in categories.items():
            if threshold != "default":
                entry: dict[str, Any] = {"type": cat, "threshold": threshold}
                if method != "default":
                    entry["method"] = method
                settings.append(entry)
        return IO.NodeOutput(settings)


class SSL_GeminiInteractionsAPIKeyConfig(IO.ComfyNode):
    GemConfig = IO.Custom("GEMINI_CONFIG")
    GeminiSafetyConfig = IO.Custom("GEMINI_SAFETY_CONFIG")

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
                IO.Boolean.Input("use_cache", default=False, optional=True, tooltip="Enable context caching workflow. Interactions API uses implicit server-side caching and session storage."),
                IO.Int.Input("cache_ttl_minutes", default=60, min=1, step=1, optional=True, tooltip="Context cache TTL minutes (retained for config compatibility)."),
                IO.Int.Input("cache_seed", default=0, min=0, max=2147483647, control_after_generate=True, optional=True, tooltip="Gemini generation seed used when context caching is enabled."),
                cls.GeminiSafetyConfig.Input("safety_override", optional=True, tooltip="Optional safety override settings from Configure Gemini Interactions Safety node."),
            ],
            outputs=[
                cls.GemConfig.Output("config")
            ]
        )

    @classmethod
    def execute(cls, api_key: str, api_version: str, use_vertexai_env: bool, vertexai_express: bool,
                vertexai_project: str | None = "", vertexai_location: str | None = "",
                google_application_credentials: str | None = "", use_cache: bool = False,
                cache_ttl_minutes: int = 60, cache_seed: int = 0,
                safety_override: GeminiSafetyConfig.Type | None = None) -> IO.NodeOutput:
        config = {
            "api_key": api_key,
            "api_version": api_version,
            "use_vertexai_env": use_vertexai_env,
            "vertexai_express": vertexai_express,
            "vertexai_project": vertexai_project,
            "vertexai_location": vertexai_location,
            "google_application_credentials": google_application_credentials,
            "use_cache": use_cache,
            "cache_ttl_minutes": cache_ttl_minutes,
            "cache_seed": cache_seed,
            "safety_override": safety_override,
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
    GEMINI_4_ARGON = "gemini-4-argon"
    GEMINI_4_ARGON_CYBER_PERMISSIVE = "gemini-4-argon-cyber-permissive"
    ARGON_MODELS = (GEMINI_4_ARGON, GEMINI_4_ARGON_CYBER_PERMISSIVE)
    VIDEO_MIME_TYPE = "video/mp4"
    VIDEO_INLINE_LIMIT_BYTES = 100 * 1024 * 1024

    GEMINI_4_FLASH_PREVIEW = "gemini-4-flash-preview"

    ALL_HARM_CATEGORIES = [
        "harassment",
        "hate_speech",
        "sexually_explicit",
        "dangerous_content",
        "civic_integrity",
        "image_harassment",
        "image_hate",
        "image_sexually_explicit",
        "image_dangerous_content",
        "jailbreak",
    ]

    THINKING_MODELS = [
        "gemini-1.5-pro-002", "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25",
        "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", GEMINI_3_7_FLASH, GEMINI_3_8_FLASH, GEMINI_4_ARGON, GEMINI_4_ARGON_CYBER_PERMISSIVE, GEMINI_4_FLASH_PREVIEW, "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"
    ]
    GEN3_THINKING_MODELS = [
        "gemini-pro-latest", "gemini-flash-latest", "gemini-3.1-pro-preview",
        "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", GEMINI_3_7_FLASH, GEMINI_3_8_FLASH, GEMINI_4_ARGON, GEMINI_4_ARGON_CYBER_PERMISSIVE, GEMINI_4_FLASH_PREVIEW
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
            is_input_list=True,
            inputs=[
                cls.GemConfig.Input("config"),
                IO.String.Input("prompt", multiline=True),
                IO.String.Input("system_instruction", default="You are a helpful AI assistant.", multiline=True),
                IO.Combo.Input("model", options=["gemini-1.5-pro-002", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", cls.GEMINI_3_7_FLASH, cls.GEMINI_3_8_FLASH, cls.GEMINI_4_ARGON, cls.GEMINI_4_ARGON_CYBER_PERMISSIVE, "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image", "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"], default="gemini-2.5-flash"),
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
                IO.Combo.Input("thinking_level", options=["None", "minimal", "low", "medium", "high"], default="low", tooltip="Thinking level for reasoning models."),
                IO.Combo.Input("media_resolution", options=["unspecified", "low", "medium", "high", "ultra_high"], default="unspecified", tooltip="Set input media resolution for image, video and pdf."),
                IO.String.Input("retry_pattern", default="(PROHIBITED_CONTENT|UNAVAILABLE|request/processing|RESOURCE_EXHAUSTED)", optional=True, multiline=False, tooltip="Regex pattern to match in response text. If matched, retry with new seed. Leave empty to disable."),
                IO.Int.Input("max_retries", default=3, min=0, max=10, step=1, tooltip="Maximum number of retry attempts when pattern matches. 0 disables retry."),
                IO.String.Input("timeout_fallback_text", default="", optional=True, multiline=True, tooltip="Text returned when the Gemini request times out. Leave empty to return the standard timeout message."),
                IO.Combo.Input("image_size", options=["None", "512", "1K", "2K", "4K"], default="None", optional=True, tooltip="Generated image resolution. Gemini 3.1 Flash Lite Image supports only 1K; 512 is supported only by Gemini 3.1 Flash Image."),
                IO.Combo.Input("store", options=["new", "previous", "manual", "disabled"], default="new", optional=True, tooltip="Session mode: 'new' starts a fresh interaction and stores it; 'previous' continues the last stored interaction; 'manual' continues from previous_interaction_id; 'disabled' does not store or continue."),
                IO.String.Input("previous_interaction_id", default="", optional=True, multiline=False, tooltip="ID of previous interaction to continue when store is set to 'manual'."),
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
                IO.String.Output("text", is_output_list=True),
                IO.Image.Output("image", is_output_list=True),
                IO.Int.Output("final_actual_seed", is_output_list=True),
                IO.String.Output("structured_output", display_name="Structured output", is_output_list=True),
                IO.String.Output("thoughts", display_name="Thoughts", is_output_list=True),
                IO.String.Output("interaction_id", display_name="Interaction ID", is_output_list=True),
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
            if isinstance(image_batch, list):
                image_batch = image_batch[0] if image_batch else None
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
    def _prepare_image_inputs(cls, image_inputs: IO.Autogrow.Type | None) -> dict[str, Any] | None:
        if not image_inputs:
            return None
        image_frames, image_batch_counts = cls._ordered_image_frames(image_inputs)
        if not image_frames:
            return {
                "image_hashes": (),
                "png_bytes_list": [],
                "batch_counts": [],
            }
        tensor_hashes = []
        png_bytes_list = []
        for _, _, image_frame in image_frames:
            np_frame = np.ascontiguousarray(image_frame.cpu().numpy())
            tensor_hashes.append(hashlib.sha256(np_frame.tobytes()).hexdigest())
            img_array = (np_frame * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            png_bytes_list.append(img_byte_arr.getvalue())
        batch_summary = ", ".join(
            f"{socket_name}={batch_size}"
            for socket_name, batch_size in image_batch_counts
        )
        print(
            f"[INFO] Prepared Gemini image inputs: {batch_summary}, "
            f"total={len(png_bytes_list)}"
        )
        return {
            "image_hashes": tuple(tensor_hashes),
            "png_bytes_list": png_bytes_list,
            "batch_counts": image_batch_counts,
        }

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
            if model_name == cls.GEMINI_4_ARGON_CYBER_PERMISSIVE:
                return "high"
            return "medium"
        if thinking_level == "minimal":
            if model_name in (cls.GEMINI_3_8_FLASH, cls.GEMINI_4_ARGON, cls.GEMINI_4_ARGON_CYBER_PERMISSIVE):
                raise ValueError(
                    f"{model_name} thinking_level must be low, medium, or high."
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

    CORE_HARM_CATEGORIES = [
        "harassment",
        "hate_speech",
        "sexually_explicit",
        "dangerous_content",
        "civic_integrity",
        "image_harassment",
        "image_hate",
        "image_sexually_explicit",
        "image_dangerous_content",
    ]

    @classmethod
    def _resolve_safety_settings(cls, config: dict[str, Any]):
        override = config.get("safety_override")
        if override and isinstance(override, list) and override:
            return list(override)

        is_enterprise = bool(config.get("use_vertexai_env", False) or config.get("vertexai_express", False))
        if is_enterprise:
            return [
                {"type": cat, "threshold": "off", "method": "severity"}
                for cat in cls.CORE_HARM_CATEGORIES
            ]
        return [
            {"type": cat, "threshold": "block_none"}
            for cat in cls.CORE_HARM_CATEGORIES
        ]

    STORE_MODES = ("new", "previous", "manual", "disabled")

    @classmethod
    def _normalize_store_mode(cls, store: Any) -> str:
        if store is None:
            return "new"
        if isinstance(store, bool):
            return "new" if store else "disabled"
        mode = str(store).strip().lower()
        if mode == "true":
            return "new"
        if mode == "false":
            return "disabled"
        if mode not in cls.STORE_MODES:
            raise ValueError(f"store must be one of: {', '.join(cls.STORE_MODES)}.")
        return mode

    @classmethod
    def _compute_fingerprint_and_check_cache(cls, config, prompt, system_instruction, model, max_output_tokens,
                                             include_images, aspect_ratio, bypass_mode, use_seed, seed,
                                             video_hash=None, video_mime_type=None, video_fps=None,
                                             image_inputs: IO.Autogrow.Type | None = None,
                                             use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, timeout=30,
                                             include_thoughts=False, thinking_level=None, media_resolution=None,
                                             retry_pattern="", max_retries=3, use_cache=False,
                                             cache_ttl_minutes=60, cache_seed=0, image_size="None",
                                             store="new", previous_interaction_id="", safety_settings=None,
                                             schema_identity=None, precomputed_image_hashes=None):

        def get_tensor_hash(tensor):
            if tensor is None:
                return "None"
            try:
                return hashlib.sha256(np.ascontiguousarray(tensor.cpu().numpy()).tobytes()).hexdigest()
            except Exception as e:
                print(f"[WARNING] Cache hashing failed for image: {e}")
                return "Error"

        if precomputed_image_hashes is not None:
            image_hashes = precomputed_image_hashes
        else:
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

        elif model in (cls.GEMINI_3_7_FLASH, cls.GEMINI_3_8_FLASH, cls.GEMINI_4_ARGON, cls.GEMINI_4_ARGON_CYBER_PERMISSIVE):
            eff_thinking_level = cls._resolve_gemini_3_7_thinking_level(thinking_level, model)
            eff_include_thoughts = include_thoughts

        elif model in cls.GEN3_THINKING_MODELS and thinking_level not in (None, "None"):
            eff_thinking_level = thinking_level
            eff_include_thoughts = include_thoughts

        elif thinking_level not in (None, "None"):
            eff_thinking_level = thinking_level
            eff_include_thoughts = include_thoughts

        sanitized_config = dict(config)
        for cache_setting in ("use_cache", "cache_ttl_minutes", "cache_seed"):
            sanitized_config.pop(cache_setting, None)
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
            bool(use_cache),
            int(cache_ttl_minutes) if use_cache else 0,
            int(cache_seed) if use_cache and use_seed else 0,
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
            str(store),
            str(previous_interaction_id),
            str(safety_settings),
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
        value = os.getenv(name)
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
    def _check_interrupt(cls):
        comfy.model_management.throw_exception_if_processing_interrupted()

    @classmethod
    def _configure_interactions_client(cls, client: Any, timeout: float) -> None:
        try:
            interactions = getattr(client, "interactions", None)
            sdk_cfg = getattr(interactions, "sdk_configuration", None)
            if sdk_cfg is not None:
                sdk_cfg.timeout_ms = max(1, int(float(timeout) * 1000))
                retry_cfg = getattr(sdk_cfg, "retry_config", None)
                if retry_cfg is not None:
                    retry_cfg.strategy = "none"
                    retry_cfg.max_retries = 0
                    retry_cfg.retry_connection_errors = False
        except Exception:
            pass

    @classmethod
    def _abort_client_transport(cls, client: Any, client_key: Any = None) -> None:
        if client_key is not None:
            cls._client_cache.pop(client_key, None)
        if client is None:
            return
        for owner in (getattr(client, "_api_client", None), getattr(getattr(client, "_interactions", None), "sdk_configuration", None)):
            if owner is None:
                continue
            for attr in ("_httpx_client", "client"):
                http_client = getattr(owner, attr, None)
                if http_client is not None and hasattr(http_client, "close"):
                    try:
                        http_client.close()
                    except Exception:
                        pass

    @classmethod
    def _wait_for_result(cls, result_queue: "queue.Queue[Tuple[str, Any]]", timeout: float) -> Tuple[str, Any]:
        deadline = time.monotonic() + float(timeout)
        while True:
            cls._check_interrupt()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty()
            try:
                return result_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise

    @classmethod
    def _execute_single(cls, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
                        include_images, aspect_ratio, bypass_mode, thinking_budget,
                        use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, use_seed=False, seed=0, timeout=30,
                        include_thoughts=False, thinking_level=None, media_resolution=None,
                        retry_pattern="", max_retries=3, timeout_fallback_text="",
                        image_size="None", store="new", previous_interaction_id="",
                        video: GeminiVideoConfig.Type | None = None,
                        image_inputs: IO.Autogrow.Type | None = None,
                        response_schema: ResponseSchema.Type | None = None,
                        prepared_images: dict[str, Any] | None = None,
                        precomputed_video: Tuple[bytes | None, str | None, str | None, int | None] | None = None,
                        chained_previous_id: str = "",
                        fingerprint_seed: int | None = None) -> IO.NodeOutput:

        cls._check_interrupt()
        print(f"[INFO] SSL_GeminiInteractionsTextPrompt execute called, model: {model}")

        # If model is an image generation model, reroute to SSL_GeminiTextPrompt (GenerateContent)
        if model in cls.IMAGE_MODELS:
            print(f"[INFO] Model '{model}' is an image generation model; rerouting to GenerateContent engine (SSL_GeminiTextPrompt).")
            gen_output = SSL_GeminiTextPrompt.execute(
                config=config,
                prompt=prompt,
                system_instruction=system_instruction,
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                include_images=include_images,
                aspect_ratio=aspect_ratio,
                bypass_mode=bypass_mode,
                thinking_budget=thinking_budget,
                use_proxy=use_proxy,
                proxy_host=proxy_host,
                proxy_port=proxy_port,
                use_seed=use_seed,
                seed=seed,
                timeout=timeout,
                include_thoughts=include_thoughts,
                thinking_level=thinking_level,
                media_resolution=media_resolution,
                retry_pattern=retry_pattern,
                max_retries=max_retries,
                timeout_fallback_text=timeout_fallback_text,
                image_size=image_size,
                video=video,
                image_inputs=image_inputs,
                response_schema=response_schema,
            )
            return IO.NodeOutput(
                gen_output[0],  # text
                gen_output[1],  # image
                gen_output[2],  # final_actual_seed
                gen_output[3],  # structured_output
                gen_output[4],  # thoughts
                "",             # interaction_id
            )

        use_cache = bool(config.get("use_cache", False))
        cache_ttl_minutes = int(config.get("cache_ttl_minutes", 60))
        cache_seed = int(config.get("cache_seed", 0))
        fp_seed = int(seed if fingerprint_seed is None else fingerprint_seed)

        schema_identity = None
        schema_snapshot = None
        if response_schema is not None:
            if not isinstance(response_schema, dict):
                raise ValueError("Response format must be a schema dictionary from the schema output.")
            if include_images and model in cls.IMAGE_MODELS:
                raise ValueError("Response format cannot be combined with image generation. Disable include_images.")
            schema_identity = json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            schema_snapshot = json.loads(schema_identity)

        if precomputed_video is not None:
            video_bytes, video_mime_type, video_hash, video_fps = precomputed_video
        else:
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
                output_seed = cache_seed if use_cache and use_seed else seed if use_seed else 0
                return IO.NodeOutput(f"Error processing input video: {e}", cls.generate_empty_image(), output_seed, "", "", "")

            video_hash = hashlib.sha256(video_bytes).hexdigest() if video_bytes is not None else None
        store_mode = cls._normalize_store_mode(store)
        store_enabled = store_mode != "disabled"
        if store_mode == "new":
            effective_previous_id = ""
        elif store_mode == "previous":
            effective_previous_id = str(chained_previous_id or "").strip() or GeminiInteractionsStoreManager.get_last_interaction_id()
            if effective_previous_id:
                print(f"[INFO] Automatically continuing last interaction from store: {effective_previous_id}")
        elif store_mode == "manual":
            effective_previous_id = str(previous_interaction_id or "").strip() or str(chained_previous_id or "").strip()
        else:
            effective_previous_id = ""

        resolved_safety_settings = cls._resolve_safety_settings(config)
        precomputed_image_hashes = prepared_images["image_hashes"] if prepared_images is not None else None
        fingerprint, cached = cls._compute_fingerprint_and_check_cache(
            config, prompt, system_instruction, model, max_output_tokens,
            include_images, aspect_ratio, bypass_mode, use_seed, fp_seed,
            video_hash, video_mime_type, video_fps,
            image_inputs,
            use_proxy, proxy_host, proxy_port, timeout,
            include_thoughts, thinking_level, media_resolution,
            retry_pattern, max_retries, use_cache=use_cache,
            cache_ttl_minutes=cache_ttl_minutes, cache_seed=cache_seed,
            image_size=image_size, store=store_mode,
            previous_interaction_id=effective_previous_id,
            safety_settings=resolved_safety_settings,
            schema_identity=schema_identity,
            precomputed_image_hashes=precomputed_image_hashes,
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

        api_seed = cache_seed if use_cache else seed
        actual_seed = cls._handle_seed(use_seed, api_seed)
        input_seed = api_seed

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
                if (use_vertexai_env or vertexai_express) and api_version == "v1beta":
                    api_version = "v1beta1"
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

                        env_project = os.getenv("GOOGLE_CLOUD_PROJECT")
                        env_proj = env_project.strip() if env_project else (project or credential_project)
                        assert env_proj, "GOOGLE_CLOUD_PROJECT is empty"

                        env_location = os.getenv("GOOGLE_CLOUD_LOCATION")
                        env_loc = env_location.strip() if env_location else location
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
            media_items = []

            if prepared_images is not None:
                for img_bytes in prepared_images["png_bytes_list"]:
                    img_item = {
                        "type": "image",
                        "data": BytesIO(img_bytes),
                        "mime_type": "image/png",
                    }
                    if media_resolution and media_resolution != "unspecified":
                        img_item["resolution"] = media_resolution.lower()
                    media_items.append(img_item)
            else:
                image_frames, image_batch_counts = cls._ordered_image_frames(image_inputs)
                if image_frames:
                    try:
                        for _, _, image_frame in image_frames:
                            img_array = image_frame.cpu().numpy()
                            img_array = (img_array * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_array)
                            img_byte_arr = BytesIO()
                            pil_img.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            img_item = {
                                "type": "image",
                                "data": img_byte_arr,
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
                            f"total={len(media_items)}"
                        )
                    except Exception as e:
                        print(f"[ERROR] Error processing input image: {str(e)}")
                        return IO.NodeOutput(f"Error processing input image: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0, "", "", "")

            if video_bytes is not None:
                video_stream = BytesIO(video_bytes)
                v_item = {
                    "type": "video",
                    "data": video_stream,
                    "mime_type": video_mime_type,
                }
                if media_resolution and media_resolution != "unspecified":
                    v_item["resolution"] = media_resolution.lower()
                media_items.append(v_item)

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
            if model in (cls.GEMINI_3_7_FLASH, cls.GEMINI_3_8_FLASH, cls.GEMINI_4_ARGON, cls.GEMINI_4_ARGON_CYBER_PERMISSIVE):
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

            # Background worker queue for API call
            cls._configure_interactions_client(client, timeout)
            start_time = time.time()
            result_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
            cancel_event = threading.Event()

            def create_interaction_once(current_seed, target_previous_id):
                if cancel_event.is_set():
                    raise RuntimeError("Request cancelled")
                call_gen_config = dict(generation_config)
                if use_seed and current_seed is not None:
                    call_gen_config["seed"] = int(current_seed)

                kwargs: dict[str, Any] = {
                    "model": model,
                    "input": input_contents,
                }
                if resolved_safety_settings is not None:
                    kwargs["safety_settings"] = list(resolved_safety_settings)
                if padded_system_instruction:
                    kwargs["system_instruction"] = padded_system_instruction
                if call_gen_config:
                    kwargs["generation_config"] = call_gen_config
                if response_formats:
                    kwargs["response_format"] = response_formats
                kwargs["store"] = store_enabled
                if target_previous_id and str(target_previous_id).strip():
                    kwargs["previous_interaction_id"] = str(target_previous_id).strip()

                return client.interactions.create(**kwargs)

            def api_call():
                nonlocal effective_previous_id
                last_api_exception = None
                api_response = None
                max_attempts = 1
                for attempt in range(max_attempts):
                    if cancel_event.is_set():
                        return
                    try:
                        current_seed = (actual_seed + attempt) if (use_seed and actual_seed is not None) else None
                        response = create_interaction_once(current_seed, effective_previous_id)
                        if cancel_event.is_set():
                            return
                        api_response = response
                        break
                    except Exception as e:
                        if cancel_event.is_set():
                            return
                        err_str = str(e).lower()
                        # If previous interaction was specified/auto-used but expired or not found, fall back to new interaction
                        if effective_previous_id and ("not found" in err_str or "expired" in err_str or "404" in err_str or "invalid" in err_str):
                            print(f"[INFO] Previous interaction '{effective_previous_id}' expired/invalid ({e}). Starting fresh interaction.")
                            effective_previous_id = ""
                            try:
                                response = create_interaction_once(current_seed, "")
                                if cancel_event.is_set():
                                    return
                                api_response = response
                                break
                            except Exception as retry_err:
                                last_api_exception = retry_err
                        else:
                            last_api_exception = e
                if cancel_event.is_set():
                    return
                if api_response is None:
                    result_queue.put(("error", last_api_exception))
                    return

                try:
                    # Check for safety blocks or API errors
                    api_errors = getattr(api_response, "errors", None) or []
                    if api_errors:
                        err_details = []
                        for err in api_errors:
                            msg = getattr(err, "message", None) or str(err)
                            code = getattr(err, "code", None)
                            err_details.append(f"[{code}] {msg}" if code else msg)
                        raise ValueError(f"Interaction blocked/failed: {'; '.join(err_details)}")

                    api_status = getattr(api_response, "status", None)
                    if api_status in ("failed", "incomplete"):
                        step_errors = []
                        for step in getattr(api_response, "steps", None) or []:
                            s_err = getattr(step, "error", None)
                            if s_err:
                                s_msg = getattr(s_err, "message", None) or str(s_err)
                                step_errors.append(s_msg)
                        if step_errors:
                            raise ValueError(f"Interaction {api_status}: {'; '.join(step_errors)}")
                        raise ValueError(f"Interaction ended with status: {api_status}")

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

                    usage = getattr(api_response, "usage", None)
                    if usage:
                        cached_tokens = getattr(usage, "total_cached_tokens", 0) or getattr(usage, "cached_content_token_count", 0) or 0
                        if cached_tokens:
                            print(f"[INFO] Gemini Interactions implicit cache hit: {cached_tokens} cached tokens")

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
                    status, result = cls._wait_for_result(result_queue, timeout)
                    if status == "success":
                        text_output, image_tensor, thoughts_output, interaction_id_output = result
                        structured_output = text_output if schema_snapshot is not None else ""
                        is_success = True

                        if retry_pattern and max_retries > 0 and retry_attempt < max_retries:
                            try:
                                compiled_pattern = re.compile(retry_pattern, re.IGNORECASE)
                                if compiled_pattern.search(text_output):
                                    cls._check_interrupt()
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
                                    cls._check_interrupt()
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
                    cancel_event.set()
                    cls._abort_client_transport(client, client_key)
                    text_output = timeout_fallback_text or f"Gemini API request/processing timed out, waited {timeout} seconds."
                except comfy.model_management.InterruptProcessingException:
                    cancel_event.set()
                    cls._abort_client_transport(client, client_key)
                    raise

        except comfy.model_management.InterruptProcessingException:
            raise
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

        # Record interaction in local store on success
        if is_success and store_enabled and interaction_id_output:
            GeminiInteractionsStoreManager.record_interaction(interaction_id_output, model)

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

    @classmethod
    def execute(cls, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
                include_images, aspect_ratio, bypass_mode, thinking_budget,
                use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, use_seed=False, seed=0, timeout=30,
                include_thoughts=False, thinking_level=None, media_resolution=None,
                retry_pattern="", max_retries=3, timeout_fallback_text="",
                image_size="None", store="new", previous_interaction_id="",
                video: GeminiVideoConfig.Type | None = None,
                image_inputs: IO.Autogrow.Type | None = None,
                response_schema: ResponseSchema.Type | None = None) -> IO.NodeOutput:

        def _is_list_arg(val):
            return isinstance(val, list) and not isinstance(val, (str, bytes, dict))

        def _extract_seed_list(val):
            if not isinstance(val, (list, tuple)):
                return None
            flat = []
            for item in val:
                if isinstance(item, (list, tuple)):
                    sub = _extract_seed_list(item)
                    if sub is not None:
                        flat.extend(sub)
                elif torch.is_tensor(item):
                    flat.extend(int(x) for x in item.detach().cpu().reshape(-1).tolist())
                elif isinstance(item, np.ndarray):
                    flat.extend(int(x) for x in item.reshape(-1).tolist())
                elif item is not None:
                    flat.append(int(item))
            return flat

        is_prompt_list = _is_list_arg(prompt)
        is_video_list = _is_list_arg(video)
        extracted_seeds = _extract_seed_list(seed)
        is_seed_list = extracted_seeds is not None
        is_config_list = _is_list_arg(config)
        max_image_list_len = (
            max((len(v) for v in image_inputs.values() if _is_list_arg(v)), default=0)
            if isinstance(image_inputs, dict)
            else 0
        )
        is_image_list = max_image_list_len > 0

        is_batch = is_prompt_list or is_video_list or is_seed_list or is_config_list or is_image_list

        if not is_batch:
            return cls._execute_single(
                config=config, prompt=prompt, system_instruction=system_instruction, model=model,
                temperature=temperature, top_p=top_p, top_k=top_k, max_output_tokens=max_output_tokens,
                include_images=include_images, aspect_ratio=aspect_ratio, bypass_mode=bypass_mode,
                thinking_budget=thinking_budget, use_proxy=use_proxy, proxy_host=proxy_host,
                proxy_port=proxy_port, use_seed=use_seed, seed=seed, timeout=timeout,
                include_thoughts=include_thoughts, thinking_level=thinking_level,
                media_resolution=media_resolution, retry_pattern=retry_pattern, max_retries=max_retries,
                timeout_fallback_text=timeout_fallback_text, image_size=image_size, store=store,
                previous_interaction_id=previous_interaction_id,
                video=video, image_inputs=image_inputs, response_schema=response_schema,
            )

        prompt_list = prompt if is_prompt_list else [prompt]
        video_list = video if is_video_list else [video]
        seed_list = extracted_seeds if is_seed_list else [seed]
        config_list = config if is_config_list else [config]

        batch_count = max(
            len(prompt_list) if is_prompt_list else 1,
            len(video_list) if is_video_list else 1,
            len(seed_list) if is_seed_list else 1,
            len(config_list) if is_config_list else 1,
            max_image_list_len if is_image_list else 1,
        )

        def _unwrap_scalar(val, default=None):
            while isinstance(val, list) and len(val) > 0 and (isinstance(val[0], list) or not isinstance(val, dict)):
                val = val[0]
            if isinstance(val, list) and len(val) == 1:
                val = val[0]
            return val if val is not None else default

        def _get_aligned(val, idx, default=None):
            if isinstance(val, list) and val:
                return val[min(idx, len(val) - 1)]
            return val if val is not None else default

        config_val = _unwrap_scalar(config, {})
        model_val = _unwrap_scalar(model, "gemini-2.5-flash")
        temp_val = float(_unwrap_scalar(temperature, 1.0))
        top_p_val = float(_unwrap_scalar(top_p, 0.95))
        top_k_val = int(_unwrap_scalar(top_k, 40))
        max_tokens_val = int(_unwrap_scalar(max_output_tokens, 8192))
        inc_imgs_val = bool(_unwrap_scalar(include_images, False))
        aspect_val = str(_unwrap_scalar(aspect_ratio, "None"))
        bypass_val = str(_unwrap_scalar(bypass_mode, "None"))
        budget_val = int(_unwrap_scalar(thinking_budget, 0))
        proxy_val = bool(_unwrap_scalar(use_proxy, False))
        phost_val = str(_unwrap_scalar(proxy_host, "127.0.0.1"))
        pport_val = int(_unwrap_scalar(proxy_port, 7890))
        use_seed_val = bool(_unwrap_scalar(use_seed, False))
        timeout_val = int(_unwrap_scalar(timeout, 30))
        thoughts_val = bool(_unwrap_scalar(include_thoughts, False))
        th_level_val = _unwrap_scalar(thinking_level, None)
        media_res_val = _unwrap_scalar(media_resolution, None)
        retry_pat_val = str(_unwrap_scalar(retry_pattern, ""))
        max_retries_val = int(_unwrap_scalar(max_retries, 3))
        timeout_fb_val = str(_unwrap_scalar(timeout_fallback_text, ""))
        img_size_val = str(_unwrap_scalar(image_size, "None"))
        store_val = cls._normalize_store_mode(_unwrap_scalar(store, "new"))
        resp_schema_val = _unwrap_scalar(response_schema, None)
        is_prev_id_list = _is_list_arg(previous_interaction_id) and len(previous_interaction_id) > 1
        base_seed = int(seed_list[0]) if (is_seed_list and len(seed_list) > 0) else int(_unwrap_scalar(seed, 0))
        has_distinct_item_inputs = (
            (is_prompt_list and len(prompt_list) > 1)
            or (is_video_list and len(video_list) > 1)
            or (max_image_list_len > 1)
        )

        def make_item_kwargs(idx, chained_prev_id="", prepared_imgs=None, precomp_vid=None):
            p = prompt_list[min(idx, len(prompt_list) - 1)] if prompt_list else ""
            v = video_list[min(idx, len(video_list) - 1)] if video_list else None
            if isinstance(image_inputs, dict):
                item_imgs = {}
                for k, val in image_inputs.items():
                    if _is_list_arg(val) and val:
                        item_imgs[k] = val[min(idx, len(val) - 1)]
                    else:
                        item_imgs[k] = val
            else:
                item_imgs = image_inputs

            s = (base_seed + idx) if (base_seed != 0 and batch_count > 1) else base_seed
            fp_seed = base_seed if has_distinct_item_inputs else s

            raw_cfg = _get_aligned(config_list, idx, config_val)
            item_cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else dict(config_val)

            sys_inst = _get_aligned(system_instruction, idx, "You are a helpful AI assistant.")
            eff_store = "previous" if (store_val == "new" and idx > 0) else store_val
            if eff_store == "manual":
                if is_prev_id_list or idx == 0:
                    prev_id = str(_get_aligned(previous_interaction_id, idx, "") or "").strip()
                else:
                    prev_id = ""
            else:
                prev_id = ""

            return {
                "config": item_cfg,
                "prompt": p,
                "system_instruction": sys_inst,
                "model": model_val,
                "temperature": temp_val,
                "top_p": top_p_val,
                "top_k": top_k_val,
                "max_output_tokens": max_tokens_val,
                "include_images": inc_imgs_val,
                "aspect_ratio": aspect_val,
                "bypass_mode": bypass_val,
                "thinking_budget": budget_val,
                "use_proxy": proxy_val,
                "proxy_host": phost_val,
                "proxy_port": pport_val,
                "use_seed": use_seed_val,
                "seed": s,
                "timeout": timeout_val,
                "include_thoughts": thoughts_val,
                "thinking_level": th_level_val,
                "media_resolution": media_res_val,
                "retry_pattern": retry_pat_val,
                "max_retries": max_retries_val,
                "timeout_fallback_text": timeout_fb_val,
                "image_size": img_size_val,
                "store": eff_store,
                "previous_interaction_id": prev_id,
                "video": v,
                "image_inputs": item_imgs,
                "response_schema": resp_schema_val,
                "prepared_images": prepared_imgs,
                "precomputed_video": precomp_vid,
                "chained_previous_id": chained_prev_id,
                "fingerprint_seed": fp_seed,
            }

        if batch_count == 1:
            single_res = cls._execute_single(**make_item_kwargs(0))
            return IO.NodeOutput(
                [single_res[0]],
                [single_res[1]],
                [single_res[2]],
                [single_res[3]],
                [single_res[4]],
                [single_res[5]],
            )

        print(f"[INFO] Preparing sequential list execution for {batch_count} items (model: {model_val})")
        prepared_images = None
        if image_inputs and max_image_list_len <= 1 and model_val not in cls.IMAGE_MODELS:
            try:
                prepared_images = cls._prepare_image_inputs(image_inputs)
            except Exception:
                prepared_images = None

        precomputed_video = None
        if len(video_list) == 1 and video_list[0] is not None and model_val not in cls.IMAGE_MODELS:
            try:
                v0 = video_list[0]
                v_input = v0.get("video")
                v_fps = int(v0.get("fps", 1))
                v_pad = bool(v0.get("pad_at_start", False))
                v_dur = bool(v0.get("duration_aware_padding", False))
                v_bytes, v_mime = cls._serialize_video(v_input, v_fps, v_pad, v_dur)
                v_hash = hashlib.sha256(v_bytes).hexdigest() if v_bytes is not None else None
                precomputed_video = (v_bytes, v_mime, v_hash, v_fps)
            except Exception:
                precomputed_video = None

        batch_results = []
        chained_interaction_id = ""
        for idx in range(batch_count):
            cls._check_interrupt()
            item_res = cls._execute_single(
                **make_item_kwargs(
                    idx,
                    chained_prev_id=chained_interaction_id,
                    prepared_imgs=prepared_images,
                    precomp_vid=precomputed_video,
                )
            )
            batch_results.append(item_res)
            if store_val != "disabled" and item_res[5]:
                chained_interaction_id = str(item_res[5]).strip()

        return IO.NodeOutput(
            [r[0] for r in batch_results],
            [r[1] for r in batch_results],
            [r[2] for r in batch_results],
            [r[3] for r in batch_results],
            [r[4] for r in batch_results],
            [r[5] for r in batch_results],
        )


class SSL_GeminiInteractionsStore(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SSL_GeminiInteractionsStore",
            display_name="Gemini Interactions Store",
            category="API/Gemini/Interactions",
            inputs=[
                IO.Boolean.Input("clear_history", default=False, tooltip="Clear all recorded interactions from local store."),
                IO.Int.Input("limit", default=10, min=1, max=100, step=1, tooltip="Number of recent sessions to display in table."),
            ],
            outputs=[
                IO.String.Output("session_table", display_name="Session Table"),
                IO.String.Output("last_interaction_id", display_name="Last Interaction ID"),
            ],
        )

    @classmethod
    def execute(cls, clear_history: bool = False, limit: int = 10) -> IO.NodeOutput:
        if clear_history:
            GeminiInteractionsStoreManager.clear()
            return IO.NodeOutput("History cleared.", "")

        table = GeminiInteractionsStoreManager.format_session_table(limit=limit)
        last_id = GeminiInteractionsStoreManager.get_last_interaction_id()
        return IO.NodeOutput(table, last_id)

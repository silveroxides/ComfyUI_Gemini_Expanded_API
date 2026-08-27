import os
import json
import uuid
import re
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import folder_paths  # type: ignore[reportMissingImports]
from comfy_api.latest import ComfyExtension, UI, IO  # type: ignore[reportMissingImports]
from google import genai
from google.genai import types
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

class GetKeyAPI(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="GetKeyAPI",
            display_name="Get API Key from JSON",
            category="utils/api_keys",
            inputs=[
                IO.String.Input("json_path", default="./input/apikeys.json", multiline=False, tooltip="Path to a .json file with simple top level structure with name as key and api-key as value. See example in custom node folder."),
                IO.Combo.Input("key_id_method", options=["custom", "random_rotate", "increment_rotate"], default="custom", tooltip="custom sets api-key to the api-key with the name set in the key_id widget. random_rotate randomly switches between keys if multiple in the .json and increment_rotate does it in order from first to last, then repeats."),
                IO.Int.Input("rotation_interval", default=0, min=0, tooltip="how many steps to jump when doing rotate."),
                IO.String.Input("key_id", default="placeholder", multiline=False, optional=True, tooltip="Put name of key in the .json here if using custom in key_id_method."),
            ],
            outputs=[
                IO.String.Output("API_KEY")
            ]
        )

    @classmethod
    def execute(cls, json_path: str, key_id_method: str, rotation_interval: int, key_id: str | None = "placeholder") -> IO.NodeOutput:
        api_keys_data = None
        absolute_json_path = os.path.abspath(json_path)

        try:
            with open(absolute_json_path, 'r') as f:
                api_keys_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"RotateKeyAPI Error: JSON file not found at {absolute_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"RotateKeyAPI Error: Could not decode JSON from {absolute_json_path}. Check file format.")
        except Exception as e:
            raise RuntimeError(f"RotateKeyAPI Error: Unexpected error reading file {absolute_json_path}: {e}")

        if not isinstance(api_keys_data, dict):
            raise ValueError(f"RotateKeyAPI Error: JSON content is not a dictionary in {absolute_json_path}. Expected format: {{'key_id': 'api_key', ...}}")

        if not api_keys_data:
             raise ValueError(f"RotateKeyAPI Error: The JSON dictionary in {absolute_json_path} is empty.")

        selected_key_value = None

        if key_id_method == "custom":
            if key_id == "placeholder":
                 print("RotateKeyAPI Warning: 'custom' method selected but 'key_id' is still the default 'placeholder'. Ensure this is intended or provide a valid key ID.")

            selected_key_value = api_keys_data.get(key_id)

            if selected_key_value is None:
                 raise ValueError(f"RotateKeyAPI Error: Custom key ID '{key_id}' not found in the JSON dictionary keys.")


        elif key_id_method == "random_rotate":
            api_keys_list = list(api_keys_data.values())

            selected_key_value = random.choice(api_keys_list)

        elif key_id_method == "increment_rotate":
             api_keys_list = list(api_keys_data.values())

             index = rotation_interval % len(api_keys_list)

             try:
                selected_key_value = api_keys_list[index]
             except IndexError:
                 raise IndexError(f"RotateKeyAPI Error: Calculated index {index} (from interval {rotation_interval}) is out of bounds for list of size {len(api_keys_list)}.")
             except Exception as e:
                  raise RuntimeError(f"RotateKeyAPI Error: Unexpected error accessing item at index {index}: {e}")

        if not isinstance(selected_key_value, str) or not selected_key_value:
             raise ValueError(f"RotateKeyAPI Error: Retrieved value for selected key is not a valid string. Value: {selected_key_value}")

        print(f"RotateKeyAPI: Successfully retrieved API key using method '{key_id_method}'.")
        return IO.NodeOutput(selected_key_value)



class SSL_GeminiAPIKeyConfig(IO.ComfyNode):
    GemConfig = IO.Custom("GEMINI_CONFIG")

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="SSL_GeminiAPIKeyConfig",
            display_name="Configure Gemini API Key",
            category="API/Gemini",
            inputs=[
                IO.String.Input("api_key", multiline=False, default=""),
                IO.Combo.Input("api_version", options=["v1", "v1alpha", "v1beta", "v1beta1", "v2beta"], default="v1alpha", tooltip="Select API version to use. v1alpha, v1beta and v2beta are Gemini API specific while v1beta1 is Vertex AI specific. Both can use v1"),
                IO.Boolean.Input("use_vertexai_env", default=False, tooltip="Uses Gemini Enterprise Agent Platform (formerly Vertex AI) with GOOGLE_GENAI_USE_ENTERPRISE or the legacy GOOGLE_GENAI_USE_VERTEXAI environment variable."),
                IO.Boolean.Input("vertexai_express", default=False),
                IO.String.Input("vertexai_project", optional=True),
                IO.String.Input("vertexai_location", optional=True),
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



class SSL_GeminiTextPrompt(IO.ComfyNode):
    GemConfig = IO.Custom("GEMINI_CONFIG")
    _cache: dict = {}
    _seed_map_cache: dict = {}  # Maps (input_seed, fingerprint_without_seed) -> successful_gemini_seed
    _client_cache: dict = {}  # Maps client_key tuple -> genai.Client instance
    GEMINI_3_7_FLASH = "gemini-3.7-flash"

    # Placeholder only. Keep this out of the model combo until Google's hosted-model
    # documentation confirms the final ID and gemini-3.6-flash-equivalent capabilities
    # for both Gemini Developer API and Enterprise Agent Platform.
    GEMINI_4_FLASH_PREVIEW = "gemini-4-flash-preview"

    # Define model lists centrally to ensure consistency between cache logic and execution logic
    THINKING_MODELS = [
        "gemini-1.5-pro-002", "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25",
        "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", GEMINI_3_7_FLASH, GEMINI_4_FLASH_PREVIEW, "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"
    ]
    GEN3_THINKING_MODELS = [
    "gemini-pro-latest", "gemini-flash-latest", "gemini-3.1-pro-preview",
    "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", GEMINI_3_7_FLASH, GEMINI_4_FLASH_PREVIEW
    ]
    IMAGE_MODELS = [
        "gemini-2.5-flash-image-preview", "gemini-2.5-flash-image",
        "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image-preview", "gemini-3-pro-image", "nano-banana-pro-preview",
    ]
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
            node_id="SSL_GeminiTextPrompt",
            display_name="Expanded Gemini Text/Image",
            category="API/Gemini",
            inputs=[
                cls.GemConfig.Input("config"),
                IO.String.Input("prompt", multiline=True),
                IO.String.Input("system_instruction", default="You are a helpful AI assistant.", multiline=True),
                IO.Combo.Input("model", options=["gemini-1.5-pro-002", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-exp-03-25", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", cls.GEMINI_3_7_FLASH, "gemini-2.5-flash-image-preview", "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image", "nano-banana-pro-preview", "gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"], default="gemini-2.5-flash"),
                IO.Float.Input("temperature", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                IO.Int.Input("top_k", default=40, min=1, max=100, step=1),
                IO.Int.Input("max_output_tokens", default=8192, min=1, max=65536, step=1),
                IO.Boolean.Input("include_images", default=False),
                IO.Combo.Input("aspect_ratio", options=["None", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"], default="None"),
                IO.Combo.Input("bypass_mode", options=["None", "system_instruction", "prompt", "both"], default="None"),
                IO.Int.Input("thinking_budget", default=0, min=-1, max=24576, step=1, tooltip="0 disables thinking mode, -1 will activate it as default dynamic thinking and anything above 0 sets specific budget"),
                IO.Boolean.Input("use_proxy", default=False),
                IO.String.Input("proxy_host", default="127.0.0.1"),
                IO.Int.Input("proxy_port", default=7890, min=1, max=65535),
                IO.Boolean.Input("use_seed", default=True),
                IO.Int.Input("seed", default=0, min=0, max=2147483647),
                IO.Int.Input("timeout", default=30, min=15, max=300, step=15),
                IO.Boolean.Input("include_thoughts", default=False),
                IO.Combo.Input("thinking_level", options=["None", "low", "medium", "high"], default="None", tooltip="Does not work at the same time as 'thinking_budget'. If this is set, then thinking budget is ignored. Gemini 3.7 Flash always uses this control, defaults to medium, and ignores thinking_budget."),
                IO.Combo.Input("media_resolution", options=["unspecified", "low", "medium", "high"], default="unspecified", tooltip="Set input media resolution for image, video and pdf. This changes tokens consumed."),
                IO.String.Input("retry_pattern", default="", optional=True, multiline=False, tooltip="Regex pattern to match in response text. If matched, retry with new seed. Leave empty to disable."),
                IO.Int.Input("max_retries", default=3, min=0, max=10, step=1, tooltip="Maximum number of retry attempts when pattern matches. 0 disables retry."),
                IO.String.Input("timeout_fallback_text", default="", optional=True, multiline=True, tooltip="Text returned when the Gemini request times out. Leave empty to return the standard timeout message."),
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
                IO.Int.Output("final_actual_seed")
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

        print(joined_textfinal)

        return joined_textfinal

    @classmethod
    def save_binary_file(cls, data, mime_type):
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
    def generate_empty_image(cls, width=64, height=64):
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
    def _resolve_gemini_3_7_thinking_level(cls, thinking_level):
        if thinking_level is None or thinking_level == "None":
            return "medium"
        if thinking_level == "minimal":
            print(
                "[WARNING] Gemini 3.7 Flash does not support minimal "
                "thinking; using low instead."
            )
            return "low"
        if thinking_level not in {"low", "medium", "high"}:
            raise ValueError(
                "Gemini 3.7 Flash thinking_level must be low, medium, or high."
            )
        return thinking_level

    @classmethod
    def _compute_fingerprint_and_check_cache(cls, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
                                             include_images, aspect_ratio, bypass_mode, thinking_budget, use_seed, seed,
                                             image_inputs: IO.Autogrow.Type | None = None,
                                             use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, timeout=30,
                                             include_thoughts=False, thinking_level=None, media_resolution=None,
                                             retry_pattern="", max_retries=3):

        # 1. Hashing Images
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

        # 2. Determine Effective Parameters based on Model
        # This ensures we don't cache-miss if an irrelevant parameter changes

        # Defaults
        eff_include_images = False
        eff_aspect_ratio = "None"
        eff_thinking_level = "None"
        eff_thinking_budget = -1
        eff_include_thoughts = False
        eff_temperature = None if model == cls.GEMINI_3_7_FLASH else float(temperature)
        eff_top_p = None if model == cls.GEMINI_3_7_FLASH else float(top_p)
        eff_top_k = None if model == cls.GEMINI_3_7_FLASH else int(top_k)

        # Logic mirroring _build_generate_content_config exactly
        if include_images and model in cls.IMAGE_MODELS:
            eff_include_images = True
            eff_aspect_ratio = str(aspect_ratio)
            # When generating images, thinking params are ignored

        elif model == cls.GEMINI_3_7_FLASH:
            eff_thinking_level = cls._resolve_gemini_3_7_thinking_level(thinking_level)
            eff_include_thoughts = include_thoughts

        elif model in cls.GEN3_THINKING_MODELS and thinking_level is not None and thinking_level != "None":
            eff_thinking_level = thinking_level
            eff_include_thoughts = include_thoughts
            # Budget is ignored in this specific branch

        elif model in cls.THINKING_MODELS:
            eff_thinking_budget = int(thinking_budget)
            eff_include_thoughts = include_thoughts
            # Thinking level is ignored in this branch

        # If none of the above, all effective thinking/image params remain default/ignored

        sanitized_config = dict(config)
        api_key = sanitized_config.get("api_key")
        if api_key:
            sanitized_config["api_key"] = f"sha256:{hashlib.sha256(str(api_key).encode('utf-8')).hexdigest()}"

        fingerprint = (
            str(sanitized_config),
            prompt,
            system_instruction,
            model,
            eff_temperature,
            eff_top_p,
            eff_top_k,
            int(max_output_tokens),
            eff_include_images,    # EFFECTIVE include_images
            eff_aspect_ratio,      # EFFECTIVE aspect_ratio
            str(bypass_mode),
            eff_thinking_budget,   # EFFECTIVE thinking_budget
            use_seed,
            int(seed) if use_seed else 0, # Only use seed in cache if use_seed is True
            image_hashes,
            bool(use_proxy),
            str(proxy_host),
            int(proxy_port),
            int(timeout),
            eff_include_thoughts, # EFFECTIVE include_thoughts
            eff_thinking_level,   # EFFECTIVE thinking_level
            str(media_resolution),
            str(retry_pattern),   # Include retry pattern in fingerprint
            int(max_retries)      # Include max retries in fingerprint
        )

        cached = cls._cache.get(fingerprint)
        if use_seed and cached is not None:
            return fingerprint, cached
        return fingerprint, None

    @classmethod
    def _handle_seed(cls, use_seed, seed):
        actual_seed = None
        if use_seed:
            if seed == 0:
                current_time = int(time.time() * 1000)
                random_component = random.randint(0, 1000000)
                actual_seed = (current_time + random_component) % 2147483647
                print(f"[INFO] Generated random seed: {actual_seed}")
            else:
                actual_seed = seed
                print(f"[INFO] Using specified seed: {actual_seed}")

            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(actual_seed)
        else:
            print("[INFO] Seed not used")

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

    @classmethod
    def _build_generate_content_config(cls, model, temperature, top_p, top_k, max_output_tokens, seed,
                                       include_images, response_modalities, aspect_ratio, padded_system_instruction,
                                       thinking_level, thinking_budget, include_thoughts, media_resolution):
        # Centralized builder for GenerateContentConfig used by different model/feature branches
        safety = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),
        ]

        # Modified: Only trigger image config if include_images is True AND the model is actually an image model
        if include_images and model in cls.IMAGE_MODELS:
            return types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                max_output_tokens=max_output_tokens,
                safety_settings=safety,
                response_modalities=response_modalities,
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                system_instruction=[types.Part.from_text(text=padded_system_instruction)],
            )

        if model == cls.GEMINI_3_7_FLASH:
            return types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                safety_settings=safety,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=include_thoughts,
                    thinking_level=cls._resolve_gemini_3_7_thinking_level(thinking_level),
                ),
                response_modalities=response_modalities,
                system_instruction=[types.Part.from_text(text=padded_system_instruction)],
            )

        G3Pro = ["gemini-3.1-pro-preview"]
        # Modified: Added check for thinking_level != "None"
        if model in G3Pro and thinking_level is not None and thinking_level != "None":
            return types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                safety_settings=safety,
                thinking_config=types.ThinkingConfig(include_thoughts=include_thoughts, thinking_level=thinking_level),
                response_modalities=response_modalities,
                system_instruction=[types.Part.from_text(text=padded_system_instruction)],
            )

        if model in cls.THINKING_MODELS:
            return types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                safety_settings=safety,
                thinking_config=types.ThinkingConfig(include_thoughts=include_thoughts, thinking_budget=thinking_budget),
                response_modalities=response_modalities,
                system_instruction=[types.Part.from_text(text=padded_system_instruction)],
            )

        return types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            safety_settings=safety,
            response_modalities=response_modalities,
            system_instruction=[types.Part.from_text(text=padded_system_instruction)],
        )

    @classmethod
    def execute(cls, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
                include_images, aspect_ratio, bypass_mode, thinking_budget,
                image_inputs: IO.Autogrow.Type | None = None,
                use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, use_seed=False, seed=0, timeout=30,
                include_thoughts=False, thinking_level=None, media_resolution=None,
                retry_pattern="", max_retries=3, timeout_fallback_text="") -> IO.NodeOutput:

        print(f"[INFO] SSL_GeminiTextPrompt execute called, model: {model}")
        fingerprint, cached = cls._compute_fingerprint_and_check_cache(
            config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
            include_images, aspect_ratio, bypass_mode, thinking_budget, use_seed, seed,
            image_inputs,
            use_proxy, proxy_host, proxy_port, timeout,
            include_thoughts, thinking_level, media_resolution,
            retry_pattern, max_retries
        )

        if cached is not None:
            cached_text, cached_image, cached_seed = cached
            print(f"[INFO] Returning cached result for fingerprint {fingerprint}")
            return IO.NodeOutput(cached_text, cached_image, cached_seed)

        print(f"[INFO] Starting generation, model: {model}, temperature: {temperature}")

        padded_prompt = prompt
        padded_system_instruction = system_instruction

        if bypass_mode == "prompt" or bypass_mode == "both":
            padded_prompt = cls._pad_text_with_joiners(prompt)
            print(padded_prompt)
        if bypass_mode == "system_instruction" or bypass_mode == "both":
            padded_system_instruction = cls._pad_text_with_joiners(system_instruction)
            print(padded_system_instruction)

        actual_seed = cls._handle_seed(use_seed, seed)
        input_seed = seed  # Store the original input seed for cache key

        # Check if we have a cached successful gemini seed for this input seed
        # Build a cache key that excludes the seed itself (to match different gemini seeds for same input)
        if use_seed and retry_pattern and max_retries > 0:
            seed_cache_key = (input_seed, fingerprint[:-4])  # Exclude seed, retry_pattern, max_retries from key
            cached_gemini_seed = cls._seed_map_cache.get(seed_cache_key)
            if cached_gemini_seed is not None:
                print(f"[INFO] Using cached successful gemini seed {cached_gemini_seed} for input seed {input_seed}")
                actual_seed = cached_gemini_seed

        text_output = ""
        image_tensor = cls.generate_empty_image()
        proxy_url: str | None = None
        timed_out = False

        try:
            if use_proxy:
                proxy_url = cls._build_proxy_url(proxy_host, proxy_port)

            try:
                vertexai_express = config.get("vertexai_express", False)
                use_vertexai_env = config.get("use_vertexai_env", False)
                api_version = config.get("api_version")
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
                        return IO.NodeOutput(f"Missing environment variable: {e}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)

                    except AssertionError as e:
                        print(f"Error: {e}")
                        return IO.NodeOutput(f"Invalid environment variable: {e}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)

                    except ValueError as e:
                        print(f"Error: {e}")
                        return IO.NodeOutput(f"Invalid Enterprise/Vertex AI configuration: {e}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)

                elif vertexai_express:
                    api_key = config.get("api_key")
                    if not api_key:
                        return IO.NodeOutput(
                            "Invalid Enterprise express configuration: API key is required",
                            cls.generate_empty_image(),
                            actual_seed if actual_seed is not None else 0,
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
                return IO.NodeOutput(f"Gemini client initialization failed: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)

            # Prepare contents (images + prompt)
            image_frames, image_batch_counts = cls._ordered_image_frames(image_inputs)

            if image_frames:
                try:
                    img_parts = []
                    for _, _, image_frame in image_frames:
                        img_array = image_frame.cpu().numpy()
                        img_array = (img_array * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array)
                        img_byte_arr = BytesIO()
                        pil_img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        if model in cls.MEDIA_RES_MODELS and media_resolution is not None and media_resolution != "unspecified":
                            img_part = types.Part.from_bytes(
                                data=img_bytes,
                                mime_type="image/png",
                                media_resolution=f"MEDIA_RESOLUTION_{media_resolution.upper()}",
                            )
                        else:
                            img_part = types.Part.from_bytes(
                                data=img_bytes,
                                mime_type="image/png",
                            )
                        img_parts.append(img_part)
                    batch_summary = ", ".join(
                        f"{socket_name}={batch_size}"
                        for socket_name, batch_size in image_batch_counts
                    )
                    print(
                        f"[INFO] Prepared Gemini image inputs: {batch_summary}, "
                        f"total={len(img_parts)}"
                    )
                    contents = [
                        types.UserContent(
                            parts=img_parts + [types.Part.from_text(text=padded_prompt)]
                        )
                    ]
                except Exception as e:
                    print(f"[ERROR] Error processing input image: {str(e)}")
                    return IO.NodeOutput(f"Error processing input image: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)
            else:
                contents = padded_prompt

            # Logic Update: Only request IMAGE modality if include_images is TRUE AND it's the correct model
            # Otherwise we stick to TEXT modality
            response_modalities = ["IMAGE", "TEXT"] if (include_images and model in cls.IMAGE_MODELS) else ["TEXT"]

            generate_content_config = cls._build_generate_content_config(
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                seed=seed,
                include_images=include_images,
                response_modalities=response_modalities,
                aspect_ratio=aspect_ratio,
                padded_system_instruction=padded_system_instruction,
                thinking_level=thinking_level,
                thinking_budget=thinking_budget,
                include_thoughts=include_thoughts,
                media_resolution=media_resolution,
            )

            if use_seed and actual_seed is not None:
                try:
                    generate_content_config.seed = actual_seed
                except Exception:
                    pass

            # API call in background thread
            start_time = time.time()
            result_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()

            def api_call():
                last_api_exception = None
                api_response = None
                max_retries = 1
                for attempt in range(max_retries):
                    try:
                        if use_seed and actual_seed is not None:
                            try:
                                generate_content_config.seed = actual_seed + attempt
                            except Exception:
                                pass
                        response = client.models.generate_content(model=model, contents=contents, config=generate_content_config)
                        if not (response.candidates and getattr(response.candidates[0].content, 'parts', None)):
                            finish_reason = "UNKNOWN"
                            if response.candidates:
                                fr = getattr(response.candidates[0], 'finish_reason', None)
                                if fr is not None and hasattr(fr, 'name'):
                                    finish_reason = fr.name
                            raise ValueError(f"Response was empty or blocked (Finish Reason: {finish_reason}).")
                        api_response = response
                        break
                    except Exception as e:
                        last_api_exception = e
                if api_response is None:
                    result_queue.put(("error", last_api_exception))
                    return
                try:
                    current_text_output = ""
                    current_image_tensor = None
                    parts = None
                    if api_response.candidates:
                        parts = getattr(api_response.candidates[0].content, 'parts', None)
                    if parts:
                        for part in parts:
                            if hasattr(part, 'text') and part.text is not None:
                                current_text_output += part.text
                            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                                try:
                                    inline_data = part.inline_data
                                    mime_type = inline_data.mime_type
                                    data = inline_data.data
                                    image_path = cls.save_binary_file(data, mime_type)
                                    img = Image.open(image_path)
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    img_array = np.array(img).astype(np.float32) / 255.0
                                    current_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                                except Exception:
                                    if current_image_tensor is None:
                                        current_image_tensor = cls.generate_empty_image()
                    if current_image_tensor is None:
                        current_image_tensor = cls.generate_empty_image()
                    result_queue.put(("success", (current_text_output, current_image_tensor)))
                except Exception as e_proc:
                    result_queue.put(("error", e_proc))

            api_thread = threading.Thread(target=api_call)
            api_thread.daemon = True
            api_thread.start()

            # Retry loop for pattern matching
            retry_attempt = 0
            retry_needed = True
            while retry_needed:
                retry_needed = False  # Will be set to True if pattern matches

                try:
                    status, result = result_queue.get(timeout=timeout)
                    elapsed_time = time.time() - start_time
                    if status == "success":
                        text_output, image_tensor = result

                        # Check if retry pattern matches
                        if retry_pattern and max_retries > 0 and retry_attempt < max_retries:
                            print(f"[DEBUG] Checking retry pattern '{retry_pattern}' against response (first 200 chars): {text_output[:200]}")
                            try:
                                compiled_pattern = re.compile(retry_pattern, re.IGNORECASE)
                                if compiled_pattern.search(text_output):
                                    retry_attempt += 1
                                    print(f"[INFO] Retry pattern matched in response. Retry attempt {retry_attempt}/{max_retries}")

                                    # Generate new random gemini seed for retry
                                    current_time = int(time.time() * 1000)
                                    random_component = random.randint(0, 1000000)
                                    actual_seed = (current_time + random_component) % 2147483647
                                    print(f"[INFO] Retrying with new gemini seed: {actual_seed}")

                                    # Update config seed and retry
                                    if use_seed:
                                        try:
                                            generate_content_config.seed = actual_seed
                                        except Exception:
                                            pass

                                    # Create new queue and thread for retry
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

                        # Check if retry pattern matches error/finish reason
                        if retry_pattern and max_retries > 0 and retry_attempt < max_retries:
                            print(f"[DEBUG] Checking retry pattern '{retry_pattern}' against error: {text_output}")
                            try:
                                compiled_pattern = re.compile(retry_pattern, re.IGNORECASE)
                                if compiled_pattern.search(text_output):
                                    retry_attempt += 1
                                    print(f"[INFO] Retry pattern matched in error. Retry attempt {retry_attempt}/{max_retries}")

                                    # Generate new random gemini seed for retry
                                    current_time = int(time.time() * 1000)
                                    random_component = random.randint(0, 1000000)
                                    actual_seed = (current_time + random_component) % 2147483647
                                    print(f"[INFO] Retrying with new gemini seed: {actual_seed}")

                                    # Update config seed and retry
                                    if use_seed:
                                        try:
                                            generate_content_config.seed = actual_seed
                                        except Exception:
                                            pass

                                    # Create new queue and thread for retry
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
                    timed_out = True
                    text_output = timeout_fallback_text or f"Gemini API request/processing timed out, waited {timeout} seconds."

        except Exception as e:
            print(f"[ERROR] Unhandled error in generate method: {str(e)}")
            text_output = f"Unhandled error: {str(e)}"
            if image_tensor is None:
                image_tensor = cls.generate_empty_image()

        final_actual_seed = actual_seed if actual_seed is not None else 0

        is_success = not timed_out and not text_output.startswith("API call/processing error:") and not text_output.startswith("Gemini API request/processing timed out")

        # Cache the result
        if use_seed and is_success:
            try:
                cls._cache[fingerprint] = (text_output, image_tensor, final_actual_seed)
            except Exception:
                pass

            # Cache the successful gemini seed for this input seed (if retry was enabled)
            if retry_pattern and max_retries > 0:
                seed_cache_key = (input_seed, fingerprint[:-4])
                cls._seed_map_cache[seed_cache_key] = final_actual_seed
                print(f"[INFO] Cached successful gemini seed {final_actual_seed} for input seed {input_seed}")


        return IO.NodeOutput(text_output, image_tensor, final_actual_seed)

# V3 uses ComfyExtension entrypoint in __init__.py to expose nodes

import os
import json
import uuid
import re
import torch
import numpy as np
import cv2
from PIL import Image
import io as stdlib_io
import folder_paths  # type: ignore[reportMissingImports]
from comfy_api.latest import ComfyExtension, io, ui  # type: ignore[reportMissingImports]
from google import genai
from google.genai import types
import time
import traceback
import threading
import queue
import sys
import importlib
import subprocess
import random
from typing import Any, Tuple

def check_and_install_dependencies():
    required_packages = {
        'requests': 'requests',
        'pysocks': 'PySocks',
    }

    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            print(f"[WARNING] {package_name} not found, trying to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"[INFO] Successfully installed {package_name}")
            except Exception as e:
                print(f"[ERROR] Failed to install {package_name}: {str(e)}")

try:
    check_and_install_dependencies()
except Exception as e:
    print(f"[WARNING] Error checking dependencies: {str(e)}")

class GetKeyAPI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetKeyAPI",
            display_name="Get API Key from JSON",
            category="utils/api_keys",
            inputs=[
                io.String.Input("json_path", default="./input/apikeys.json", multiline=False, tooltip="Path to a .json file with simple top level structure with name as key and api-key as value. See example in custom node folder."),
                io.Combo.Input("key_id_method", options=["custom", "random_rotate", "increment_rotate"], default="custom", tooltip="custom sets api-key to the api-key with the name set in the key_id widget. random_rotate randomly switches between keys if multiple in the .json and increment_rotate does it in order from first to last, then repeats."),
                io.Int.Input("rotation_interval", default=0, min=0, tooltip="how many steps to jump when doing rotate."),
                io.String.Input("key_id", default="placeholder", multiline=False, optional=True, tooltip="Put name of key in the .json here if using custom in key_id_method."),
            ],
            outputs=[
                io.String.Output("API_KEY")
            ]
        )

    @classmethod
    def execute(cls, json_path: str, key_id_method: str, rotation_interval: int, key_id: str | None = "placeholder") -> io.NodeOutput:
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
        return io.NodeOutput(selected_key_value)



class SSL_GeminiAPIKeyConfig(io.ComfyNode):
    GemConfig = io.Custom("GEMINI_CONFIG")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SSL_GeminiAPIKeyConfig",
            display_name="Configure Gemini API Key",
            category="API/Gemini",
            inputs=[
                io.String.Input("api_key", multiline=False),
                io.Combo.Input("api_version", options=["v1", "v1alpha", "v1beta", "v2beta"], default="v1alpha"),
                io.Boolean.Input("vertexai", default=False),
                io.String.Input("vertexai_project", default="placeholder", optional=True),
                io.String.Input("vertexai_location", default="placeholder", optional=True),
            ],
            outputs=[
                cls.GemConfig.Output("config")
            ]
        )

    @classmethod
    def execute(cls, api_key: str, api_version: str, vertexai: bool, vertexai_project: str | None = "placeholder", vertexai_location: str | None = "placeholder") -> io.NodeOutput:
        config = {"api_key": api_key, "api_version": api_version, "vertexai": vertexai, "vertexai_project": vertexai_project, "vertexai_location": vertexai_location}
        return io.NodeOutput(config)



class SSL_GeminiTextPrompt(io.ComfyNode):
    GemConfig = io.Custom("GEMINI_CONFIG")
    _cache: dict = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SSL_GeminiTextPrompt",
            display_name="Expanded Gemini Text/Image",
            category="API/Gemini",
            inputs=[
                cls.GemConfig.Input("config"),
                io.String.Input("prompt", multiline=True),
                io.String.Input("system_instruction", default="You are a helpful AI assistant.", multiline=True),
                io.Combo.Input("model", options=["learnlm-2.0-flash-experimental", "gemini-exp-1206", "gemini-2.0-flash", "gemini-2.0-flash-lite-001", "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219", "gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025", "gemini-2.5-flash-lite-preview-06-17", "gemini-3-pro-preview", "gemini-2.5-flash-image-preview"], default="gemini-2.0-flash"),
                io.Float.Input("temperature", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Int.Input("top_k", default=40, min=1, max=100, step=1),
                io.Int.Input("max_output_tokens", default=8192, min=1, max=65536, step=1),
                io.Boolean.Input("include_images", default=False),
                io.Combo.Input("aspect_ratio", options=["None", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"], default="None"),
                io.Combo.Input("bypass_mode", options=["None", "system_instruction", "prompt", "both"], default="None"),
                io.Int.Input("thinking_budget", default=0, min=-1, max=24576, step=1, tooltip="0 disables thinking mode, -1 will activate it as default dynamic thinking and anything above 0 sets specific budget"),
                io.Image.Input("input_image", optional=True),
                io.Image.Input("input_image_2", optional=True),
                io.Boolean.Input("use_proxy", default=False),
                io.String.Input("proxy_host", default="127.0.0.1"),
                io.Int.Input("proxy_port", default=7890, min=1, max=65535),
                io.Boolean.Input("use_seed", default=True),
                io.Int.Input("seed", default=0, min=0, max=2147483647),
                io.Int.Input("timeout", default=30, min=15, max=300, step=15),
                io.Boolean.Input("include_thoughts", default=False),
                io.Combo.Input("thinking_level", options=["None", "low", "medium", "high"], default="None", tooltip="Does not work at the same time as 'thinking_budget'. if this is set, then thinking budget is ignored."),
                io.Combo.Input("media_resolution", options=["unspecified", "low", "medium", "high"], default="unspecified", tooltip="Set input media resolution for image, video and pdf. This changes tokens consumed."),
            ],
            outputs=[
                io.String.Output("text"),
                io.Image.Output("image"),
                io.Int.Output("final_actual_seed")
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
    def _compute_fingerprint_and_check_cache(cls, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
                                             include_images, aspect_ratio, bypass_mode, thinking_budget, use_seed, seed,
                                             input_image=None, input_image_2=None):
        fingerprint = (
            str(config),
            prompt,
            system_instruction,
            model,
            float(temperature),
            float(top_p),
            int(top_k),
            int(max_output_tokens),
            bool(include_images),
            str(aspect_ratio),
            str(bypass_mode),
            int(thinking_budget),
            use_seed,
            int(seed)
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
    def _setup_proxy_env(cls, proxy_host, proxy_port):
        if not proxy_host.startswith(('http://', 'https://')):
            proxy_url = f"http://{proxy_host}:{proxy_port}"
        else:
            proxy_url = f"{proxy_host}:{proxy_port}"

        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['REQUESTS_CA_BUNDLE'] = ''

        print(f"[INFO] Proxy enabled: {proxy_url}")
        return proxy_url

    @classmethod
    def _build_generate_content_config(cls, model, temperature, top_p, top_k, max_output_tokens, seed,
                                       include_images, response_modalities, aspect_ratio, padded_system_instruction,
                                       thinking_level, thinking_budget, include_thoughts, media_resolution):
        # Centralized builder for GenerateContentConfig used by different model/feature branches
        safety = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),  # type: ignore
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="BLOCK_NONE"),  # type: ignore
        ]

        if include_images:
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

        if model == "gemini-3-pro-preview" and thinking_level is not None:
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

        if model in [
            "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219",
            "gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025",
            "gemini-2.5-flash-lite-preview-06-17", "gemini-3-pro-preview"]:
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
                include_images, aspect_ratio, bypass_mode, thinking_budget, input_image=None, input_image_2=None,
                use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, use_seed=False, seed=0, timeout=30,
                include_thoughts=False, thinking_level=None, media_resolution=None) -> io.NodeOutput:

        def tensor_equal(t1, t2):
            if t1 is None and t2 is None:
                return True
            if t1 is not None and t2 is not None:
                try:
                    return torch.equal(t1, t2)
                except Exception:
                    return False
            return False

        fingerprint, cached = cls._compute_fingerprint_and_check_cache(
            config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens,
            include_images, aspect_ratio, bypass_mode, thinking_budget, use_seed, seed, input_image, input_image_2
        )

        if cached is not None:
            cached_text, cached_image, cached_seed = cached
            if tensor_equal(cached_image, input_image) or tensor_equal(cached_image, input_image_2):
                print(f"[INFO] Returning cached result for fingerprint {fingerprint}")
                return io.NodeOutput(cached_text, cached_image, cached_seed)

        # --- keep most of the original implementation but converted to classmethod usage ---
        original_http_proxy = os.environ.get('HTTP_PROXY')
        original_https_proxy = os.environ.get('HTTPS_PROXY')
        original_http_proxy_lower = os.environ.get('http_proxy')
        original_https_proxy_lower = os.environ.get('https_proxy')
        thinking_models = ["gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219", "gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025", "gemini-2.5-flash-lite-preview-06-17", "gemini-3-pro-preview"]
        media_res_models = ["gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp-1219", "gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025", "gemini-2.5-flash-lite-preview-06-17", "gemini-3-pro-preview"]
        thinking_levels = ["low", "medium", "high"]

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

        # Flatten and simplify nested try/except blocks to ensure correct pairing
        original_http_proxy = os.environ.get('HTTP_PROXY')
        original_https_proxy = os.environ.get('HTTPS_PROXY')
        original_http_proxy_lower = os.environ.get('http_proxy')
        original_https_proxy_lower = os.environ.get('https_proxy')

        text_output = ""
        image_tensor = cls.generate_empty_image()
        network_ok = True
        proxy_url: str | None = None

        try:
            if use_proxy:
                proxy_url = cls._setup_proxy_env(proxy_host, proxy_port)

            # Initialize Gemini client
            client_options = {}
            if use_proxy:
                try:
                    import google.api_core.http_client  # type: ignore[import]
                    import google.auth.transport.requests  # type: ignore[import]
                    import requests
                    from requests.adapters import HTTPAdapter

                    class ProxyAdapter(HTTPAdapter):
                        def __init__(self, proxy_url, **kwargs):
                            self.proxy_url = proxy_url
                            super().__init__(**kwargs)

                        def add_headers(self, request, **kwargs):
                            super().add_headers(request, **kwargs)

                    session = requests.Session()
                    proxies = {"http": str(proxy_url), "https": str(proxy_url)}
                    session.proxies.update(proxies)
                    adapter = ProxyAdapter(proxy_url, max_retries=1)
                    session.mount('http://', adapter)
                    session.mount('https://', adapter)
                    session.verify = False
                    http_client = google.api_core.http_client.RequestsHttpClient(session=session)
                    client_options["http_client"] = http_client
                except Exception:
                    # best-effort proxy HTTP client setup; fall back if imports fail
                    pass

            try:
                vertexai = config.get("vertexai", False)
                if vertexai:
                    project = config.get("vertexai_project")
                    location = config.get("vertexai_location")
                    client = genai.Client(vertexai=vertexai, project=project, location=location, http_options=types.HttpOptions(api_version=config.get("api_version")), **client_options)  # type: ignore
                else:
                    client = genai.Client(api_key=config.get("api_key"), http_options=types.HttpOptions(api_version=config.get("api_version")), **client_options)  # type: ignore
            except Exception as e:
                print(f"[ERROR] Gemini client initialization failed: {str(e)}")
                return io.NodeOutput(f"Gemini client initialization failed: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)

            # Network test (best-effort)
            try:
                import socket
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5)
                test_host = "generativelanguage.googleapis.com"
                if use_proxy:
                    try:
                        import socks  # type: ignore[import]
                        test_socket = socks.socksocket()
                        test_socket.set_proxy(socks.HTTP, proxy_host, proxy_port)
                        test_socket.settimeout(5)
                    except Exception:
                        pass
                test_socket.connect((test_host, 443))
                test_socket.close()
                network_ok = True
            except Exception:
                network_ok = False

            # Prepare contents (images + prompt)
            images_to_process = []
            if input_image is not None:
                images_to_process.append(input_image)
            if input_image_2 is not None:
                images_to_process.append(input_image_2)

            if images_to_process:
                try:
                    img_parts = []
                    for img in images_to_process:
                        img_array = img[0].cpu().numpy()
                        img_array = (img_array * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array)
                        img_byte_arr = stdlib_io.BytesIO()
                        pil_img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        if model in media_res_models and media_resolution is not None:
                            img_part = {"inline_data": {"mime_type": "image/png", "data": img_bytes}, "media_resolution": {"level": f"MEDIA_RESOLUTION_{media_resolution.upper()}"}}
                        else:
                            img_part = {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
                        img_parts.append(img_part)
                    contents = img_parts + [{"text": padded_prompt}]
                except Exception as e:
                    print(f"[ERROR] Error processing input image: {str(e)}")
                    return io.NodeOutput(f"Error processing input image: {str(e)}", cls.generate_empty_image(), actual_seed if actual_seed is not None else 0)
            else:
                contents = padded_prompt

            response_modalities = ["IMAGE", "TEXT"] if include_images else ["TEXT"]

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

            try:
                status, result = result_queue.get(timeout=timeout)
                elapsed_time = time.time() - start_time
                if status == "success":
                    text_output, image_tensor = result
                else:
                    error_exception = result
                    text_output = f"API call/processing error: {str(error_exception)}"
                    if any(term in str(error_exception).lower() for term in ["timeout", "connection", "network", "socket", "连接", "网络"]) and not network_ok and not use_proxy:
                        text_output += " Network connection test failed, consider enabling proxy."
            except queue.Empty:
                text_output = f"Gemini API request/processing timed out, waited {timeout} seconds."
                if not network_ok:
                    if use_proxy:
                        text_output += f" Network connection test failed, the current proxy ({proxy_host}:{proxy_port}) may be invalid, please check proxy settings."
                    else:
                        text_output += " Network connection test failed, consider enabling proxy."
                else:
                    text_output += " API request timed out despite network being OK."

        except Exception as e:
            print(f"[ERROR] Unhandled error in generate method: {str(e)}")
            text_output = f"Unhandled error: {str(e)}"
            if image_tensor is None:
                image_tensor = cls.generate_empty_image()

        final_actual_seed = actual_seed if actual_seed is not None else 0
        if use_seed:
            try:
                cls._cache[fingerprint] = (text_output, image_tensor, final_actual_seed)
            except Exception:
                pass

        try:
            return io.NodeOutput(text_output, image_tensor, final_actual_seed)
        finally:
            if original_http_proxy:
                os.environ['HTTP_PROXY'] = original_http_proxy
            else:
                if 'HTTP_PROXY' in os.environ:
                    os.environ.pop('HTTP_PROXY')

            if original_https_proxy:
                os.environ['HTTPS_PROXY'] = original_https_proxy
            else:
                if 'HTTPS_PROXY' in os.environ:
                    os.environ.pop('HTTPS_PROXY')

            if original_http_proxy_lower:
                os.environ['http_proxy'] = original_http_proxy_lower
            else:
                if 'http_proxy' in os.environ:
                    os.environ.pop('http_proxy')

            if original_https_proxy_lower:
                os.environ['https_proxy'] = original_https_proxy_lower
            else:
                if 'https_proxy' in os.environ:
                    os.environ.pop('https_proxy')

            if 'REQUESTS_CA_BUNDLE' in os.environ:
                os.environ.pop('REQUESTS_CA_BUNDLE')

            try:
                import requests
                if hasattr(requests, 'Session'):
                    clean_session = requests.Session()
                    if hasattr(requests, 'session'):
                        requests.session = lambda: clean_session
            except:
                pass

# V3 uses ComfyExtension entrypoint in __init__.py to expose nodes
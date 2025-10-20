import os
import json
import uuid
import re
import torch
import numpy as np
from PIL import Image
import io
import folder_paths
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
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

class GetKeyAPI(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "json_path": (IO.STRING, {"default": "./input/apikeys.json", "multiline": False, "tooltip": "Path to a .json file with simple top level structure with name as key and api-key as value. See example in custom node folder."}),
                "key_id_method": (["custom", "random_rotate", "increment_rotate"], {"default": "custom", "tooltip": "custom sets api-key to the api-key with the name set in the key_id widget. random_rotate randomly switches between keys if multiple in the .json and increment_rotate does it in order from first to last, then repeats."}),
                "rotation_interval": (IO.INT, {"default": 0, "min": 0, "tooltip": "how many steps to jump when doing rotate."}),
            },
            "optional": {
                "key_id": (IO.STRING, {"default": "placeholder", "multiline": False, "tooltip": "Put name of key in the .json here if using custom in key_id_method."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("API_KEY",)
    FUNCTION = "getapikey"
    CATEGORY = "utils/api_keys"

    def getapikey(self, json_path, key_id_method, rotation_interval, key_id="placeholder"):
        """
        Loads API keys from a JSON file (top-level dictionary)
        and selects one based on the specified method.

        Args:
            json_path (str): Path to the JSON file. Expected format:
                             {"key_id_1": "api_key_value_1", "key_id_2": "api_key_value_2", ...}
            key_id_method (str): Method to select the key ('custom', 'random_rotate', 'increment_rotate').
            rotation_interval (int): Used as index for 'increment_rotate'.
            key_id (str, optional): ID (key name) of the key to select if key_id_method is 'custom'. Defaults to "placeholder".

        Returns:
            str: The selected API key string.
            Raises: ValueError or RuntimeError if unable to find or select a key.
        """
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
        return (selected_key_value,)



class SSL_GeminiAPIKeyConfig(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "api_key": (IO.STRING, {"multiline": False}),
                "api_version": (["v1", "v1alpha", "v1beta", "v2beta"], {"default": "v1alpha"}),
                "vertexai": (IO.BOOLEAN, {"default": False}),
            },
            "optional": {
                "vertexai_project": (IO.STRING, {"default": "placeholder", "multiline": False}),
                "vertexai_location": (IO.STRING, {"default": "placeholder", "multiline": False}),
            }
        }

    RETURN_TYPES = ("GEMINI_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "API/Gemini"

    def configure(self, api_key, api_version, vertexai, vertexai_project, vertexai_location):
        config = {"api_key": api_key, "api_version": api_version, "vertexai": vertexai, "vertexai_project": vertexai_project, "vertexai_location": vertexai_location}
        return (config,)


class SSL_GeminiTextPrompt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "config": ("GEMINI_CONFIG",),
                "prompt": (IO.STRING, {"multiline": True}),
                "system_instruction": (IO.STRING, {"default": "You are a helpful AI assistant.", "multiline": True}),
                "model": (["gemini-1.0-pro", "gemini-exp-1206", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite-001", "gemini-2.0-flash-exp", "gemini-2.0-pro", "gemini-2.0-flash-live", "gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-image-preview"], {"default": "gemini-2.0-flash"}),
                "temperature": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": (IO.FLOAT, {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": (IO.INT, {"default": 40, "min": 1, "max": 100, "step": 1}),
                "max_output_tokens": (IO.INT, {"default": 8192, "min": 1, "max": 65536, "step": 1}),
                "include_images": (IO.BOOLEAN, {"default": True}),
                "aspect_ratio": (["None", "1:1", "9:16", "16:9", "3:4", "4:3", "3:2", "2:3", "5:4", "4:5", "21:9"], {"default": "None"}),
                "bypass_mode": (["None", "system_instruction", "prompt", "both"], {"default": "None"}),
                "thinking_budget": (IO.INT, {"default": 0, "min": -1, "max": 24576, "step": 1, "tooltip": "0 disables thinking mode, -1 will activate it as default dynamic thinking and anything above 0 sets specific budget"}),
            },
            "optional": {
                "input_image": (IO.IMAGE,),
                "input_image_2": (IO.IMAGE,),
                "use_proxy": (IO.BOOLEAN, {"default": False}),
                "proxy_host": (IO.STRING, {"default": "127.0.0.1"}),
                "proxy_port": (IO.INT, {"default": 7890, "min": 1, "max": 65535}),
                "use_seed": (IO.BOOLEAN, {"default": True}),
                "seed": (IO.INT, {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("text", "image", "seed")
    FUNCTION = "generate"
    CATEGORY = "API/Gemini"

    def _pad_text_with_joiners(self, text: str) -> str:
        if not text:
            return ""


        # Build the pattern using an f-string to correctly embed the unicode char.
        # This is not a raw string.
        patternperiod = r"\."
        patternspace = r"\s"
        patterncomma = r","
        patterndash = r"\-"
        patternsingq = r"\'"
        patterndoubq = r"\""
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

    def save_binary_file(self, data, mime_type):
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

    def generate_empty_image(self, width=64, height=64):
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0)
        return tensor

    def generate(self, config, prompt, system_instruction, model, temperature, top_p, top_k, max_output_tokens, include_images,
                aspect_ratio, bypass_mode, thinking_budget, input_image=None, input_image_2=None,
                use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890, use_seed=True, seed=0):
        original_http_proxy = os.environ.get('HTTP_PROXY')
        original_https_proxy = os.environ.get('HTTPS_PROXY')
        original_http_proxy_lower = os.environ.get('http_proxy')
        original_https_proxy_lower = os.environ.get('https_proxy')
        thinking_models = ["gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash", "gemini-2.5-flash-preview-04-17", "gemini-2.5-flash-lite-preview-06-17", "gemini-beta-3.0-pro"]

        print(f"[INFO] Starting generation, model: {model}, temperature: {temperature}")

        # Pad text based on bypass_mode
        padded_prompt = prompt
        padded_system_instruction = system_instruction

        if bypass_mode == "prompt" or bypass_mode == "both":
            padded_prompt = self._pad_text_with_joiners(prompt)
            print(padded_prompt)
        if bypass_mode == "system_instruction" or bypass_mode == "both":
            padded_system_instruction = self._pad_text_with_joiners(system_instruction)
            print(padded_system_instruction)

        actual_seed = None
        if use_seed == True:
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

        try:
            if use_proxy == True:
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

            print(f"[INFO] Initializing Gemini client")
            try:
                client_options = {}

                if use_proxy == True:
                    try:
                        import google.api_core.http_client
                        import google.auth.transport.requests

                        try:
                            import requests
                            from requests.adapters import HTTPAdapter

                            class ProxyAdapter(HTTPAdapter):
                                def __init__(self, proxy_url, **kwargs):
                                    self.proxy_url = proxy_url
                                    super().__init__(**kwargs)

                                def add_headers(self, request, **kwargs):
                                    super().add_headers(request, **kwargs)

                            session = requests.Session()
                            proxies = {
                                "http": proxy_url,
                                "https": proxy_url
                            }
                            session.proxies.update(proxies)

                            adapter = ProxyAdapter(proxy_url, max_retries=1)
                            session.mount('http://', adapter)
                            session.mount('https://', adapter)

                            session.verify = False

                            http_client = google.api_core.http_client.RequestsHttpClient(session=session)
                            client_options["http_client"] = http_client

                            print(f"[INFO] Used requests library to set proxy for HTTP client")
                        except Exception as proxy_error:
                            print(f"[WARNING] Failed to set HTTP client proxy: {str(proxy_error)}")
                    except ImportError as e:
                        print(f"[WARNING] Failed to import Google API HTTP client library: {str(e)}")
                vertexai = config["vertexai"]
                if vertexai == True:
                    project = config["vertexai_project"]
                    location = config["vertexai_location"]
                    client = genai.Client(vertexai=vertexai, project=project, location=location, http_options=types.HttpOptions(api_version=config["api_version"]), **client_options)
                else:
                    client = genai.Client(api_key=config["api_key"], http_options=types.HttpOptions(api_version=config["api_version"]), **client_options)
                print(f"[INFO] Gemini client initialized successfully")
            except Exception as e:
                print(f"[ERROR] Gemini client initialization failed: {str(e)}")
                return (f"Gemini client initialization failed: {str(e)}", self.generate_empty_image(), actual_seed if actual_seed is not None else 0)

            try:
                import socket
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5)

                test_host = "generativelanguage.googleapis.com"

                if use_proxy == True:
                    try:
                        import socks
                        test_socket = socks.socksocket()
                        test_socket.set_proxy(socks.HTTP, proxy_host, proxy_port)
                        test_socket.settimeout(5)
                    except ImportError:
                        print(f"[WARNING] PySocks library not installed, cannot test connection via proxy")

                test_socket.connect((test_host, 443))
                test_socket.close()
                network_ok = True
            except Exception as e:
                network_ok = False
                print(f"[WARNING] Network connection test failed: {str(e)}")

                if use_proxy == True:
                    try:
                        import subprocess
                        cmd = f"curl -x {proxy_url} -s -o /dev/null -w '%{{http_code}}' https://{test_host}"
                        try:
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                            if result.returncode == 0 and result.stdout.strip() in ['200', '301', '302']:
                                network_ok = True
                            else:
                                print(f"[WARNING] curl proxy test failed, status code: {result.stdout.strip() if result.stdout else 'N/A'}")
                        except Exception as curl_error:
                            print(f"[WARNING] curl test failed: {str(curl_error)}")
                    except ImportError:
                        pass

            contents = []

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

                        img_byte_arr = io.BytesIO()
                        pil_img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()

                        img_part = {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
                        img_parts.append(img_part)

                    contents = img_parts + [{"text": padded_prompt}]
                except Exception as e:
                    print(f"[ERROR] Error processing input image: {str(e)}")
                    return (f"Error processing input image: {str(e)}", self.generate_empty_image(), actual_seed if actual_seed is not None else 0)
            else:
                contents = padded_prompt

            if include_images == True:
                response_modalities = ["IMAGE", "TEXT"]
            else:
                response_modalities = ["TEXT"]

            print(padded_prompt)
            print(padded_system_instruction)
            if include_images == True:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    max_output_tokens=max_output_tokens,
                    safety_settings=[types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                        threshold="BLOCK_NONE"
                    )],
                    response_modalities=response_modalities,
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                    system_instruction=[types.Part.from_text(text=padded_system_instruction)],
                )
            elif model in thinking_models:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    max_output_tokens=max_output_tokens,
                    safety_settings=[types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                        threshold="BLOCK_NONE"
                    )],
                    thinking_config = types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                    ),
                    response_modalities=response_modalities,
                    system_instruction=[types.Part.from_text(text=padded_system_instruction)],
                )
            else:
                generate_content_config = types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    max_output_tokens=max_output_tokens,
                    safety_settings=[types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    ),types.SafetySetting(
                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                        threshold="BLOCK_NONE"
                    )],
                    response_modalities=response_modalities,
                    system_instruction=[types.Part.from_text(text=padded_system_instruction)],
                )

            # if use_seed == True and actual_seed is not None:
            #     try:
            #         generate_content_config.seed = actual_seed
            #     except Exception as seed_error:
            #         print(f"[WARNING] Failed to set seed for API request: {str(seed_error)}")

            try:
                print(f"[INFO] Sending API request to Gemini")

                start_time = time.time()
                timeout = 30

                def api_call():
                    last_api_exception = None
                    max_retries = 1
                    api_response = None

                    for attempt in range(max_retries):
                        try:
                            print(f"[INFO] API call attempt {attempt + 1}/{max_retries}")

                            # if use_seed and actual_seed is not None:
                            #     current_seed_for_api_call = actual_seed + attempt
                            #     try:
                            #         generate_content_config.seed = current_seed_for_api_call
                            #         if attempt > 0:
                            #             print(f"[INFO] Retrying with incremented seed: {current_seed_for_api_call}")
                            #     except AttributeError:
                            #         print(f"[WARNING] Could not set 'seed' attribute on generate_content_config.")
                            #     except Exception as e_set_seed:
                            #         print(f"[WARNING] Error setting seed {current_seed_for_api_call} for attempt {attempt + 1}: {str(e_set_seed)}")

                            response = client.models.generate_content(
                                model=model,
                                contents=contents,
                                config=generate_content_config,
                            )
                            print(f"[INFO] Received API response for attempt {attempt + 1}")

                            # --- START OF CORRECTED CODE ---
                            # This is the robust validation check. It verifies that the necessary parts of the
                            # response exist and are not None before proceeding.
                            if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
                                finish_reason = "UNKNOWN"
                                if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                                    # Get the reason if available (e.g., SAFETY, RECITATION, etc.)
                                    finish_reason = response.candidates[0].finish_reason.name

                                # Raise a specific error that will be caught by the except block below,
                                # forcing a retry. This now correctly handles blocked/empty responses.
                                raise ValueError(f"Response was empty or blocked (Finish Reason: {finish_reason}).")
                            # --- END OF CORRECTED CODE ---

                            api_response = response
                            break  # Success! Exit the retry loop.

                        except Exception as e:
                            last_api_exception = e
                            print(f"[ERROR] API call attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                            if attempt < max_retries - 1:
                                wait_time = 2**attempt
                                print(f"[INFO] Waiting {wait_time}s before next attempt...")
                                time.sleep(wait_time)

                    if api_response is None:
                        print(f"[ERROR] All {max_retries} API call attempts failed. Last error: {str(last_api_exception)}")
                        result_queue.put(("error", last_api_exception))
                        return

                    try:
                        current_text_output = ""
                        current_image_tensor = None

                        for part in api_response.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text is not None:
                                current_text_output += part.text

                            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                                try:
                                    inline_data = part.inline_data
                                    mime_type = inline_data.mime_type
                                    data = inline_data.data

                                    image_path = self.save_binary_file(data, mime_type)
                                    img = Image.open(image_path)

                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')

                                    img_array = np.array(img).astype(np.float32) / 255.0
                                    current_image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                                except Exception as img_e:
                                    print(f"[ERROR] Image processing error: {str(img_e)}")
                                    current_text_output += f"\nImage processing error: {str(img_e)}"
                                    if current_image_tensor is None:
                                        current_image_tensor = self.generate_empty_image()

                        if current_image_tensor is None:
                            current_image_tensor = self.generate_empty_image()

                        result_queue.put(("success", (current_text_output, current_image_tensor)))
                        print("[INFO] API response processing successful.")

                    except Exception as e_proc:
                        print(f"[ERROR] A critical error occurred while processing the successful API response: {str(e_proc)}")
                        result_queue.put(("error", e_proc))


                result_queue = queue.Queue()

                api_thread = threading.Thread(target=api_call)
                api_thread.daemon = True
                api_thread.start()

                text_output = ""
                image_tensor = self.generate_empty_image()

                try:
                    status, result = result_queue.get(timeout=timeout)
                    elapsed_time = time.time() - start_time

                    if status == "success":
                        text_output, image_tensor = result
                        print(f"[INFO] API request and processing successfully completed in {elapsed_time:.2f} seconds")
                    else:
                        error_exception = result
                        print(f"[ERROR] Error in API request/processing thread after {elapsed_time:.2f} seconds: {str(error_exception)}")
                        text_output = f"API call/processing error: {str(error_exception)}"
                        error_str = str(error_exception).lower()
                        if any(term in error_str for term in ["timeout", "connection", "network", "socket", "连接", "网络"]):
                            if not network_ok and use_proxy != True:
                                text_output += " Network connection test failed, consider enabling proxy."
                except queue.Empty:
                    elapsed_time = time.time() - start_time
                    print(f"[ERROR] API request/processing timed out in main thread, waited: {elapsed_time:.2f} seconds")

                    timeout_msg = f"Gemini API request/processing timed out, waited {timeout} seconds."
                    if not network_ok:
                        if use_proxy == True:
                            timeout_msg += f" Network connection test failed, the current proxy ({proxy_host}:{proxy_port}) may be invalid, please check proxy settings."
                        else:
                            timeout_msg += " Network connection test failed, consider enabling proxy."
                    else:
                        timeout_msg += " Network connection test successful, but API request/processing still timed out. This could be due to a busy server or a large request."
                    text_output = timeout_msg
            except Exception as e:
                print(f"[ERROR] Unhandled error in generate method: {str(e)}")
                text_output = f"Unhandled error: {str(e)}"
                if image_tensor is None:
                    image_tensor = self.generate_empty_image()

            return (text_output, image_tensor, actual_seed if actual_seed is not None else 0)
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

NODE_CLASS_MAPPINGS = {
    "SSL_GeminiAPIKeyConfig": SSL_GeminiAPIKeyConfig,
    "SSL_GeminiTextPrompt": SSL_GeminiTextPrompt,
    "GetKeyAPI": GetKeyAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSL_GeminiAPIKeyConfig": "Configure Gemini API Key",
    "SSL_GeminiTextPrompt": "Expanded Gemini Text/Image",
    "GetKeyAPI": "Get API Key from JSON",
}
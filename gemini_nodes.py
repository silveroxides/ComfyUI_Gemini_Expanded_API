import os
import json
import uuid
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

class SSL_GeminiAPIKeyConfig(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "api_key": (IO.STRING, {"multiline": False}),
            }
        }

    RETURN_TYPES = ("GEMINI_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "💠SSL/API/Gemini"

    def configure(self, api_key):
        config = {"api_key": api_key}
        return (config,)


class SSL_GeminiTextPrompt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "config": ("GEMINI_CONFIG",),
                "prompt": (IO.STRING, {"multiline": True}),
                "system_instruction": (IO.STRING, {"default": "You are a helpful AI assistant.", "multiline": True}),
                "model": (["gemini-1.0-pro", "gemini-exp-1206", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite-001", "gemini-2.0-flash-exp", "gemini-2.0-pro", "gemini-2.0-flash-live", "gemini-2.5-flash", "gemini-2.5", "gemini-2.5-pro-1p-freebie"], {"default": "gemini-2.0-flash"}),
                "temperature": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": (IO.FLOAT, {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": (IO.INT, {"default": 40, "min": 1, "max": 100, "step": 1}),
                "max_output_tokens": (IO.INT, {"default": 8192, "min": 1, "max": 8192, "step": 1}),
                "include_images": (IO.BOOLEAN, {"default": True}),
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
    CATEGORY = "💠SSL/API/Gemini"

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
                input_image=None, input_image_2=None, use_proxy=False, proxy_host="127.0.0.1", proxy_port=7890,
                use_seed=True, seed=0):
        original_http_proxy = os.environ.get('HTTP_PROXY')
        original_https_proxy = os.environ.get('HTTPS_PROXY')
        original_http_proxy_lower = os.environ.get('http_proxy')
        original_https_proxy_lower = os.environ.get('https_proxy')

        print(f"[INFO] Starting generation, model: {model}, temperature: {temperature}")

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

                            adapter = ProxyAdapter(proxy_url, max_retries=2)
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

                client = genai.Client(api_key=config["api_key"], http_options=types.HttpOptions(api_version='v1alpha'), **client_options)
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

                    contents = img_parts + [{"text": prompt}]
                except Exception as e:
                    print(f"[ERROR] Error processing input image: {str(e)}")
                    return (f"Error processing input image: {str(e)}", self.generate_empty_image(), actual_seed if actual_seed is not None else 0)
            else:
                contents = prompt

            response_modalities = ["text"]
            if include_images == True:
                response_modalities.append("image")

            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
                response_modalities=response_modalities,
                system_instruction=[
                    types.Part.from_text(text=system_instruction),
                ],
            )

            if use_seed == True and actual_seed is not None:
                try:
                    generate_content_config.seed = actual_seed
                except Exception as seed_error:
                    print(f"[WARNING] Failed to set seed for API request: {str(seed_error)}")

            try:
                print(f"[INFO] Sending API request to Gemini")

                start_time = time.time()
                timeout = 30

                def api_call():
                    last_api_exception = None
                    max_retries = 3
                    api_response = None

                    for attempt in range(max_retries):
                        try:
                            print(f"[INFO] API call attempt {attempt + 1}/{max_retries}")

                            if use_seed and actual_seed is not None:
                                current_seed_for_api_call = actual_seed + attempt
                                try:
                                    generate_content_config.seed = current_seed_for_api_call
                                    if attempt > 0:
                                        print(f"[INFO] Retrying with incremented seed: {current_seed_for_api_call}")
                                except AttributeError:
                                    print(f"[WARNING] Could not set 'seed' attribute on generate_content_config.")
                                except Exception as e_set_seed:
                                    print(f"[WARNING] Error setting seed {current_seed_for_api_call} for attempt {attempt + 1}: {str(e_set_seed)}")

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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSL_GeminiAPIKeyConfig": "Configure Gemini API Key",
    "SSL_GeminiTextPrompt": "Expanded Gemini Text/Image",
}
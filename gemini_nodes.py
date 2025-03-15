import os
import json
import uuid
import torch
import numpy as np
from PIL import Image
import io
import folder_paths
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
            print(f"[WARNING] 未找到 {package_name}，尝试安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"[INFO] 成功安装 {package_name}")
            except Exception as e:
                print(f"[ERROR] 安装 {package_name} 失败: {str(e)}")

try:
    check_and_install_dependencies()
except Exception as e:
    print(f"[WARNING] 检查依赖时出错: {str(e)}")

class SSL_GeminiAPIKeyConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
            }
        }
    
    RETURN_TYPES = ("GEMINI_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "💠SSL/API/Gemini"
    
    def configure(self, api_key):
        config = {"api_key": api_key}
        return (config,)


class SSL_GeminiTextPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("GEMINI_CONFIG",),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.0-pro"], {"default": "gemini-2.0-flash"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192, "step": 1}),
                "include_images": (["True", "False"], {"default": "True"}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "use_proxy": (["False", "True"], {"default": "False"}),
                "proxy_host": ("STRING", {"default": "127.0.0.1"}),
                "proxy_port": ("INT", {"default": 7890, "min": 1, "max": 65535}),
                "use_seed": (["False", "True"], {"default": "False"}),
                "seed_type": (["random", "fixed"], {"default": "random"}),
                "seed_value": ("INT", {"default": 66666, "min": 0, "max": 2147483647}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
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
    
    def generate(self, config, prompt, model, temperature, top_p, top_k, max_output_tokens, include_images, 
                input_image=None, use_proxy="False", proxy_host="127.0.0.1", proxy_port=7890,
                use_seed="False", seed_type="random", seed_value=42):
        original_http_proxy = os.environ.get('HTTP_PROXY')
        original_https_proxy = os.environ.get('HTTPS_PROXY')
        original_http_proxy_lower = os.environ.get('http_proxy')
        original_https_proxy_lower = os.environ.get('https_proxy')
        
        print(f"[INFO] 开始生成，模型: {model}, 温度: {temperature}")
        
        if use_seed == "True":
            if seed_type == "random":
                actual_seed = random.randint(0, 2147483647)
            else:
                actual_seed = seed_value
                
            random.seed(actual_seed)
            np.random.seed(actual_seed)
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(actual_seed)
        else:
            actual_seed = None
        
        try:
            if use_proxy == "True":
                if not proxy_host.startswith(('http://', 'https://')):
                    proxy_url = f"http://{proxy_host}:{proxy_port}"
                else:
                    proxy_url = f"{proxy_host}:{proxy_port}"
                
                os.environ['HTTP_PROXY'] = proxy_url
                os.environ['HTTPS_PROXY'] = proxy_url
                os.environ['http_proxy'] = proxy_url
                os.environ['https_proxy'] = proxy_url
                os.environ['REQUESTS_CA_BUNDLE'] = ''
                
                print(f"[INFO] 已启用代理: {proxy_url}")
            
            print(f"[INFO] 初始化Gemini客户端")
            try:
                client_options = {}
                
                if use_proxy == "True":
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
                            
                            adapter = ProxyAdapter(proxy_url, max_retries=3)
                            session.mount('http://', adapter)
                            session.mount('https://', adapter)
                            
                            session.verify = False
                            
                            http_client = google.api_core.http_client.RequestsHttpClient(session=session)
                            client_options["http_client"] = http_client
                            
                            print(f"[INFO] 已使用requests库设置代理到HTTP客户端")
                        except Exception as proxy_error:
                            print(f"[WARNING] 设置HTTP客户端代理失败: {str(proxy_error)}")
                    except ImportError as e:
                        print(f"[WARNING] 导入Google API HTTP客户端库失败: {str(e)}")
                
                client = genai.Client(api_key=config["api_key"], **client_options)
                print(f"[INFO] Gemini客户端初始化成功")
            except Exception as e:
                print(f"[ERROR] Gemini客户端初始化失败: {str(e)}")
                return (f"Gemini客户端初始化失败: {str(e)}", self.generate_empty_image())
            
            try:
                import socket
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5)
                
                test_host = "generativelanguage.googleapis.com"
                
                if use_proxy == "True":
                    try:
                        import socks
                        test_socket = socks.socksocket()
                        test_socket.set_proxy(socks.HTTP, proxy_host, proxy_port)
                        test_socket.settimeout(5)
                    except ImportError:
                        print(f"[WARNING] 未安装PySocks库，无法通过代理测试连接")
                
                test_socket.connect((test_host, 443))
                test_socket.close()
                network_ok = True
            except Exception as e:
                network_ok = False
                print(f"[WARNING] 网络连接测试失败: {str(e)}")
                
                if use_proxy == "True":
                    try:
                        import subprocess
                        cmd = f"curl -x {proxy_url} -s -o /dev/null -w '%{{http_code}}' https://{test_host}"
                        try:
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                            if result.returncode == 0 and result.stdout.strip() in ['200', '301', '302']:
                                network_ok = True
                            else:
                                print(f"[WARNING] curl代理测试失败，状态码: {result.stdout.strip() if result.stdout else 'N/A'}")
                        except Exception as curl_error:
                            print(f"[WARNING] curl测试失败: {str(curl_error)}")
                    except ImportError:
                        pass
            
            contents = []
            
            if input_image is not None:
                try:
                    img_array = input_image[0].cpu().numpy()
                    img_array = (img_array * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_array)
                    
                    img_byte_arr = io.BytesIO()
                    pil_img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    img_part = {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
                    txt_part = {"text": prompt}
                    
                    contents = [img_part, txt_part]
                except Exception as e:
                    print(f"[ERROR] 处理输入图像时出错: {str(e)}")
                    return (f"处理输入图像时出错: {str(e)}", self.generate_empty_image())
            else:
                contents = prompt
            
            response_modalities = ["text"]
            if include_images == "True":
                response_modalities.append("image")
                
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                response_modalities=response_modalities,
            )
            
            if use_seed == "True" and actual_seed is not None:
                try:
                    generate_content_config.seed = actual_seed
                except Exception as seed_error:
                    print(f"[WARNING] 无法设置种子到API请求: {str(seed_error)}")
            
            text_output = ""
            image_tensor = None
            
            try:
                print(f"[INFO] 发送API请求到Gemini")
                
                start_time = time.time()
                timeout = 30
                
                def api_call():
                    try:
                        api_response = client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=generate_content_config,
                        )
                        result_queue.put(("success", api_response))
                    except Exception as e:
                        print(f"[ERROR] API线程中出错: {str(e)}")
                        result_queue.put(("error", e))

                result_queue = queue.Queue()
                
                api_thread = threading.Thread(target=api_call)
                api_thread.daemon = True
                api_thread.start()
                
                try:
                    status, result = result_queue.get(timeout=timeout)
                    elapsed_time = time.time() - start_time
                    
                    if status == "success":
                        response = result
                        print(f"[INFO] API请求成功完成，耗时: {elapsed_time:.2f}秒")
                    else:
                        print(f"[ERROR] API请求线程中出错，耗时: {elapsed_time:.2f}秒，错误: {str(result)}")
                        error_str = str(result).lower()
                        if any(term in error_str for term in ["timeout", "connection", "network", "socket", "连接", "网络"]):
                            if not network_ok and use_proxy != "True":
                                return (f"API请求失败: {str(result)}。网络连接测试失败，建议启用代理。", self.generate_empty_image())
                        raise result
                except queue.Empty:
                    elapsed_time = time.time() - start_time
                    print(f"[ERROR] API请求超时，已等待: {elapsed_time:.2f}秒")
                    
                    timeout_msg = f"Gemini API请求超时，已等待{timeout}秒。"
                    if not network_ok:
                        if use_proxy == "True":
                            timeout_msg += f"网络连接测试失败，当前使用的代理({proxy_host}:{proxy_port})可能无效，请检查代理设置。"
                        else:
                            timeout_msg += "网络连接测试失败，建议启用代理。"
                    else:
                        timeout_msg += "网络连接测试成功，但API请求仍然超时，可能是服务器繁忙或请求内容过大。"
                    
                    return (timeout_msg, self.generate_empty_image())
                
                print(f"[INFO] 收到API响应")
                
                if hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text is not None:
                            text_output += part.text
                        
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
                                image_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            except Exception as e:
                                print(f"[ERROR] 图像处理错误: {str(e)}")
                                text_output += f"\n图像处理错误: {str(e)}"
            except Exception as e:
                print(f"[ERROR] API调用错误: {str(e)}")
                text_output = f"API调用错误: {str(e)}"
            
            if image_tensor is None:
                image_tensor = self.generate_empty_image()
                
            if use_seed == "True" and actual_seed is not None:
                seed_info = f"\n\n[种子信息: {actual_seed}]"
                text_output += seed_info
                
            return (text_output, image_tensor)
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
    "SSL_GeminiAPIKeyConfig": "💠SSL Gemini API Key",
    "SSL_GeminiTextPrompt": "💠SSL Gemini Text Prompt",
}
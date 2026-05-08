import os
import asyncio
from comfy_api.latest import ComfyExtension, io
from .gemini_nodes import GetKeyAPI, SSL_GeminiAPIKeyConfig, SSL_GeminiTextPrompt


class GeminiExtension(ComfyExtension):
	async def on_load(self) -> None:
		# Check if environment variables indicate Vertex AI usage
		if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "True" or os.environ.get("GOOGLE_CLOUD_PROJECT"):
			asyncio.create_task(self._background_auth())

	async def _background_auth(self) -> None:
		try:
			print("[INFO] Attempting background Vertex AI authentication...")
			from google import genai
			# This will absorb the potential 60s metadata server timeout
			client = genai.Client(vertexai=True)
			await asyncio.to_thread(client._api_client._access_token)
			print("[INFO] Background Vertex AI authentication successful.")
		except Exception as e:
			print(f"[WARNING] Background Vertex AI auth failed (will retry at inference): {e}")

	async def get_node_list(self) -> list[type[io.ComfyNode]]:
		return [
			GetKeyAPI,
			SSL_GeminiAPIKeyConfig,
			SSL_GeminiTextPrompt,
		]


async def comfy_entrypoint() -> GeminiExtension:
	return GeminiExtension()


__all__ = ['comfy_entrypoint']

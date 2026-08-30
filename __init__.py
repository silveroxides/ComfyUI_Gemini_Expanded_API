from comfy_api.latest import ComfyExtension, io
from .gemini_nodes import GetKeyAPI, SSL_GeminiAPIKeyConfig, SSL_GeminiVideoConfig, SSL_GeminiTextPrompt


class GeminiExtension(ComfyExtension):
	async def on_load(self) -> None:
		pass

	async def get_node_list(self) -> list[type[io.ComfyNode]]:
		return [
			GetKeyAPI,
			SSL_GeminiAPIKeyConfig,
			SSL_GeminiVideoConfig,
			SSL_GeminiTextPrompt,
		]


async def comfy_entrypoint() -> GeminiExtension:
	return GeminiExtension()


__all__ = ['comfy_entrypoint']

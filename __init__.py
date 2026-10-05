from comfy_api.latest import ComfyExtension, io
from .gemini_nodes import GetKeyAPI, SSL_GeminiAPIKeyConfig, SSL_GeminiVideoConfig, SSL_GeminiTextPrompt
from .interactions_nodes import (
	SSL_GeminiInteractionsAPIKeyConfig,
	SSL_GeminiInteractionsSafetyConfig,
	SSL_GeminiInteractionsTextPrompt,
)
from .response_schema import SSL_GeminiResponseSchema

WEB_DIRECTORY = "./web"


class GeminiExtension(ComfyExtension):
	async def on_load(self) -> None:
		pass

	async def get_node_list(self) -> list[type[io.ComfyNode]]:
		return [
			GetKeyAPI,
			SSL_GeminiAPIKeyConfig,
			SSL_GeminiVideoConfig,
			SSL_GeminiResponseSchema,
			SSL_GeminiTextPrompt,
			SSL_GeminiInteractionsAPIKeyConfig,
			SSL_GeminiInteractionsSafetyConfig,
			SSL_GeminiInteractionsTextPrompt,
		]


async def comfy_entrypoint() -> GeminiExtension:
	return GeminiExtension()


__all__ = ['comfy_entrypoint', 'WEB_DIRECTORY']

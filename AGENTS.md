# Agent Instructions

Use `$comfyui-custom-node-development` for all development, testing, review, and packaging work in this repository.

If the skill is unavailable, stop and tell the user that it must be installed before continuing.

## Local ComfyUI Context

<!-- comfyui-custom-node-context:start -->
comfyui_root: C:\Users\ishim\Tools\ComfyUI
custom_nodes_root: C:\Users\ishim\Tools\ComfyUI\custom_nodes
repository_root: C:\Users\ishim\Tools\ComfyUI\custom_nodes\ComfyUI_Gemini_Expanded_API
virtual_environment_root: C:\Users\ishim\Tools\ComfyUI\.venv
python_executable: C:\Users\ishim\Tools\ComfyUI\.venv\Scripts\python.exe
<!-- comfyui-custom-node-context:end -->

## Repository-Specific Instructions

- Use `tests/run_tests.py` for all Python test selection.
- During iteration, run an exact test or one explicit group.
- Before handoff, run `tests/run_tests.py --changed` once for accumulated changes.
- Run `tests/run_tests.py --final` only as a deliberate broader gate; do not repeat a successful gate without relevant changes.
- Stage a new production source before relying on `--changed`, or select its group explicitly.
- Treat the user's stated task boundary as a hard workspace-flow constraint. Do not expand a targeted change or question into adjacent authentication modes, transports, model catalogs, dependencies, UI redesigns, or speculative audits unless their relevance to the requested workflow is first established or the user explicitly approves the expanded scope.
- Keep inspection proportional to the exact question: search for the concrete identifier, read only its containing boundary, and stop once the behavior is supported. Do not replace a direct answer with broad file reads, multi-subsystem analysis, or oversized diffs.
- Describe plans and reviews through the user-facing controls and behaviors they affect before implementation details. Explicitly identify which visible fields are used, ignored, conditional, renamed, or preserved in each mode.

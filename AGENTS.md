# Agent Instructions

Read `AGENTS-LOCAL.md` when present. It contains repository-machine-specific
operating rules and must remain local only; do not stage or commit it.

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

### Gemini API workflow

- Use `$gemini-api-dev` alongside `$comfyui-custom-node-development` for Gemini
  API development and review in this repository.
- Preserve `client.models.generate_content`. Do not recommend or perform migration
  to the Interactions API unless the user explicitly asks to evaluate or implement
  that migration. A request to evaluate migration does not authorize implementing it.
- Use current official GenerateContent documentation through the skill's web
  fallback when its documentation MCP is unavailable. A docs MCP is not required.
- Verify documentation examples target GenerateContent and apply to the
  repository's SDK usage before using them. New documentation or skill guidance
  does not authorize changing API families.

### Evidence and claim discipline

- Do not present assumptions, plausible explanations, remembered behavior, or
  pattern-based guesses as established facts.
- Before making a causal, behavioral, compatibility, or architectural claim,
  verify it against direct evidence covering the relevant path. Inspect the
  actual source, configuration, inputs, consumers, connections, and runtime
  assembly needed to support the claim.
- Distinguish verified facts, supported inferences, and unknowns explicitly.
  When required evidence is unavailable, state that the conclusion cannot yet
  be determined instead of filling the gap with a likely explanation.
- Do not infer an end-to-end result from an isolated helper, preset, schema,
  unit test, string-construction test, or passing test suite. Verify the actual
  integration and consumer behavior before generalizing beyond what was tested.
- Do not invent workflow topology, data flow, precedence, model behavior, or
  user intent. Inspect the relevant artifact or ask for the specific missing
  evidence when it cannot be discovered locally.
- Do not generalize a model or prompt protocol from a shared tokenizer, helper,
  socket type, or similarly named node. Verify the target model, node contract,
  template assembly, and consuming path. A node-specific prefix or label does
  not establish generic encoder behavior.
- When correcting an unsupported claim, identify exactly which parts were
  verified and which parts were assumed. Do not replace one unsupported
  explanation with another.
- Accept failure and request guidance or advice on the approach when continued
  attempts are unlikely to work and are being made only to force the task toward
  apparent completion; do not continue speculative trial-and-error to avoid
  acknowledging that the current approach has failed.

### Test selection

- Use `tests/run_tests.py` for all Python test selection.
- During iteration, run an exact test or one explicit group.
- Before handoff, run `tests/run_tests.py --changed` once for accumulated changes.
- Run `tests/run_tests.py --final` only as a deliberate broader gate; do not repeat a successful gate without relevant changes.
- Stage a new production source before relying on `--changed`, or select its group explicitly.
- Treat the user's stated task boundary as a hard workspace-flow constraint. Do not expand a targeted change or question into adjacent authentication modes, transports, model catalogs, dependencies, UI redesigns, or speculative audits unless their relevance to the requested workflow is first established or the user explicitly approves the expanded scope.
- Keep inspection proportional to the exact question: search for the concrete identifier, read only its containing boundary, and stop once the behavior is supported. Do not replace a direct answer with broad file reads, multi-subsystem analysis, or oversized diffs.
- Describe plans and reviews through the user-facing controls and behaviors they affect before implementation details. Explicitly identify which visible fields are used, ignored, conditional, renamed, or preserved in each mode.

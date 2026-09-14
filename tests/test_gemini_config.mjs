import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../web/gemini_config.js", import.meta.url), "utf8");
let extension;
vm.runInNewContext(source.replace(/^import .*;\r?$/m, ""), {
  app: { registerExtension(value) { extension = value; } },
});

function nodeType() {
  return class {
    constructor() {
      this.widgets = [
        { name: "cache_seed", value: 42, linkedWidgets: [{ value: "randomize" }] },
        { name: "use_cache", value: false },
        { name: "seed", value: 7, linkedWidgets: [{ value: "randomize" }] },
      ];
    }
    onNodeCreated() { this.created = true; return "original result"; }
    configure(savedControl) { this.widgets[0].linkedWidgets[0].value = savedControl; }
  };
}

test("new API config nodes default only cache_seed control to fixed", () => {
  const Node = nodeType();
  extension.beforeRegisterNodeDef(Node, { name: "SSL_GeminiAPIKeyConfig" });
  const node = new Node();
  assert.equal(node.onNodeCreated(), "original result");
  assert.equal(node.created, true);
  assert.equal(node.widgets[0].linkedWidgets[0].value, "fixed");
  assert.equal(node.widgets[0].value, 42);
  assert.equal(node.widgets[1].value, false);
  assert.equal(node.widgets[2].linkedWidgets[0].value, "randomize");
  // Workflow configuration restores serialized choices after node creation.
  node.configure("increment");
  assert.equal(node.widgets[0].linkedWidgets[0].value, "increment");
  node.configure("randomize");
  assert.equal(node.widgets[0].linkedWidgets[0].value, "randomize");
});

test("other node classes and missing seed controls are untouched", () => {
  const Other = nodeType();
  extension.beforeRegisterNodeDef(Other, { name: "SSL_GeminiTextPrompt" });
  const other = new Other();
  other.onNodeCreated();
  assert.equal(other.widgets[0].linkedWidgets[0].value, "randomize");
  const Config = nodeType();
  extension.beforeRegisterNodeDef(Config, { name: "SSL_GeminiAPIKeyConfig" });
  const config = new Config();
  config.widgets = [];
  assert.doesNotThrow(() => config.onNodeCreated());
});

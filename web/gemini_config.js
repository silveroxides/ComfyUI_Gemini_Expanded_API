import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "GeminiExpandedAPI.ConfigSeedDefault",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SSL_GeminiAPIKeyConfig") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const seed = this.widgets?.find((widget) => widget.name === "cache_seed");
      const control = seed?.linkedWidgets?.[0];
      if (control) control.value = "fixed";
      return result;
    };
  },
});

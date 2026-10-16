import { loadModel, infer, WEIGHTS_SHA256 } from "./lib/engine.mjs";
self.onmessage = async ({ data }) => {
  let model;
  try {
    model = await loadModel(data.base);
    const predictions = await infer(
      model,
      data.rows,
      data.stride,
      (done, total) => self.postMessage({ kind: "progress", done, total }),
    );
    self.postMessage({
      kind: "result",
      predictions,
      parameters: model.countParams(),
      fingerprint: WEIGHTS_SHA256,
    });
  } catch (error) {
    self.postMessage({
      kind: "error",
      message: error.message || "Inference failed.",
    });
  } finally {
    model?.dispose();
  }
};

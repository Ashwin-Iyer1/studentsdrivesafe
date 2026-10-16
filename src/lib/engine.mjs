import * as tf from "@tensorflow/tfjs-core";
import { loadLayersModel } from "@tensorflow/tfjs-layers";
import "@tensorflow/tfjs-backend-cpu";
import { validateRows, windowStarts, WINDOW } from "./signals.mjs";
export const WEIGHTS_SHA256 =
  "3c2e40c6054c6359ca445d33894a4e374891630119f881f4b93644429396c76f";
export async function loadModel(base, fetcher = fetch) {
  const response = await fetcher(new URL("model.json", base));
  if (!response.ok)
    throw new Error(
      "Model metadata could not be loaded. Serve the app with its model folder.",
    );
  const spec = await response.json(),
    manifest = spec.weightsManifest;
  if (
    manifest.length !== 1 ||
    manifest[0].paths.length !== 1 ||
    manifest[0].paths[0] !== "group1-shard1of1.bin"
  )
    throw new Error("Unexpected model manifest.");
  const weights = await fetcher(new URL(manifest[0].paths[0], base));
  if (!weights.ok) throw new Error("Model weights could not be loaded.");
  const weightData = await weights.arrayBuffer();
  const hash = [
    ...new Uint8Array(await crypto.subtle.digest("SHA-256", weightData)),
  ]
    .map((v) => v.toString(16).padStart(2, "0"))
    .join("");
  if (hash !== WEIGHTS_SHA256)
    throw new Error("Weight fingerprint does not match the original artifact.");
  const modelTopology = structuredClone(spec.modelTopology);
  delete modelTopology.training_config;
  await tf.setBackend("cpu");
  await tf.ready();
  const model = await loadLayersModel(
    tf.io.fromMemory({
      modelTopology,
      weightSpecs: manifest[0].weights,
      weightData,
    }),
  );
  if (
    JSON.stringify(model.inputs[0].shape.slice(1)) !== "[100,6]" ||
    model.outputs[0].shape.at(-1) !== 1
  ) {
    model.dispose();
    throw new Error("Expected 100 × 6 input and one output.");
  }
  return model;
}
export async function infer(model, rows, stride, progress = () => {}) {
  validateRows(rows);
  const starts = windowStarts(rows.length, stride),
    results = [];
  for (const start of starts) {
    const input = tf.tensor3d(
      [rows.slice(start, start + WINDOW)],
      [1, WINDOW, 6],
    );
    let output;
    try {
      output = model.predict(input);
      const score = (await output.data())[0];
      if (!Number.isFinite(score) || score < 0 || score > 1)
        throw new Error("Model returned an invalid score.");
      results.push({ start, score });
    } finally {
      input.dispose();
      output?.dispose();
    }
    progress(results.length, starts.length);
  }
  return results;
}

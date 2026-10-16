import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import * as tf from "@tensorflow/tfjs-core";
import { loadModel, infer } from "../src/lib/engine.mjs";
import {
  syntheticTrace,
  toCSV,
  parseCSV,
  windowStarts,
  splitPlan,
  summarizeScores,
} from "../src/lib/signals.mjs";
const base = new URL("../public/model/", import.meta.url);
const localFetch = async (url) => {
  const bytes = await readFile(url);
  return new Response(bytes);
};
test("CSV preserves order and distinct samples; rejects missing features and short sequences", () => {
  const rows = syntheticTrace();
  assert.deepEqual(parseCSV(toCSV(rows)).rows, rows);
  assert.notDeepEqual(rows[0], rows[1]);
  assert.throws(() => parseCSV("Acc X,Acc Y\n1,2"));
  assert.throws(() =>
    parseCSV(toCSV(rows).split("\n").slice(0, 40).join("\n")),
  );
});
test("windows never pad and chronological split shares no rows", () => {
  assert.deepEqual(windowStarts(125, 25), [0, 25]);
  assert.deepEqual(windowStarts(100, 100), [0]);
  const split = splitPlan(600);
  assert.ok(split.train.at(-1) + 100 <= split.boundary);
  assert.ok(split.evaluation[0] >= split.boundary);
});
test("strict threshold and label aggregation retain original semantics", () => {
  const labels = Array(100).fill(0);
  labels.fill(1, 0, 5);
  const at = summarizeScores([{ start: 0, score: 0.5 }], 0.5, labels);
  assert.equal(at.flagged, 0);
  assert.equal(at.windows[0].label, 1);
  assert.equal(at.metrics.precision, null);
  assert.equal(summarizeScores([{ start: 0, score: 0.5 }], 0.49).flagged, 1);
});
test("tampered weight bytes fail integrity before inference", async () => {
  await assert.rejects(
    loadModel(base, async (url) =>
      String(url).endsWith(".bin")
        ? new Response(new Uint8Array([0]))
        : localFetch(url),
    ),
    /fingerprint/,
  );
});
test("original browser artifacts execute deterministically with tensor cleanup and progress", async () => {
  const model = await loadModel(base, localFetch);
  assert.equal(model.countParams(), 68097);
  const rows = syntheticTrace("impulse", 125),
    progress = [];
  const before = tf.memory().numTensors;
  const first = await infer(model, rows, 25, (done, total) =>
    progress.push([done, total]),
  );
  assert.deepEqual(first, await infer(model, rows, 25));
  assert.deepEqual(progress, [
    [1, 2],
    [2, 2],
  ]);
  assert.deepEqual(
    first.map((x) => x.start),
    [0, 25],
  );
  assert.ok(first.every((x) => x.score >= 0 && x.score <= 1));
  assert.equal(tf.memory().numTensors, before);
  model.dispose();
});
test("inference rejects invalid rows before model use", async () => {
  await assert.rejects(
    infer(
      {
        predict() {
          throw Error("must not run");
        },
      },
      [[1, 2]],
      25,
    ),
    /samples/,
  );
});

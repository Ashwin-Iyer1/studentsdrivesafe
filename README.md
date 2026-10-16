# Students Drive Safe · Signal Review

An offline research workbench for six-channel inertial-sensor sequences. Import a CSV or explore synthetic traces, run the original saved model, inspect each 100-sample window, vary the decision threshold, and export an auditable JSON report.

## Run

Requires Node.js 22+ and a modern browser (Chrome, Edge or Firefox). Initial package installation needs internet; the running application uses only its own bundled assets and model.

```sh
npm ci --ignore-scripts
npm test
npm run dev
```

For the production bundle: `npm run build`, then `npm run preview`. Serve `dist/` over HTTP on localhost or HTTPS; opening `index.html` with `file://` will not load the worker. The app requests no motion, location or camera permissions, and makes no inference API calls. Imported data stays in memory until reload; explicit CSV/JSON downloads are the only persistence.

## Workflow

1. Choose a clearly labeled synthetic trace or import 100–10,000 consecutive rows.
2. Select stride 25, 50 or 100. Each model input contains 100 distinct rows; incomplete trailing windows are omitted.
3. Analyze. A CPU web worker verifies the original weight fingerprint, loads the fixed architecture, reports progress, and can be canceled. Changing source or stride terminates previous work and clears its results.
4. Inspect a channel and select a window. The signal chart shades its exact sample span; the score chart always uses a 0–1 axis.
5. Adjust the threshold without rerunning inference. Export the sequence CSV or result JSON with provenance, limitations and an explicit raw-row split plan.

CSV headers: `Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z`. Reordering is supported; an optional binary `label` column is accepted. Blank, nonnumeric, duplicate, unknown or missing columns are rejected. The CSV format is numeric and deliberately does not support quoted cells. Optional window labels follow the original notebook: at least five positive row labels in 100 samples. A score is flagged only when strictly greater than the threshold.

## Evidence and limits

This repairs the original browser prototype's H5 loading, mutation of model architecture, and repetition of one hardcoded row. It does **not** establish road detection accuracy. The saved model responds near 0.98 on several synthetic fixtures; those are uncalibrated outputs, not safety probabilities or validated detections.

The companion [Pothole-detector notebook and model](https://github.com/Ashwin-Iyer1/Pothole-detector) define the six-feature order and 100-row input. The notebook shuffled overlapping windows before splitting, allowing raw-sample overlap between train and evaluation. Historical metrics are not independent validation. Training data, units, sensor orientation, sampling rate and a validated calibration set are unavailable. No normalization is invented here. Imported-label agreement is descriptive and affected by correlated windows.

`public/model/` contains byte-identical existing converted artifacts from that public companion repository at `4a6a2aad56c5e42fdf647b8930d7f4ac1974c2bb`, originally present before this upgrade. The model has two bidirectional LSTM layers, one LSTM and dense layers, totaling 68,097 parameters. SHA-256 of the original weight shard:

```text
3c2e40c6054c6359ca445d33894a4e374891630119f881f4b93644429396c76f
```

`src/lib/signals.mjs` shares the documented numeric contract from the companion at that commit; keep both copies and regression fixtures synchronized when changing it. `legacy/` preserves the original CRA prototype and original artifacts without executing or bundling them. Existing Git history retains authorship; this upgrade makes no new claim about dataset ownership or model performance.

## Verification

`npm test` covers sequence parsing, sample boundaries, disjoint partitioning, strict thresholds and original label rules, tampered model rejection, real-weight deterministic inference, progress, tensor cleanup, and rejection before inference. CI also builds the complete production worker. Browser checks should exercise preset → analyze → inspect → threshold → export, invalid CSV, source changes, cancellation, and a narrow layout.

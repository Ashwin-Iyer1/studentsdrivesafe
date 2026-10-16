import { useEffect, useMemo, useRef, useState } from "react";
import {
  FEATURES,
  parseCSV,
  toCSV,
  syntheticTrace,
  summarize,
  summarizeScores,
  windowStarts,
  splitPlan,
} from "./lib/signals.mjs";
const fmt = (n, digits = 4) => (n == null ? "—" : n.toFixed(digits));
function download(name, value, type) {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Chart({
  values,
  label,
  fixed = false,
  threshold,
  selected,
  span = 1,
}) {
  const rawMin = Math.min(...values),
    rawMax = Math.max(...values),
    padding = rawMax === rawMin ? Math.abs(rawMin) * 0.05 || 1 : 0;
  const min = fixed ? 0 : rawMin - padding,
    max = fixed ? 1 : rawMax + padding,
    range = max - min;
  const x = (i) => 45 + (i / Math.max(1, values.length - 1)) * 710,
    y = (v) => 176 - ((v - min) / range) * 144;
  return (
    <svg viewBox="0 0 800 215" role="img" aria-label={label}>
      <title>{label}</title>
      {[0, 0.5, 1].map((t) => (
        <g key={t}>
          <path d={`M45,${32 + t * 144}H755`} className="gridline" />
          <text x="0" y={37 + t * 144}>
            {fmt(max - t * range, 2)}
          </text>
        </g>
      ))}
      {selected != null && (
        <rect
          x={x(selected)}
          y="26"
          width={Math.max(
            2,
            x(Math.min(values.length - 1, selected + span - 1)) - x(selected),
          )}
          height="156"
          className="selection"
        />
      )}
      <path
        d={values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ")}
        className="trace"
      />
      {threshold != null && (
        <path d={`M45,${y(threshold)}H755`} className="threshold" />
      )}
      <text x="45" y="207">
        0
      </text>
      <text x="690" y="207">
        {values.length - 1} {fixed ? "window" : "sample"}
      </text>
    </svg>
  );
}
export default function App() {
  const [session, setSession] = useState(() => ({
    rows: syntheticTrace(),
    labels: null,
    source: "Synthetic impulse",
    synthetic: true,
  }));
  const [channel, setChannel] = useState(2),
    [threshold, setThreshold] = useState(0.5),
    [stride, setStride] = useState(25),
    [selected, setSelected] = useState(0);
  const [result, setResult] = useState(null),
    [busy, setBusy] = useState(false),
    [progress, setProgress] = useState(0),
    [error, setError] = useState(""),
    [notice, setNotice] = useState("Ready to inspect a synthetic sequence."),
    [csv, setCSV] = useState("");
  const worker = useRef(null),
    generation = useRef(0);
  const stop = () => {
    generation.current++;
    worker.current?.terminate();
    worker.current = null;
    setBusy(false);
  };
  useEffect(
    () => () => {
      generation.current++;
      worker.current?.terminate();
    },
    [],
  );
  const replace = (next) => {
    stop();
    setSession(next);
    setSelected(0);
    setResult(null);
    setProgress(0);
    setError("");
    setNotice(`${next.source} loaded. Run the model to generate new scores.`);
  };
  const importCSV = (text, source) => {
    try {
      replace({ ...parseCSV(text), source, synthetic: false });
    } catch (e) {
      setError(e.message);
    }
  };
  const stats = useMemo(() => summarize(session.rows), [session]);
  const starts = useMemo(
    () => windowStarts(session.rows.length, stride),
    [session, stride],
  );
  const decisions = useMemo(
    () =>
      result
        ? summarizeScores(result.predictions, threshold, session.labels)
        : null,
    [result, threshold, session],
  );
  const current = decisions?.windows[selected];
  const run = () => {
    stop();
    const id = generation.current;
    setBusy(true);
    setError("");
    setResult(null);
    setProgress(0);
    setNotice("Loading verified weights and computing on CPU…");
    const active = new Worker(
      new URL("./inference.worker.mjs", import.meta.url),
      { type: "module" },
    );
    worker.current = active;
    active.onmessage = ({ data }) => {
      if (id !== generation.current) return;
      if (data.kind === "progress") {
        setProgress(data.done / data.total);
        setNotice(`Computing window ${data.done} of ${data.total}…`);
      } else {
        setBusy(false);
        active.terminate();
        worker.current = null;
        if (data.kind === "error") {
          setError(data.message);
          setNotice("Inference stopped.");
        } else {
          setResult(data);
          setNotice(
            `${data.predictions.length} windows analyzed locally. Review scores and their limitations below.`,
          );
        }
      }
    };
    active.onerror = () => {
      if (id !== generation.current) return;
      active.terminate();
      worker.current = null;
      setBusy(false);
      setError(
        "The inference worker failed to load. Check model assets and refresh.",
      );
    };
    active.postMessage({
      rows: session.rows,
      stride,
      base: new URL(`${import.meta.env.BASE_URL}model/`, document.baseURI).href,
    });
  };
  const exportReport = () =>
    download(
      "signal-review.json",
      JSON.stringify(
        {
          schemaVersion: 1,
          source: session.source,
          synthetic: session.synthetic,
          featureOrder: FEATURES,
          sampleCount: session.rows.length,
          stride,
          ...decisions,
          model: {
            parameters: result.parameters,
            weightsSHA256: result.fingerprint,
          },
          featureSummary: stats,
          splitPlan: splitPlan(session.rows.length, 0.8, stride),
          limitations: [
            "Uncalibrated score, not a safety probability.",
            "Training units, orientation, sampling rate and independent validation are unavailable.",
            "Overlapping windows are correlated. Synthetic data demonstrates execution only.",
          ],
        },
        null,
        2,
      ),
      "application/json",
    );
  return (
    <>
      <header>
        <a className="brand" href="#main">
          <span className="brand-icon">∿</span> students drive safe{" "}
          <small>RESEARCH WORKBENCH</small>
        </a>
        <span className="badge">LOCAL REPLAY / NO LIVE SENSING</span>
      </header>
      <main id="main">
        <section className="hero">
          <div>
            <div className="eyebrow">INERTIAL SIGNAL REVIEW · 01</div>
            <h1>
              See what the
              <br />
              <em>sequence</em> says.
            </h1>
            <p>
              Inspect six channels of motion, follow every model window, and
              understand where the evidence ends.
            </p>
          </div>
          <div className="hero-note">
            <span className="dot" /> ORIGINAL MODEL, EXPLICIT LIMITS
            <p>
              68,097 parameters. 100 samples per window. A real saved model
              running locally, with no sensor permissions or account.
            </p>
            <a href="#method">Read the method ↗</a>
          </div>
        </section>
        <div className="workspace">
          <aside className="panel inputs">
            <div className="section-number">01 / INPUT</div>
            <h2>Choose a sequence</h2>
            <p className="muted">
              Synthetic fixtures exercise the pipeline; they have no road
              labels.
            </p>
            <div className="presets">
              {["steady", "impulse", "oscillation"].map((kind) => (
                <button
                  key={kind}
                  aria-pressed={session.source === `Synthetic ${kind}`}
                  onClick={() =>
                    replace({
                      rows: syntheticTrace(kind),
                      labels: null,
                      source: `Synthetic ${kind}`,
                      synthetic: true,
                    })
                  }
                >
                  {kind[0].toUpperCase() + kind.slice(1)}
                </button>
              ))}
            </div>
            <label className="file-button">
              Import sensor CSV
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={async (e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    if (file.size > 4_000_000)
                      setError("CSV must be smaller than 4 MB.");
                    else
                      try {
                        importCSV(await file.text(), file.name);
                      } catch {
                        setError("Could not read that file.");
                      }
                  }
                  e.target.value = "";
                }}
              />
            </label>
            <details>
              <summary>Paste CSV instead</summary>
              <label htmlFor="csv">Six feature columns, optional label</label>
              <textarea
                id="csv"
                rows="6"
                value={csv}
                onChange={(e) => setCSV(e.target.value)}
                placeholder="Acc X,Acc Y,Acc Z,gyro_x,gyro_y,gyro_z"
              />
              <button onClick={() => importCSV(csv, "Pasted CSV")}>
                Load pasted CSV
              </button>
            </details>
            <button
              className="text-button"
              onClick={() =>
                download("sensor-sequence.csv", toCSV(session.rows), "text/csv")
              }
            >
              Download this sequence ↓
            </button>
            <div className="rule" />
            <label htmlFor="stride">Window stride</label>
            <select
              id="stride"
              value={stride}
              onChange={(e) => {
                stop();
                setStride(Number(e.target.value));
                setSelected(0);
                setResult(null);
                setNotice("Stride changed. Run the model to recompute.");
              }}
            >
              <option value="25">25 samples · 75% overlap</option>
              <option value="50">50 samples · 50% overlap</option>
              <option value="100">100 samples · no overlap</option>
            </select>
            <button className="primary" onClick={run} disabled={busy}>
              {busy ? "Analyzing…" : "Analyze sequence →"}
            </button>
            {busy && (
              <button
                onClick={() => {
                  stop();
                  setNotice("Analysis canceled.");
                }}
              >
                Cancel analysis
              </button>
            )}
            <progress
              value={progress}
              max="1"
              aria-label="Inference progress"
              hidden={!busy}
            />
            <p className="status" role="status">
              {notice}
            </p>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
          </aside>
          <div className="plots">
            <section className="panel">
              <div className="section-heading">
                <div>
                  <div className="section-number">02 / SIGNAL</div>
                  <h2>{session.source}</h2>
                </div>
                <span className="pill">
                  {session.synthetic ? "SYNTHETIC" : "IMPORTED CSV"}
                </span>
              </div>
              <div className="metrics">
                <div>
                  <strong>{session.rows.length}</strong>
                  <span>consecutive samples</span>
                </div>
                <div>
                  <strong>{starts.length}</strong>
                  <span>complete windows</span>
                </div>
                <div>
                  <strong>6</strong>
                  <span>input features</span>
                </div>
              </div>
              <div className="channel-control">
                <label htmlFor="channel">Visible channel</label>
                <select
                  id="channel"
                  value={channel}
                  onChange={(e) => setChannel(Number(e.target.value))}
                >
                  {FEATURES.map((f, i) => (
                    <option value={i} key={f}>
                      {f}
                    </option>
                  ))}
                </select>
                <span>
                  mean {fmt(stats[channel].mean)} · σ{" "}
                  {fmt(stats[channel].standardDeviation)}
                </span>
              </div>
              <Chart
                values={session.rows.map((r) => r[channel])}
                label={`${FEATURES[channel]} values; selected 100-sample window is shaded`}
                selected={starts[selected]}
                span={100}
              />
              <p className="caption">
                Original numeric values. Units and sampling rate are
                undocumented, so the horizontal axis uses sample indices.
              </p>
              <label htmlFor="window">
                Inspect window {selected + 1} of {starts.length}{" "}
                <span className="muted">
                  · samples {starts[selected]}–{starts[selected] + 99}
                </span>
              </label>
              <input
                id="window"
                type="range"
                min="0"
                max={starts.length - 1}
                value={selected}
                onChange={(e) => setSelected(Number(e.target.value))}
              />
            </section>
            <section className="panel results">
              <div className="section-heading">
                <div>
                  <div className="section-number">03 / MODEL OUTPUT</div>
                  <h2>A score needs context.</h2>
                </div>
                {result && (
                  <button onClick={exportReport}>Export report ↓</button>
                )}
              </div>
              {result ? (
                <>
                  <div className="metrics">
                    <div>
                      <strong>{fmt(current?.score)}</strong>
                      <span>selected raw score</span>
                    </div>
                    <div>
                      <strong>
                        {decisions.flagged}
                        <small> / {starts.length}</small>
                      </strong>
                      <span>above threshold</span>
                    </div>
                    <div>
                      <strong>{current?.label ?? "—"}</strong>
                      <span>supplied window label</span>
                    </div>
                  </div>
                  <Chart
                    values={decisions.windows.map((w) => w.score)}
                    label="Raw model scores on a fixed zero to one axis; dashed line is threshold"
                    fixed
                    threshold={threshold}
                    selected={selected}
                  />
                  <label htmlFor="threshold">
                    Decision threshold <output>{threshold.toFixed(2)}</output>
                  </label>
                  <input
                    id="threshold"
                    type="range"
                    min="0"
                    max="1"
                    step=".01"
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                  />
                  <p className="caption">
                    A score strictly greater than the threshold is flagged. This
                    threshold is exploratory and has not been calibrated.
                  </p>
                  <details>
                    <summary>
                      Inspect window decisions ({starts.length})
                    </summary>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Window</th>
                            <th>Samples</th>
                            <th>Score</th>
                            <th>Decision</th>
                          </tr>
                        </thead>
                        <tbody>
                          {decisions.windows.map((w, i) => (
                            <tr key={w.start}>
                              <td>
                                <button
                                  aria-label={`Inspect window ${i + 1}`}
                                  onClick={() => setSelected(i)}
                                >
                                  {i + 1}
                                </button>
                              </td>
                              <td>
                                {w.start}–{w.endExclusive - 1}
                              </td>
                              <td>{fmt(w.score)}</td>
                              <td>
                                {w.flagged
                                  ? "Above threshold"
                                  : "Below / equal"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                  {decisions.metrics && (
                    <p className="caption">
                      Supplied-label agreement: TP {decisions.metrics.tp}, FP{" "}
                      {decisions.metrics.fp}, TN {decisions.metrics.tn}, FN{" "}
                      {decisions.metrics.fn}. Precision{" "}
                      {fmt(decisions.metrics.precision)}, recall{" "}
                      {fmt(decisions.metrics.recall)}. Correlated windows are
                      not independent evaluation.
                    </p>
                  )}
                </>
              ) : (
                <div className="empty">
                  <span>∿</span>
                  <h3>Your trace is ready for review.</h3>
                  <p>
                    Analyze the sequence to compute scores from the saved
                    weights. Changing the source or stride clears previous
                    results.
                  </p>
                </div>
              )}
            </section>
          </div>
        </div>
        <section id="method" className="method">
          <div>
            <div className="eyebrow">METHOD & PROVENANCE</div>
            <h2>
              Be precise about
              <br />
              what this proves.
            </h2>
            <a href="https://github.com/Ashwin-Iyer1/Pothole-detector">
              Model notebook & artifacts ↗
            </a>
          </div>
          <div>
            <h3>Faithful inference</h3>
            <p>
              The original two bidirectional LSTMs, LSTM and dense layers
              consume 100 distinct consecutive rows. The weight shard is checked
              against its SHA-256 fingerprint before loading. No architecture is
              appended and no invented normalization is applied.
            </p>
            <h3>Research limitations</h3>
            <p>
              The historical notebook shuffled overlapping windows before
              splitting, allowing raw samples to overlap between sets. Its
              reported accuracy is not independent validation. Training data,
              sensor units, orientation and sampling rate are unavailable. Treat
              the output as an uncalibrated model response, never a road-safety
              probability.
            </p>
            <h3>Local and inspectable</h3>
            <p>
              Inference runs in a cancellable CPU web worker. Inputs stay in
              this tab; exporting is your explicit action. Optional row labels
              use the notebook’s rule: at least five positive labels in a
              100-sample window.
            </p>
          </div>
        </section>
      </main>
      <footer>
        Students Drive Safe / Offline signal review{" "}
        <a href="https://github.com/Ashwin-Iyer1/studentsdrivesafe">Source ↗</a>
      </footer>
    </>
  );
}

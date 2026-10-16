/** The notebook's recorded feature order; units and sample rate were not recorded. */
export const FEATURES = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z'];
export const WINDOW = 100;
export function validateRows(rows) {
  if (!Array.isArray(rows) || rows.length < WINDOW || rows.length > 10000) throw new Error('Supply 100–10,000 consecutive samples.');
  rows.forEach((row, i) => {
    if (!Array.isArray(row) || row.length !== 6 || row.some(v => !Number.isFinite(v) || Math.abs(v) > 1e6)) throw new Error(`Sample ${i + 1} must contain six finite numbers within ±1,000,000.`);
  });
  return rows;
}
export function parseCSV(text) {
  if (typeof text !== 'string' || text.length > 4_000_000) throw new Error('CSV must be text smaller than 4 MB.');
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  const header = lines.shift().split(',').map(v => v.trim());
  if (new Set(header).size !== header.length || FEATURES.some(f => !header.includes(f))) throw new Error(`Use the columns: ${FEATURES.join(',')}. Optional: label.`);
  if (header.some(f => !FEATURES.includes(f) && f !== 'label')) throw new Error('Unexpected CSV column. Use the six feature columns and optional label only.');
  const labels = header.includes('label') ? [] : null;
  const rows = lines.map((line, i) => {
    const cells = line.split(',').map(v => v.trim());
    if (cells.length !== header.length || cells.some(v => !v || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(v))) throw new Error(`CSV row ${i + 2} has missing or nonnumeric values.`);
    if (labels) {
      const label = Number(cells[header.indexOf('label')]);
      if (label !== 0 && label !== 1) throw new Error(`CSV row ${i + 2}: label must be 0 or 1.`);
      labels.push(label);
    }
    return FEATURES.map(f => Number(cells[header.indexOf(f)]));
  });
  validateRows(rows);
  return { rows, labels };
}
export function toCSV(rows) { validateRows(rows); return FEATURES.join(',') + '\n' + rows.map(r => r.join(',')).join('\n') + '\n'; }
export function windowStarts(length, stride = 25, start = 0, end = length) {
  if (![length, stride, start, end].every(Number.isInteger) || length < WINDOW || stride < 1 || stride > WINDOW || start < 0 || end > length || end < start) throw new Error('Invalid window bounds or stride (1–100).');
  const starts = [];
  for (let i = start; i + WINDOW <= end; i += stride) starts.push(i);
  return starts;
}
export function splitPlan(length, fraction = .8, stride = 25) {
  if (!Number.isFinite(fraction) || fraction <= 0 || fraction >= 1) throw new Error('Split fraction must be between 0 and 1.');
  const boundary = Math.floor(length * fraction);
  const train = windowStarts(length, stride, 0, boundary);
  const evaluation = windowStarts(length, stride, boundary, length);
  return { boundary, train, evaluation, sharedSamples: 0, note: 'Partition raw rows before windowing. This plan does not establish that this saved model has not seen the data.' };
}
export function summarize(rows) {
  validateRows(rows);
  return FEATURES.map((name, channel) => {
    const values = rows.map(row => row[channel]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    return { name, min: Math.min(...values), max: Math.max(...values), mean, standardDeviation: Math.sqrt(variance), constant: variance < 1e-12 };
  });
}
export function summarizeScores(predictions, threshold = .5, labels = null) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Threshold must be between 0 and 1.');
  let tp = 0, fp = 0, tn = 0, fn = 0;
  const windows = predictions.map(({ start, score }) => {
    if (!Number.isInteger(start) || start < 0 || !Number.isFinite(score) || score < 0 || score > 1) throw new Error('Invalid prediction.');
    const flagged = score > threshold; // Matches the original notebook's strict comparison.
    let label = null;
    if (labels) {
      const slice = labels.slice(start, start + WINDOW);
      if (slice.length !== WINDOW || slice.some(v => v !== 0 && v !== 1)) throw new Error('Labels do not cover this window.');
      label = slice.reduce((a, b) => a + b, 0) >= 5 ? 1 : 0;
      if (flagged && label) tp++; else if (flagged) fp++; else if (label) fn++; else tn++;
    }
    return { start, endExclusive: start + WINDOW, score, flagged, label };
  });
  const ratio = (a, b) => b ? a / b : null;
  return { threshold, windows, flagged: windows.filter(w => w.flagged).length, metrics: labels ? { tp, fp, tn, fn, precision: ratio(tp, tp + fp), recall: ratio(tp, tp + fn), accuracy: ratio(tp + tn, windows.length), note: 'Descriptive agreement with supplied labels; overlapping windows are correlated. Not independent validation.' } : null };
}
export function syntheticTrace(kind = 'impulse', length = 600) {
  if (!['steady', 'impulse', 'oscillation'].includes(kind) || !Number.isInteger(length) || length < 100 || length > 10000) throw new Error('Invalid synthetic trace.');
  return Array.from({ length }, (_, i) => {
    const wave = Math.sin(i * .13), bump = kind === 'impulse' ? 8 * Math.exp(-(((i - length / 2) / 7) ** 2)) : kind === 'oscillation' ? 2 * Math.sin(i * .6) : 0;
    return [.12 * wave, .08 * Math.cos(i * .17), 9.81 + .1 * wave + bump, .01 * wave, .02 * Math.cos(i * .1), .01 * wave + bump * .04];
  });
}

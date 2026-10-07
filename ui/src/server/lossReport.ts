import sqlite3 from 'sqlite3';

export interface LossReportOptions {
  mode: 'threshold' | 'top';
  target: 'step' | 'image';
  window: number;
  threshold: number;
  top: number;
  start: number | null;
  end: number | null;
  offset: number;
  limit: number;
}

export function parseLossReportOptions(params: URLSearchParams): LossReportOptions {
  const number = (key: string, fallback: number | null, min: number, max: number, integer = true) => {
    const raw = params.get(key);
    if (raw === null || raw.trim() === '') return fallback;
    const value = Number(raw);
    if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value)))
      throw new Error(`Invalid ${key}.`);
    return value;
  };
  const mode = params.get('mode') ?? 'threshold';
  const target = params.get('target') ?? 'step';
  if (!['threshold', 'top'].includes(mode) || !['step', 'image'].includes(target))
    throw new Error('Invalid report mode or target.');
  const options: LossReportOptions = {
    mode: mode as LossReportOptions['mode'],
    target: target as LossReportOptions['target'],
    window: number('window', 100, 1, 10000)!,
    threshold: number('threshold', 3, 1, 1000000, false)!,
    top: number('top', 20, 1, 10000)!,
    start: number('start', null, 0, Number.MAX_SAFE_INTEGER),
    end: number('end', null, 0, Number.MAX_SAFE_INTEGER),
    offset: number('offset', 0, 0, Number.MAX_SAFE_INTEGER)!,
    limit: number('limit', 50, 1, 200)!,
  };
  if (options.start !== null && options.end !== null && options.start > options.end)
    throw new Error('Start step must be less than or equal to end step.');
  return options;
}

export function all<T = any>(db: sqlite3.Database, sql: string, params: unknown[] = []): Promise<T[]> {
  return new Promise((resolve, reject) =>
    db.all(sql, params, (err, rows) => (err ? reject(err) : resolve(rows as T[]))),
  );
}

export async function hasTrainingExamples(db: sqlite3.Database) {
  return (await all(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_examples'")).length > 0;
}

/** Compute baselines over full history BEFORE applying the selected step range. */
export async function queryLossReport(db: sqlite3.Database, options: LossReportOptions) {
  if (!(await hasTrainingExamples(db))) return { available: false, total: 0, entries: [], recorded: 0 };
  const { window, target, mode, threshold, start, end, top, limit, offset } = options;
  const minHistory = Math.min(20, window);
  const observed = target === 'step' ? 'step_loss' : 'weighted_loss';
  const anomalous = target === 'step' ? 'step_loss IS NULL' : "loss_kind NOT IN ('finite', 'unattributed')";
  // The only interpolated input is a bounded integer window; everything else is bound.
  const history = `WITH step_values AS (
    SELECT s.step, s.wall_time,
      CASE WHEN COALESCE(a.value_real, b.value_real) BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308
        THEN COALESCE(a.value_real, b.value_real) ELSE NULL END AS step_loss
    FROM steps s
    LEFT JOIN metrics a ON a.step=s.step AND a.key='loss/loss'
    LEFT JOIN metrics b ON b.step=s.step AND b.key='loss'
    WHERE a.step IS NOT NULL OR b.step IS NOT NULL
  ), history AS (
    SELECT *, AVG(step_loss) OVER (ORDER BY step ROWS BETWEEN ${window} PRECEDING AND 1 PRECEDING) AS baseline,
      COUNT(step_loss) OVER (ORDER BY step ROWS BETWEEN ${window} PRECEDING AND 1 PRECEDING) AS history_count
    FROM step_values
  ), examples AS (
    SELECT e.*, h.wall_time, h.step_loss, h.baseline, h.history_count
    FROM training_examples e JOIN history h ON h.step=e.step
    WHERE (? IS NULL OR e.step>=?) AND (? IS NULL OR e.step<=?)
  ), scored AS (
    SELECT *, CASE WHEN ${anomalous} THEN 'nonfinite'
      WHEN baseline=0 AND ${observed}>0 THEN 'zero_baseline' ELSE 'finite' END AS spike_kind,
      CASE WHEN baseline>0 THEN ${observed}/baseline ELSE NULL END AS spike_ratio,
      CASE WHEN ${anomalous} THEN 2 WHEN baseline=0 AND ${observed}>0 THEN 1 ELSE 0 END AS priority
    FROM examples
    WHERE ${anomalous} OR (history_count>=${minHistory} AND baseline>=0 AND ${observed}>baseline)
  ), matching AS (
    SELECT * FROM scored WHERE ?='top' OR priority>0 OR ${observed}>?*baseline
  )`;
  const args: unknown[] = [start, start, end, end, mode, threshold];
  // Top X STEPS returns every image in those steps, even if batches contain >1 image.
  const selection =
    mode === 'top' && target === 'step'
      ? `, ranked_steps AS (SELECT step, MAX(priority) AS p, MAX(spike_ratio) AS r FROM matching GROUP BY step
        ORDER BY p DESC, r DESC, step DESC LIMIT ?), selected AS (
        SELECT m.* FROM matching m JOIN ranked_steps r ON r.step=m.step)`
      : `, selected AS (SELECT * FROM matching ORDER BY priority DESC, spike_ratio DESC, step DESC, microbatch, item_index
        ${mode === 'top' ? 'LIMIT ?' : ''})`;
  if (mode === 'top') args.push(top);
  const rows = await all(
    db,
    `${history}${selection}
    SELECT e.*, m.path, m.dataset_path, COUNT(*) OVER () AS total
    FROM selected e JOIN training_example_metadata m ON m.id=e.metadata_id
    ORDER BY priority DESC, spike_ratio DESC, step DESC, microbatch, item_index LIMIT ? OFFSET ?`,
    [...args, limit, offset],
  );
  const total = rows.length
    ? rows[0].total
    : (await all(db, `${history}${selection} SELECT COUNT(*) AS n FROM selected`, args))[0].n;
  const recorded = (await all(db, 'SELECT COUNT(*) AS n FROM training_examples'))[0].n;
  const entries = rows.map(({ total: _total, presentation_json, ...row }) => ({
    ...row,
    presentation: JSON.parse(presentation_json),
    spike_ratio: Number.isFinite(row.spike_ratio) ? row.spike_ratio : null,
  }));
  return { available: true, total, entries, recorded, min_history: minHistory };
}

export async function queryLossReportDetail(db: sqlite3.Database, step: number, microbatch: number, item: number) {
  if (!(await hasTrainingExamples(db))) return null;
  const rows = await all(
    db,
    `SELECT e.*, m.metadata_json, b.rng_state_json, s.wall_time, lr.value_real AS learning_rate,
      (SELECT COUNT(*) FROM training_examples batch WHERE batch.step=e.step AND batch.microbatch=e.microbatch) AS microbatch_size
    FROM training_examples e JOIN training_example_metadata m ON m.id=e.metadata_id
    JOIN steps s ON s.step=e.step
    LEFT JOIN training_batches b ON b.step=e.step AND b.microbatch=e.microbatch
    LEFT JOIN metrics lr ON lr.step=e.step AND lr.key='learning_rate'
    WHERE e.step=? AND e.microbatch=? AND e.item_index=?`,
    [step, microbatch, item],
  );
  if (!rows.length) return null;
  const { metadata_json, presentation_json, rng_state_json, ...row } = rows[0];
  return {
    ...row,
    metadata: JSON.parse(metadata_json),
    presentation: JSON.parse(presentation_json),
    rng_state: rng_state_json ? JSON.parse(rng_state_json) : null,
  };
}

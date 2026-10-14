"""Candidate schema v1 and authoritative repeated-development OOF boundaries.

Task 12 supplies independently loaded ``expected_rows`` and a JSON manifest:
schema_version, execution_mode, non_comparative, n03_eligible, source_notebook,
source_cell_id, run_id, split_sha256, data_sha256, config_sha256,
vault_row_indices, vault_building_ids, protocol, candidates.

``protocol`` contains repeat_ids, fold_ids and split_seeds (string repeat keys).
Full protocols are 3 repeats / 4 folds; synthetic_smoke is 1 repeat / 2 folds.
Each candidate declares CANDIDATE_COLUMNS and one model_fit_config_sha256;
only the HPO challenger instead declares fold_fit_config_sha256, keyed "r:f".
These are evidence contracts, never real-run execution authorization.

Writer failure leaves a confined partial file for the runner's failure lifecycle.
Successful writing reports written_unvalidated, never completed or promoted.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from project.experiments.candidate_contracts import SOURCE_NOTEBOOKS, CandidateScore
from project.models.candidate_scores import evaluate_candidate_metrics

ROW_COLUMNS = ('row_index', 'repeat', 'fold', 'building_id', 'timestamp', 'target',
               'split_seed', 'split_sha256', 'data_sha256', 'config_sha256', 'run_id')
CANDIDATE_COLUMNS = ('recipe_id', 'model_family', 'model_config_id', 'experiment_id',
                     'arm_id', 'score_kind', 'positive_label')
HPO_EXPERIMENT = 'regime_same_family_hpo_vs_default'
MAX_BATCH_ROWS = 64_000
RUN_COLUMNS = ('split_sha256', 'data_sha256', 'config_sha256', 'run_id',
               'execution_mode', 'non_comparative', 'n03_eligible')


def candidate_oof_schema(building_id_type=None, timestamp_type=None):
    """Canonical baseline lineage plus candidate v1 fields; all fields are nonnull."""
    building_id_type = pa.string() if building_id_type is None else building_id_type
    timestamp_type = pa.timestamp('ns') if timestamp_type is None else timestamp_type
    if building_id_type not in (pa.string(), pa.int64()) or not pa.types.is_timestamp(timestamp_type):
        raise ValueError('schema requires opaque string/int64 buildings and Arrow timestamps')
    types = {
        'candidate_id': pa.string(), 'recipe_id': pa.string(), 'model_family': pa.string(),
        'model_config_id': pa.string(), 'row_index': pa.int64(), 'repeat': pa.int64(),
        'fold': pa.int64(), 'building_id': building_id_type, 'timestamp': timestamp_type,
        'prediction': pa.float64(), 'target': pa.int64(), 'split_sha256': pa.string(),
        'data_sha256': pa.string(), 'config_sha256': pa.string(), 'run_id': pa.string(),
        'split_seed': pa.int64(), 'experiment_id': pa.string(), 'arm_id': pa.string(),
        'score_kind': pa.string(), 'positive_label': pa.int64(), 'execution_mode': pa.string(),
        'non_comparative': pa.bool_(), 'n03_eligible': pa.bool_(),
        'model_fit_config_sha256': pa.string(),
    }
    return pa.schema([pa.field(name, dtype, nullable=False) for name, dtype in types.items()],
                     metadata={b'candidate_schema_version': b'1'})


CANDIDATE_OOF_SCHEMA = candidate_oof_schema()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{name} must be nonempty unpadded text')


def _hash(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('lineage/config hash must be lowercase SHA-256')


def _flags(record):
    mode = record.get('execution_mode')
    if (mode not in ('full', 'synthetic_smoke')
            or type(record.get('non_comparative')) is not bool
            or type(record.get('n03_eligible')) is not bool):
        raise ValueError('invalid execution/eligibility metadata')
    if record['non_comparative'] != (mode == 'synthetic_smoke') or (
            mode == 'synthetic_smoke' and record['n03_eligible']):
        raise ValueError('synthetic evidence cannot claim comparative eligibility')


def _is_hpo(record):
    return record['experiment_id'] == HPO_EXPERIMENT and record['arm_id'] == 'challenger'


def _candidate_metadata(record):
    for name in CANDIDATE_COLUMNS:
        if name != 'positive_label':
            _text(record.get(name), name)
    if type(record.get('positive_label')) is not int or record['positive_label'] != 1:
        raise ValueError('positive class must be integer 1')
    if record['score_kind'] not in ('positive_probability', 'decision_margin'):
        raise ValueError('unsupported score kind')


def _json_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError('manifest JSON keys must be strings, without coercion')
            _json_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _json_keys(child)


def _manifest(manifest):
    if not isinstance(manifest, dict):
        raise TypeError('manifest must be a JSON object')
    required = {'schema_version', 'source_notebook', 'source_cell_id', 'vault_row_indices',
                'vault_building_ids', 'protocol', 'candidates', *RUN_COLUMNS}
    if set(manifest) != required:
        raise ValueError('manifest fields must match candidate schema v1 contract')
    # Copy into JSON-only primitive containers; no arbitrary caller objects survive.
    encoded = json.dumps(manifest, allow_nan=False)
    _json_keys(manifest)
    result = json.loads(encoded)
    if type(result['schema_version']) is not int or result['schema_version'] != 1:
        raise ValueError('unsupported candidate schema version')
    _flags(result)
    if result['source_notebook'] not in SOURCE_NOTEBOOKS:
        raise ValueError('manifest source notebook is not approved')
    _text(result['source_cell_id'], 'source_cell_id')
    _text(result['run_id'], 'run_id')
    for name in RUN_COLUMNS[:3]:
        _hash(result[name])
    for name in ('vault_row_indices', 'vault_building_ids'):
        values = result[name]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError('vault IDs must be explicit unique lists')
        for value in values:
            if name == 'vault_row_indices' or type(value) is int:
                if type(value) is not int or value < 0:
                    raise ValueError('vault IDs require nonnegative integers or opaque buildings')
            else:
                _text(value, name)
    protocol = result['protocol']
    if not isinstance(protocol, dict) or set(protocol) != {'repeat_ids', 'fold_ids', 'split_seeds'}:
        raise ValueError('explicit repeat/fold/seed protocol is required')
    repeats, folds = ([0, 1, 2], [0, 1, 2, 3]) if result['execution_mode'] == 'full' else ([0], [0, 1])
    if (protocol['repeat_ids'] != repeats or protocol['fold_ids'] != folds
            or any(type(v) is not int for v in protocol['repeat_ids'] + protocol['fold_ids'])):
        raise ValueError('repeat/fold protocol must match full or diagnostic execution')
    seeds = protocol['split_seeds']
    if (not isinstance(seeds, dict) or set(seeds) != {str(r) for r in repeats}
            or any(type(v) is not int or not 0 <= v < 2**32 for v in seeds.values())
            or len(set(seeds.values())) != len(repeats)):
        raise ValueError('protocol requires a distinct integer split seed per repeat')
    candidates = result['candidates']
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError('manifest requires explicit candidate arms')
    fold_keys = {f'{r}:{f}' for r in repeats for f in folds}
    for candidate, meta in candidates.items():
        _text(candidate, 'candidate_id')
        if not isinstance(meta, dict):
            raise TypeError('candidate metadata must be a JSON object')
        _candidate_metadata(meta)
        fit_field = 'fold_fit_config_sha256' if _is_hpo(meta) else 'model_fit_config_sha256'
        if set(meta) != {*CANDIDATE_COLUMNS, fit_field}:
            raise ValueError('candidate fit hash declaration must be fixed or HPO fold-scoped')
        if _is_hpo(meta):
            hashes = meta[fit_field]
            if not isinstance(hashes, dict) or set(hashes) != fold_keys:
                raise ValueError('HPO fit hashes must cover every declared repeat/fold')
            for value in hashes.values():
                _hash(value)
        else:
            _hash(meta[fit_field])
    return result


def _frame(value, expected=False):
    if isinstance(value, (pa.Table, pa.RecordBatch)):
        if not expected:
            _writer_schema(value.schema)
        value = pl.from_arrow(value)
    if not isinstance(value, pl.DataFrame):
        raise TypeError('artifact and authoritative rows must be Polars or Arrow tables')
    names = ROW_COLUMNS if expected else tuple(CANDIDATE_OOF_SCHEMA.names)
    if (expected and not set(names).issubset(value.columns)) or (
            not expected and tuple(value.columns) != names):
        raise ValueError('candidate artifact schema columns are missing or reordered')
    if value.height == 0 or any(value[n].null_count() for n in names):
        raise ValueError('artifact/expected rows must be nonempty and nonnull')
    types = CANDIDATE_OOF_SCHEMA
    for name in names:
        dtype = value.schema[name]
        if name == 'building_id':
            valid = dtype in (pl.String, pl.Int64)
        elif name == 'timestamp':
            valid = isinstance(dtype, pl.Datetime)
        else:
            arrow_type = types.field(name).type
            required = pl.String if pa.types.is_string(arrow_type) else (
                pl.Boolean if pa.types.is_boolean(arrow_type) else (
                    pl.Float64 if pa.types.is_float64(arrow_type) else pl.Int64))
            valid = dtype == required
        if not valid:
            raise ValueError(f'{name} dtype violates candidate schema v1')
    return value


def _row(record, prediction=True):
    for name in ('row_index', 'repeat', 'fold', 'split_seed'):
        if record[name] < 0:
            raise ValueError('row keys and split seeds must be nonnegative integers')
    if record['split_seed'] >= 2**32:
        raise ValueError('split seed must fit the declared uint32 protocol')
    building = record['building_id']
    if isinstance(building, str):
        _text(building, 'building_id')
    elif building < 0:
        raise ValueError('building IDs must be nonnegative')
    if record['target'] not in (0, 1):
        raise ValueError('target must be integer binary 0 or 1')
    for name in ('split_sha256', 'data_sha256', 'config_sha256'):
        _hash(record[name])
    _text(record['run_id'], 'run_id')
    if prediction:
        _text(record['candidate_id'], 'candidate_id')
        _candidate_metadata(record)
        _flags(record)
        repeat_count, fold_count = (3, 4) if record['execution_mode'] == 'full' else (1, 2)
        if record['repeat'] >= repeat_count or record['fold'] >= fold_count:
            raise ValueError('row repeat/fold is outside the declared execution protocol')
        _hash(record['model_fit_config_sha256'])
        value = record['prediction']
        if not np.isfinite(value) or (record['score_kind'] == 'positive_probability' and not 0 <= value <= 1):
            raise ValueError('prediction must be finite and probabilities within [0, 1]')


def _expected_protocol(expected, manifest):
    repeats = manifest['protocol']['repeat_ids']
    folds = manifest['protocol']['fold_ids']
    reference = None
    expected_map = {}
    for repeat in repeats:
        part = expected.filter(pl.col('repeat') == repeat)
        if set(part['fold'].to_list()) != set(folds):
            raise ValueError('authoritative expected fold coverage differs from protocol')
        population = set()
        natural_keys = set()
        group_folds = {}
        for row in _rows(part):
            _row(row, prediction=False)
            key = (row['row_index'], row['repeat'])
            if key in expected_map:
                raise ValueError('duplicate authoritative row/repeat key')
            natural = (row['building_id'], row['timestamp'])
            if natural in natural_keys:
                raise ValueError('duplicate authoritative natural key')
            natural_keys.add(natural)
            if row['building_id'] in group_folds and group_folds[row['building_id']] != row['fold']:
                raise ValueError('authoritative building spans multiple folds')
            group_folds[row['building_id']] = row['fold']
            if (row['row_index'] in manifest['vault_row_indices']
                    or row['building_id'] in manifest['vault_building_ids']):
                raise ValueError('vault membership is forbidden in development evidence')
            for name in RUN_COLUMNS[:4]:
                if row[name] != manifest[name]:
                    raise ValueError('authoritative lineage differs from manifest')
            if row['split_seed'] != manifest['protocol']['split_seeds'][str(repeat)]:
                raise ValueError('authoritative split seed differs from protocol')
            expected_map[key] = tuple(row[n] for n in ROW_COLUMNS)
            population.add((row['row_index'], row['building_id'], row['timestamp'], row['target']))
        if reference is not None and population != reference:
            raise ValueError('authoritative development population differs across repeats')
        reference = population
    if set(expected['repeat'].to_list()) != set(repeats):
        raise ValueError('authoritative repeat coverage differs from protocol')
    return expected_map


def validate_candidate_oof(oof, expected_rows, manifest) -> dict:
    """Check every arm's row/target/lineage alignment before computing any metrics.

    Never derive expected membership from predictions. The caller must independently
    load authoritative expected_rows, including repeat, seed and frozen run hashes.
    Results retain per-fold/per-repeat metrics; fold dependence is not a CI claim.
    """
    declaration = _manifest(manifest)
    frame, expected = _frame(oof), _frame(expected_rows, expected=True)
    if (frame.schema['building_id'] != expected.schema['building_id']
            or frame.schema['timestamp'] != expected.schema['timestamp']):
        raise ValueError('artifact natural key dtypes differ from authoritative rows')
    expected_map = _expected_protocol(expected, declaration)
    candidates = declaration['candidates']
    if set(frame['candidate_id'].to_list()) != set(candidates):
        raise ValueError('missing or unexpected candidate arms')
    # Phase one: all alignment checks finish before any arm is scored.
    for candidate, meta in candidates.items():
        observed = set()
        for row in _rows(frame.filter(pl.col('candidate_id') == candidate)):
            _row(row)
            key = (row['row_index'], row['repeat'])
            if key in observed:
                raise ValueError('duplicate candidate,row_index,repeat key')
            observed.add(key)
            if expected_map.get(key) != tuple(row[n] for n in ROW_COLUMNS):
                raise ValueError('OOF target/key/hash alignment differs from authoritative rows')
            for name in (*CANDIDATE_COLUMNS, *RUN_COLUMNS):
                if row[name] != (meta[name] if name in CANDIDATE_COLUMNS else declaration[name]):
                    raise ValueError(f'candidate metadata/lineage mismatch: {name}')
            fit_hash = (meta['fold_fit_config_sha256'][f"{row['repeat']}:{row['fold']}"]
                        if _is_hpo(meta) else meta['model_fit_config_sha256'])
            if row['model_fit_config_sha256'] != fit_hash:
                raise ValueError('candidate fit hash differs from declared fixed/fold configuration')
        if observed != set(expected_map):
            raise ValueError('candidate has incomplete expected development coverage')
    return _metric_report(frame, declaration, len(expected_map))


def _metric_report(frame, manifest, expected_count):
    repeat_metrics, fold_metrics, candidates = [], [], {}
    names = ('roc_auc', 'average_precision', 'log_loss', 'brier_score')
    for candidate, meta in manifest['candidates'].items():
        repeats = []
        for repeat in manifest['protocol']['repeat_ids']:
            part = frame.filter((pl.col('candidate_id') == candidate) & (pl.col('repeat') == repeat))
            report = evaluate_candidate_metrics(part['target'].to_numpy(),
                CandidateScore(meta['score_kind'], part['prediction'].to_numpy()))
            repeats.append(report)
            repeat_metrics.append({'candidate_id': candidate, 'repeat': repeat, **report})
            for fold in manifest['protocol']['fold_ids']:
                fold_part = part.filter(pl.col('fold') == fold)
                fold_report = evaluate_candidate_metrics(fold_part['target'].to_numpy(),
                    CandidateScore(meta['score_kind'], fold_part['prediction'].to_numpy()))
                fold_metrics.append({'candidate_id': candidate, 'repeat': repeat,
                                     'fold': fold, **fold_report})
        candidates[candidate] = {
            'mean_repeat_metrics': {name: None if repeats[0][name] is None else
                                    float(np.mean([row[name] for row in repeats])) for name in names},
            'guardrail_status': repeats[0]['guardrail_status'],
            'probability_pool_eligible': bool(meta['score_kind'] == 'positive_probability'
                and manifest['n03_eligible'] and not manifest['non_comparative']),
        }
    return {'schema_version': 1, 'status': 'validated', 'oof_record_count': frame.height,
            'expected_row_repeat_count': expected_count,
            'summary_basis': 'equal_mean_of_repeats_not_pooled_rows',
            'repeat_metrics': repeat_metrics, 'fold_metrics': fold_metrics,
            'candidate_metrics': candidates}


def _rows(frame):
    # Python datetime truncates nanoseconds to microseconds. Preserve the exact
    # physical timestamp for natural-key alignment rather than lossy conversion.
    return frame.with_columns(pl.col('timestamp').cast(pl.Int64)).iter_rows(named=True)


def _writer_schema(schema):
    if not isinstance(schema, pa.Schema):
        raise TypeError('explicit Arrow candidate schema v1 is required')
    if schema.names != CANDIDATE_OOF_SCHEMA.names:
        raise ValueError('writer schema must contain exact ordered candidate v1 fields')
    expected = candidate_oof_schema(schema.field('building_id').type, schema.field('timestamp').type)
    if not schema.equals(expected, check_metadata=True):
        raise ValueError('writer schema must preserve float64, nonnull fields and version metadata')


def _stream_identity(connection, row):
    try:
        connection.execute('INSERT INTO keys VALUES (?, ?, ?)',
                           (row['candidate_id'], row['row_index'], row['repeat']))
    except sqlite3.IntegrityError as exc:
        raise ValueError('duplicate candidate,row_index,repeat key across streamed batches') from exc
    metadata = {n: row[n] for n in (*CANDIDATE_COLUMNS, *RUN_COLUMNS)}
    if not _is_hpo(row):
        metadata['model_fit_config_sha256'] = row['model_fit_config_sha256']
    encoded = json.dumps(metadata, sort_keys=True)
    stored = connection.execute('SELECT metadata FROM candidates WHERE candidate=?',
                                (row['candidate_id'],)).fetchone()
    if stored is None:
        connection.execute('INSERT INTO candidates VALUES (?, ?)', (row['candidate_id'], encoded))
    elif stored[0] != encoded:
        raise ValueError('candidate metadata or fixed fit hash changed across streamed batches')
    key = (row['candidate_id'], row['repeat'], row['fold'])
    stored = connection.execute('SELECT hash FROM fits WHERE candidate=? AND repeat=? AND fold=?',
                                key).fetchone()
    if stored is None:
        connection.execute('INSERT INTO fits VALUES (?, ?, ?, ?)', (*key, row['model_fit_config_sha256']))
    elif stored[0] != row['model_fit_config_sha256']:
        raise ValueError('fold fit hash changed across streamed batches')


def write_candidate_oof_batches(path: Path, batches: Iterable, schema: pa.Schema) -> dict:
    """Exclusively write <=64,000-row Arrow batches without collecting OOF in RAM.

    A disk-backed temporary SQLite index detects cross-batch duplicates and metadata
    drift. Parent directories must already exist, be absolute and symlink-free.
    Failure raises, retaining any partial output; success is still unvalidated.
    """
    _writer_schema(schema)
    path = Path(path)
    if (not path.is_absolute() or path.parent.resolve(strict=True) != path.parent
            or path.is_symlink()):
        raise ValueError('artifact path must be absolute and confined to an existing symlink-free parent')
    count, run_identity = 0, None
    # Empty sqlite filename creates a temporary disk database, not an in-memory
    # all-OOF key set. Both handles close even when an iterator or validation fails.
    with path.open('xb') as output, closing(sqlite3.connect('')) as connection:
        connection.execute('CREATE TABLE keys(candidate TEXT, row INTEGER, repeat INTEGER, '
                           'PRIMARY KEY(candidate,row,repeat))')
        connection.execute('CREATE TABLE candidates(candidate TEXT PRIMARY KEY, metadata TEXT)')
        connection.execute('CREATE TABLE fits(candidate TEXT, repeat INTEGER, fold INTEGER, hash TEXT, '
                           'PRIMARY KEY(candidate,repeat,fold))')
        with pq.ParquetWriter(output, schema, compression='zstd') as writer:
            for batch in batches:
                if not isinstance(batch, (pa.Table, pa.RecordBatch)):
                    raise TypeError('writer accepts Arrow Table or RecordBatch batches')
                if not 0 < batch.num_rows <= MAX_BATCH_ROWS:
                    raise ValueError('stream batches must be nonempty and at most 64000 rows')
                if not batch.schema.equals(schema, check_metadata=True):
                    raise ValueError('batch schema must exactly match writer schema without casting')
                frame = _frame(batch)
                for row in frame.iter_rows(named=True):
                    _row(row)
                    identity = tuple(row[n] for n in RUN_COLUMNS)
                    if run_identity is not None and identity != run_identity:
                        raise ValueError('run lineage changed across streamed batches')
                    run_identity = identity
                    _stream_identity(connection, row)
                writer.write_table(pa.Table.from_batches([batch]) if isinstance(batch, pa.RecordBatch)
                                   else batch, row_group_size=MAX_BATCH_ROWS)
                count += batch.num_rows
            if count == 0:
                raise ValueError('cannot write an empty candidate artifact')
    return {'status': 'written_unvalidated', 'oof_record_count': count}

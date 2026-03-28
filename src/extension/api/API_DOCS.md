# API Documentation

Compact route summary for the Python Testing Pipeline API.

Base URL:

```text
http://localhost:8000
```

Notes:
- `/docs` and `/openapi.json` are the generated source of truth.
- `coverage` is a compatibility input on pipeline runs; coverage is already
  collected whenever generated tests are executed.
- Pipeline status tracking is in-memory only.

## Quick Examples

`POST /pipeline/run`

```json
{
  "codebase_path": "/absolute/path/to/codebase",
  "auto_approve": true,
  "run_tests": true,
  "coverage": false,
  "model": null
}
```

```json
{
  "success": true,
  "test_file": "/absolute/path/to/tests/test_generated_1712345678.py",
  "coverage_percent": 75.5,
  "error": null
}
```

`POST /agents/safety/validate`

```json
{
  "prompt": "Generate tests for the authentication module",
  "model": "meta-llama/llama-guard-4-12b"
}
```

## Shared Shapes

- `TestScenario`: `{ "scenario_description": "...", "priority": "High|Medium|Low" }`
- `SecurityIssue`: `{ "severity": "critical|high|medium|low", "issue": "...", "location": "...", "recommendation": "..." }`
- Error envelope: `{ "success": false, "error": "..." }`

## Health

- `GET /health`
  Returns `{ "status": "healthy", "version": "1.0.0" }`.
- `GET /info`
  Returns `{ "version", "available_models", "default_model" }`.
  Model metadata comes from `scripts/llm_config.py` when available.

## Pipeline

- `POST /pipeline/run`
  Request keys: `codebase_path`, `auto_approve`, `run_tests`, `coverage`, `model`.
  Returns: `success`, `scenarios_count`, `test_file`, `execution`, `coverage_percent`,
  `security_issues`, `recommendations`, `prompts_file`, `error`.
- `POST /pipeline/run/stream`
  Server-sent events. Emits a start line, then a serialized `PipelineResponse`.
- `GET /pipeline/status/{run_id}`
  Returns `PipelineStatusResponse` for an internally tracked run id.

## Agents

- `POST /agents/identify`
  Request: `codebase_path`, optional `model`.
  Returns: `success`, `scenarios`, `error`.
- `POST /agents/identify/refine`
  Request: `scenarios`, `feedback`, optional `model`.
  Returns the same shape as `/agents/identify`.
- `POST /agents/implement`
  Request: `scenarios`, `codebase_path`, optional `output_dir`, optional `model`.
  Returns: `success`, `test_file`, `test_code`, `error`.
- `POST /agents/implement/improve`
  Request: `codebase_path`, `existing_test_file`, `coverage_percentage`,
  `uncovered_areas`, optional `syntax_errors`, optional `security_issues`,
  optional `model`.
  Returns: `success`, `test_code`, `error`.
- `POST /agents/implement/fix-syntax`
  Request: `code`, `error_msg`, `codebase_path`, optional `model`.
  Returns: `success`, `test_code`, `error`.
- `POST /agents/evaluate`
  Request: `test_results`, `scenarios`, `codebase_path`, optional `model`.
  Returns: `success`, `evaluation`, `error`.
- `POST /agents/safety/validate`
  Request: `prompt`, optional `model`.
  Returns: `success`, `is_safe`, `reason`, `error`.
  If no safety client is configured, the service may return `reason: "skipped"`.
- `POST /agents/interpret-input`
  Request: `user_input`, `scenarios`, optional `model`.
  Returns: `success`, `action`, `feedback`, `error`.

## Tests

- `POST /tests/run`
  Request: `test_file`, `codebase_path`.
  Returns: `success`, `total`, `passed`, `failed`, `coverage_percent`, `output`, `error`.
- `POST /tests/parse-output`
  Request: `output`.
  Returns: `total`, `passed`, `failed`.
- `POST /tests/coverage`
  Request: `codebase_path`, optional `coverage_json_path`.
  Returns: `success`, `total_percent`, `files`, `error`.
  If `coverage_json_path` is omitted, the API reads `<codebase_path>/coverage.json`.
- `POST /tests/validate-syntax`
  Request: `code`.
  Returns: `success`, `is_valid`, `errors`.

## Codebase

- `POST /codebase/analyze`
  Request: `codebase_path`, optional `include_hidden`.
  Returns: `success`, `files`, `total_files`, `total_lines`, `by_extension`, `error`.
- `POST /codebase/files`
  Request: `path`, optional `extensions`, optional `recursive`.
  Returns: `success`, `files`, `total`, `error`.

## Prompts

- `GET /prompts/history`
  Returns discovered prompt runs from common output directories:
  `success`, `runs`, `total`, `error`.
- `GET /prompts/{run_id}`
  Returns `success`, `run_id`, `prompts`, `error`.

## Utilities

- `POST /utils/extract-dependencies`
  Request: `test_code`.
  Returns: `packages`.
- `POST /utils/install-dependencies`
  Request: `packages`, `cwd`.
  Returns: `success`, `installed`, `failed`.
- `POST /utils/parse-log`
  Request: `line`.
  Returns parsed UI metadata such as `phase_update`, `coverage`, `tests`, and
  `agent_activation`.
- `GET /utils/models`
  Returns `{ "models": [...], "default": "..." }`.

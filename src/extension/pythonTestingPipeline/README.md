# Python Automated Testing Pipeline

Current docs for the CLI, API, GUI, and extension-facing parts of the pipeline.

## Flow

```text
Identify -> Approve or refine -> Implement -> Run tests -> Evaluate
                                      ^                         |
                                      |------ improve loop -----|
Artifacts: tests, prompts, report, governance, coverage report
```

## Entry Points

- CLI
  Run:
  `python src/extension/pythonTestingPipeline/scripts/pythonTestingPipeline.py <codebase_path>`
  Common options: `--auto-approve`, `--no-run-tests`, `--output-dir`, `--model`.
  `--coverage` and `--run-tests` are compatibility flags.
- API
  Run:
  `uvicorn src.extension.api.main:app --reload`
- GUI
  Run:
  `python -m src.extension.GUI.main`
- VS Code extension
  The command palette action is `Agentic Testing: Generate Tests`
  (`agentic-testing.generateTests`).
  It currently handles folder selection and progress UI, but it does not yet run
  the full end-to-end pipeline.
- Internal model tools
  `generatePythonTests`, `implementPythonTests`, and `evaluatePythonTests` are
  internal tools, not public slash commands.

## Outputs

Default output location: `<codebase_path>/tests`, unless `--output-dir` is set.

Generated artifacts may include:
- `test_generated_<timestamp>.py`
- `prompts_<run_id>.json`
- `report_<run_id>.md`
- `governance_<run_id>.json`
- `coverage_report_<run_id>.json`

## Runtime Notes

- Python 3.10+ is the practical baseline for the code in `src/extension`.
- Pipeline-related Python dependencies live in `requirements.txt`.
- `scripts/llm_config.py` is the source of truth for model ordering.
- Current model selection prefers Ollama-hosted models first, then Groq-backed
  fallbacks.
- `GROQ_API_KEY`, `GROQ_API_KEY_1`, and similar variables are rotated when the
  client needs another key.
- Safety checks are implemented in `scripts/prompt_safety.py`.

## Current Caveats

- Coverage is effectively always collected when generated tests are executed.
- API prompt-history discovery can legitimately return no runs.
- API pipeline status is in-memory only.
- The Python CLI is the most complete execution path today.

## Keep In Sync

When updating docs here, cross-check:
- `scripts/pythonTestingPipeline.py`
- `scripts/llm_config.py`
- `src/extension/api/main.py`
- `src/extension/api/schemas.py`
- `package.json`
- `src/extension.ts`

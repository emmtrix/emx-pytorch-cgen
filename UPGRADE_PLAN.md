# Upgrade Plan: Bring emx-pytorch-cgen to emx-onnx-cgen maturity (docs nearly identical)

This repo (`emx-pytorch-cgen`) currently contains some ONNX support (`cli/onnx2c.py`, `tests/test_onnx2c_golden.py`, `onnx2pytorch` dependency). The target state is:

1. Remove ONNX support from this repo (it lives in `emx-onnx-cgen`).
2. Upgrade this repo's maturity to match `emx-onnx-cgen` in tooling, CI, docs, and verification discipline.
3. Make documentation structure and style **nearly identical** to `emx-onnx-cgen` (adjusted for PyTorch instead of ONNX).

Reference implementation for maturity and docs: `emx-onnx-cgen-org/` (submodule).

## Ground rules for multi-agent execution

- One Codex instance should execute **one step** (or two explicitly marked "small paired" steps).
- Each step must end with:
  - Summary of changes
  - Tests run (if applicable) including duration
  - Next step ID to pick up
- Prefer ASCII-only docs in this repo to avoid Windows encoding issues.

## Local commands (from `AGENTS.md`)

- Run tests: `PYTHONPATH=src pytest -n auto --maxfail=5 -q`
- Update golden refs (intentional output changes): `UPDATE_REFS=1 PYTHONPATH=src pytest -n auto --maxfail=5 -q`
- Documentation-only `*.md` changes do not require tests.

---

# A. Feature-gap inventory (what ONNX repo has that PyTorch repo should mirror)

This section is used to identify missing features and to keep the plan honest. Each item cites where it exists in the ONNX repo and what we should do here.

## A1. Developer experience and docs (structure parity)

- ONNX repo:
  - `emx-onnx-cgen-org/README.md`
  - `emx-onnx-cgen-org/DEVELOPMENT.md`
  - `emx-onnx-cgen-org/docs/output-format.md`
  - `emx-onnx-cgen-org/SUPPORT_OPS.md`
  - `emx-onnx-cgen-org/ONNX_SUPPORT.md` (large support matrix)
  - `emx-onnx-cgen-org/ONNX_ERRORS_HISTOGRAM.md` (error taxonomy + counts)
- This repo:
  - `README.md`, `ARCHITECTURE.md`, no `docs/` directory, no `DEVELOPMENT.md`
  - operator coverage exists but is fragmented (`tests/list_*_ops_ref.md`, `tests/test_codegen_ops.py`)
- Target:
  - Mirror the ONNX doc set and structure, with PyTorch-specific content:
    - `README.md`
    - `DEVELOPMENT.md`
    - `docs/output-format.md`
    - `SUPPORT_OPS.md`
    - `PYTORCH_SUPPORT.md` (analog to `ONNX_SUPPORT.md`)
    - `PYTORCH_ERRORS_HISTOGRAM.md` (analog to `ONNX_ERRORS_HISTOGRAM.md`)

## A2. Tooling (lint/format) and config centralization

- ONNX repo:
  - `emx-onnx-cgen-org/pyproject.toml` (Ruff config, packaging config)
  - `emx-onnx-cgen-org/requirements-ci.txt` includes `ruff`, `black`
  - `.github/workflows/tests.yml` runs `ruff check src tests`
- This repo:
  - `setup.cfg`/`setup.py` packaging
  - no ruff/black in CI
- Target:
  - Add Ruff + Black and run them in CI
  - Consolidate config in `pyproject.toml` (even if we keep `setup.cfg` initially)

## A3. CI discipline (xdist, golden refs, auto-updating expectations)

- ONNX repo:
  - `.github/workflows/tests.yml` runs `pytest -n auto -q`
  - PRs run with `UPDATE_REFS=3` and optionally auto-commit updated references
  - submodule init in CI (recursive)
- This repo:
  - `.github/workflows/test.yml` runs `pytest -q` (no xdist, no PYTHONPATH=src)
  - golden refs exist and use `UPDATE_REFS`, but no CI "update refs" mode
- Target:
  - Align CI with `AGENTS.md` command and ONNX repo ref update workflow

## A4. Determinism and end-to-end verification harness

- ONNX repo:
  - `emx-onnx-cgen-org/src/emx_onnx_cgen/determinism.py`
  - `emx-onnx-cgen-org/src/emx_onnx_cgen/verification.py`
  - CLI `verify` with structured reporting and ULP-based diffs in `emx-onnx-cgen-org/src/emx_onnx_cgen/cli.py`
  - output format spec explicitly mentions determinism and stable ordering
- This repo:
  - strong golden tests for generated code, but no standardized "verify generated C vs eager" test suite
  - determinism is implied, not formalized
- Target:
  - Add a small end-to-end verification harness for PyTorch graphs:
    - generate C
    - compile + run
    - compare results to PyTorch eager with clear tolerances (ULP/abs)
  - Formalize determinism expectations and test them

## A5. Packaging maturity (PEP517 build backend, build info, PyPI README links, release workflows)

- ONNX repo:
  - `emx-onnx-cgen-org/build_backend.py` writes build info and rewrites relative links for PyPI README
  - `emx-onnx-cgen-org/pyproject.toml` uses setuptools_scm and custom build backend
  - `.github/workflows/release.yml` publishes to PyPI on release
- This repo:
  - minimal `setup.py` and `setup.cfg`, no release automation
- Target:
  - Migrate to `pyproject.toml` and adopt optional build info
  - Add release workflow if publishing is desired

## A6. Test performance observability

- ONNX repo:
  - `emx-onnx-cgen-org/tools/pytest_speed_report.py`
  - `emx-onnx-cgen-org/reports/pytest_speed_report.md`
- This repo:
  - no speed report tooling
- Target:
  - Add a simple tooling path to produce and store a speed report markdown (optional but helpful)

---

# B. Docs parity contract (must be "nearly identical")

This is the contract for doc parity. The goal is that a reviewer can open ONNX docs and PyTorch docs side-by-side and see the same structure, headings, and style, with only domain-specific substitutions.

## B1. README structure parity checklist

PyTorch `README.md` should follow the same outline as ONNX `README.md`:

- Title + badges
- One-paragraph "what it does"
- Goals / Non-goals
- Features (bullet list, same style)
- Installation (pip and optional extras)
- Quickstart (compile/export + verify)
- Usage scenarios (embedded vs host, large weights, etc.) adapted for PyTorch
- Links to:
  - `docs/output-format.md`
  - `DEVELOPMENT.md`
  - `SUPPORT_OPS.md`
  - `PYTORCH_SUPPORT.md`
  - `PYTORCH_ERRORS_HISTOGRAM.md` (if used)

## B2. DEVELOPMENT.md structure parity checklist

PyTorch `DEVELOPMENT.md` should mirror ONNX `DEVELOPMENT.md`:

- Prerequisites
- Repo overview
- Setup
- Common workflows (run CLI locally, compile/export, verify)
- Testing (quick targeted, full suite)
- Updating golden references
- Submodules (if any are used by tests; otherwise explicitly state "none")
- Formatting and linting (ruff/black)

## B3. Output format spec parity checklist

`docs/output-format.md` should mirror ONNX `docs/output-format.md`:

- Artifacts
- High-level file layout (deterministic ordering, naming scheme)
- Public C API
- Tensor representation (dtype mapping, shapes, VLA policy)
- Constants/weights representation (inlined vs external, if applicable)
- Optional: testbench layout (if we provide one)

## B4. Support tracking parity checklist

- `SUPPORT_OPS.md` should be a table like ONNX's:
  - Operator name
  - Supported marker
  - Optional notes column (PyTorch-specific constraints)
- `PYTORCH_SUPPORT.md` should be the long-form matrix:
  - Track a curated model/file suite and expected outcomes
  - Include the verification tolerance policy and failure taxonomy
- `PYTORCH_ERRORS_HISTOGRAM.md` should track top error messages (counts + categories)

---

# C. Step-by-step upgrade plan (atomic steps)

The steps are designed to be executed in order, because later steps assume ONNX removal and stable tooling.

## Phase 0: Remove ONNX from this repo (required)

### P0-01 Remove ONNX CLI

- Scope:
  - Delete `cli/onnx2c.py`
  - Remove ONNX references from `README.md` and `ARCHITECTURE.md`
- Commands:
  - `rg -n "onnx2c|onnx2pytorch|\\bonnx\\b" -S README.md ARCHITECTURE.md cli src tests requirements-ci.txt setup.cfg`
- Done when:
  - No ONNX CLI remains; docs no longer mention it.
- Tests:
  - `PYTHONPATH=src pytest -n auto --maxfail=5 -q`

### P0-02 Remove ONNX tests and test data

- Scope:
  - Delete `tests/test_onnx2c_golden.py`
  - Delete `tests/onnx2c/`
  - Delete `tests/onnx2c_refs/`
  - Remove any README references to those directories
- Done when:
  - Test suite runs without importing `onnx` or `onnx2pytorch`.
- Tests:
  - `PYTHONPATH=src pytest -n auto --maxfail=5 -q`

### P0-03 Remove ONNX dependencies from requirements

- Scope:
  - Update `requirements-ci.txt` to remove `onnx2pytorch==...` (and any ONNX-only pins if present)
  - Update docs to remove ONNX optional dependency mentions
- Done when:
  - `pip install -r requirements-ci.txt` no longer requires ONNX.
- Tests:
  - `PYTHONPATH=src pytest -n auto --maxfail=5 -q`

## Phase 1: Test runner hygiene and CI alignment

### P1-01 Add pytest.ini (match ONNX behavior)

- Scope:
  - Add `pytest.ini`:
    - `testpaths = tests`
    - `norecursedirs = *-org`
- Done when:
  - `pytest` does not recurse into submodules; test discovery is stable.
- Tests:
  - `PYTHONPATH=src pytest -n auto --maxfail=5 -q`

### P1-02 Align CI test command with AGENTS.md

- Scope:
  - Update `.github/workflows/test.yml`:
    - set `PYTHONPATH=src`
    - run `pytest -n auto --maxfail=5 -q`
- Done when:
  - CI command matches local recommended command.

### P1-03 Add golden-ref update mode in CI (PRs)

- Scope:
  - Update `.github/workflows/test.yml`:
    - on `push`: run normal tests
    - on `pull_request`: run with `UPDATE_REFS=1` (or a repo-specific policy)
    - optionally auto-commit updated refs to PR branch (same-repo PRs only)
- Done when:
  - Golden refs drift is handled by a standard workflow (like ONNX repo).

## Phase 2: Tooling parity (Ruff/Black) and config centralization

### P2-01 Introduce Ruff + Black (requirements + config)

- Scope:
  - Add `pyproject.toml` with:
    - `[tool.ruff]` (exclude `*-org` directories, mirror ONNX style)
    - `[tool.black]`
  - Update `requirements-ci.txt` to include `ruff` and `black`
- Done when:
  - `ruff check src tests` and `black --check .` work locally.
- Tests:
  - `PYTHONPATH=src pytest -n auto --maxfail=5 -q`

### P2-02 Run Ruff/Black in CI

- Scope:
  - Update `.github/workflows/test.yml`:
    - `ruff check src tests`
    - `black --check .`
- Done when:
  - CI blocks on lint/format violations.

## Phase 3: Docs parity (nearly identical) and support tracking

### P3-01 Rewrite README.md to mirror ONNX README structure (PyTorch content)

- Scope:
  - Update `README.md` to follow ONNX README outline and style.
  - Ensure links to the new doc set.
- Done when:
  - Side-by-side comparison: headings and section order are nearly identical.
- Tests:
  - none required (docs-only)

### P3-02 Add DEVELOPMENT.md (mirror ONNX DEVELOPMENT.md structure)

- Scope:
  - Add `DEVELOPMENT.md` with the same flow and tone as ONNX repo.
  - Include:
    - recommended local CLI/workflows for PyTorch
    - test commands (including golden ref updates)
    - lint/format commands
- Done when:
  - New contributors can follow DEVELOPMENT.md without additional context.

### P3-03 Add docs/output-format.md (mirror ONNX output format spec)

- Scope:
  - Add `docs/output-format.md` aligned to ONNX spec headings, but describing:
    - `export_generic_c` artifacts and layout
    - temp allocation policy (stack vs heap threshold)
    - naming scheme and determinism guarantees
    - optional testbench strategy (if implemented)
- Done when:
  - The output spec is precise enough to be used as acceptance criteria for golden tests.

### P3-04 Add SUPPORT_OPS.md (table format like ONNX)

- Scope:
  - Add `SUPPORT_OPS.md` as a table (no unicode checkmarks; use ASCII: "YES"/"NO").
  - Populate from a reproducible source:
    - either generated from `tests/test_codegen_ops.py` coverage, or a dedicated script.
- Done when:
  - Operator support is reviewable and can be updated intentionally.

### P3-05 Add PYTORCH_SUPPORT.md (analog to ONNX_SUPPORT.md)

- Scope:
  - Add `PYTORCH_SUPPORT.md` that tracks a curated suite of PyTorch graphs/models.
  - For each model/case:
    - version info (torch version)
    - supported: YES/NO
    - error (if NO)
    - verification policy (tolerance, ULP rules)
- Done when:
  - There is a single "official support matrix" document, like ONNX repo.

### P3-06 Add PYTORCH_ERRORS_HISTOGRAM.md (analog to ONNX_ERRORS_HISTOGRAM.md)

- Scope:
  - Add `PYTORCH_ERRORS_HISTOGRAM.md` with:
    - error message
    - count
    - optional category / op names
  - Start manually; later steps may automate generation.
- Done when:
  - Top failure modes are visible and trackable.

## Phase 4: End-to-end verification and determinism

### P4-01 Add end-to-end verification harness (PyTorch eager vs generated C)

- Goal:
  - Mirror ONNX repo's "verify" discipline in a PyTorch-appropriate way.
- Scope (proposed):
  - Add a small verification module, e.g. `src/codegen_backend/verification.py`
  - Add tests, e.g. `tests/test_verification_e2e.py`
  - Policy:
    - deterministic seeds
    - float comparisons: "ignore tiny diffs up to epsilon" then ULP distance (mirror ONNX wording)
    - integer/bool exact match
- Done when:
  - At least a handful of representative graphs are verified end-to-end on CI.
- Tests:
  - `PYTHONPATH=src pytest -n auto --maxfail=5 -q`

### P4-02 Formalize determinism guarantees and test them

- Scope:
  - Document determinism in `docs/output-format.md` (and/or a dedicated doc)
  - Add a test asserting:
    - same graph + inputs -> identical generated source bytes
  - Fix any non-deterministic iteration/orderings found in codegen.
- Done when:
  - Determinism is explicit and protected by tests.

## Phase 5: Packaging and release maturity (optional, if you want parity)

### P5-01 Migrate packaging to pyproject.toml (PEP517)

- Scope:
  - Add PEP517 `pyproject.toml` build-system config (keep `setup.cfg` temporarily if needed).
  - Ensure template package-data remains correct (`templates/*.j2`).
- Done when:
  - `pip install -e .` works and CI installs cleanly.

### P5-02 Add build info (optional parity with ONNX build_backend.py)

- Scope:
  - Add a small build backend or build hook that writes:
    - build date
    - git short SHA
  - (Optional) rewrite relative README links for PyPI the way ONNX does.
- Done when:
  - Wheels include build info and PyPI README renders correctly.

### P5-03 Add release workflow (optional)

- Scope:
  - Add `.github/workflows/release.yml` to publish on GitHub releases.
- Done when:
  - One-button release is possible.

## Phase 6: Test performance observability (optional)

### P6-01 Add pytest speed report tooling

- Scope:
  - Add `tools/pytest_speed_report.py` (or port minimal subset from ONNX repo)
  - Add `reports/pytest_speed_report.md` generation target (gitignored or committed, decide policy)
- Done when:
  - It is easy to identify slow tests and regressions.

---

# D. Recommended execution order

1. Phase 0 (remove ONNX from this repo)
2. Phase 1-2 (pytest.ini, CI alignment, Ruff/Black)
3. Phase 3 (docs parity + support tracking)
4. Phase 4 (end-to-end verify + determinism)
5. Phase 5-6 (packaging/release, speed reports) as needed


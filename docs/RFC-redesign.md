# RFC: Redesign of the portfolio-analyzer package

Status: Draft — feedback requested.

Target package: `src/portfolio_analyzer`

## 1. Summary

Propose a modular, well-typed, plugin-friendly redesign of `portfolio-analyzer` to improve maintainability, testability, extensibility, and performance. Key changes:

- Clear layer separation (api/core/data/analysis/reporting/interactive).
- Well-defined interfaces (ABCs) for pluggable components (data sources, return estimators, optimizers, reporters).
- Stronger typing and runtime validation for public APIs.
- Improved packaging (entry points/CLI), configuration, CI, and docs.
- Migration plan, tests, and acceptance criteria.

## 2. Motivation & goals

Problems observed and goals:

- Monolithic modules: several responsibilities in single modules make testing and extension hard.
- Inconsistent public API and unclear contracts between components.
- Weak plugin/strategy support for return estimators and optimizers.
- Limited automated checks and packaging ergonomics.
- Notebooks and CLI/interactive code mixed with core logic.

Goals:

- Enforce single responsibility boundaries and small surface APIs.
- Make return estimation and optimization pluggable via ABCs + entry points.
- Add static typing, runtime validation, and a stable public API with semantic versioning rules.
- Add CI pipeline with tests, type checks, formatting, and packaging.
- Provide migration path and compatibility layer for existing code/users.

## 3. Scope

In-scope:

- Package/module layout and public APIs.
- Interfaces for pluggable components.
- Configuration strategy (env + config file + defaults).
- Tests, CI, docs, and packaging/entry points.

Out-of-scope:

- Rewriting numerical algorithms (except to adapt them to new interfaces).

## 4. Current state (high-level)

Key folders: `analysis/`, `config/`, `core/`, `data/`, `reporting/`, `return_estimator/`, `interactive/`.

Observed patterns:

- `return_estimator` contains multiple implementations; formal interface needs verification and standardization.
- `core/optimizer.py` and `core/objectives.py` contain optimizer logic that is tightly coupled to data and estimators.
- `data/` mixes fetching, preparing, and caching.

## 5. Problems & Evidence (concrete)

1. Lack of explicit interfaces/ABCs for strategies.
2. Layer coupling between data, analysis, and optimization.
3. Type/validation gaps for DataFrame inputs/outputs and public models.
4. No plugin discovery or formal entry points for custom estimators/optimizers.
5. Tests exist but CI, type checks, and linters are not enforced in repo root.
6. Documentation is limited to notebooks without a canonical API reference.
7. Config sourcing and precedence are not standardized.

## 6. Proposed Design (high level)

6.1 Module/Layers (explicit)

- `portfolio_analyzer/` (public package)
  - `api/` — thin public API wrappers (stable surface)
  - `core/` — core domain logic (typing, contracts, core algorithms)
    - `interfaces.py` — ABCs for DataSource, ReturnEstimator, Optimizer, Reporter
    - `models.py` — typed dataclasses (PortfolioSpec, AssetUniverse, ReturnsFrame)
  - `data/` — data adapters and caching
    - `providers/` — adapters for external sources
    - `schema.py` — dataframe schemas + validators
  - `analysis/` — flows (backtest, monte_carlo, frontier)
  - `estimators/` — return estimation implementations (each in its own module)
  - `reporting/` — exporters, visualizers, summaries
  - `interactive/` — REPL/session wrappers that call into `api`
  - `config/` — centralized config loader (env + file + defaults)
  - `cli.py` — console entry points

Rationale: split responsibilities, make small modules easier to test and document.

6.2 Contracts (examples)

- BaseReturnEstimator (signature):
  - inputs: AssetUniverse (list[str]), HistoricalReturns (pd.DataFrame indexed by date)
  - outputs: ExpectedReturns (pd.Series indexed by asset), CovarianceMatrix (pd.DataFrame)
  - behavior: deterministic for fixed seed; supports optional `fit(X)` and `estimate()`.

- DataSource:
  - `fetch(asset_list, start, end) -> pd.DataFrame`
  - `available_assets() -> list[str]`

- Optimizer:
  - `optimize(spec: PortfolioSpec, expected_returns, covariance) -> PortfolioResult`

Define these types in `core/models.py` as dataclasses and export them.

6.3 Typing & Validation

- Add type hints for public APIs.
- Use `pydantic` or dataclasses + validators for config models.
- Use `pandera` or custom validators for DataFrame schema checks.

6.4 Pluggability & Plugin Discovery

- Use setuptools entry points (e.g., `portfolio_analyzer.return_estimators`) and document how to register.
- Load external implementations lazily via entry points for extensibility.

6.5 Public API & Deprecation

- Provide `portfolio_analyzer.api` with stable functions: `run_backtest()`, `optimize_portfolio()`, `estimate_returns()`.
- Mark internal modules as private and export a small surface.
- Add deprecation wrappers to redirect old API calls to new ones with warnings.

6.6 CLI & Interactive

- Provide a `pa` CLI entry point with subcommands: `fetch`, `estimate`, `optimize`, `backtest`, `report`.
- Keep notebooks as examples, not sources of core logic.

6.7 Packaging & Distribution

- Add `pyproject.toml` entry points for plugins and CLI.

6.8 Testing & CI

- Add GitHub Actions workflow that runs pre-commit (black/ruff/isort), pytest, mypy, and coverage.
- Improve tests with fixtures and integration tests using small datasets.

6.9 Docs

- Add `docs/` using MkDocs or Sphinx with API reference, plugin guide, and migration guide.

## 7. Detailed API sketches (short)

- `core/interfaces.py` (conceptual):
  - `class BaseReturnEstimator(ABC):`
    - `def fit(self, prices: pd.DataFrame) -> None: ...`
    - `def estimate(self) -> Tuple[pd.Series, pd.DataFrame]: ...`

- `core/models.py`:
  - `@dataclass PortfolioSpec(assets: List[str], constraints: Dict[str, Any], objective: str)`
  - `@dataclass PortfolioResult(weights: pd.Series, metrics: Dict[str, float])`

## 8. Migration plan & compatibility

Phased approach:

- Phase 0 — discovery (0.5 week): add RFC and run tests for baseline.
- Phase 1 — interfaces & models (1 week): implement `core/interfaces.py` and `core/models.py`; add tests.
- Phase 2 — data layer & validators (1 week): implement `data/schema.py`, migrate `data_fetcher.py` into `data/providers/`.
- Phase 3 — estimators & optimizer decoupling (1–2 weeks): extract estimators and implement plugin loader.
- Phase 4 — api, cli, docs, tests, CI (1 week): create API layer, CLI, docs, and CI.
- Phase 5 — release & cleanup (0.5 week): deprecation warnings, tag release.

Backward compatibility:

- Provide temporary compatibility adapters in `compat/` that wrap old functions and emit DeprecationWarning. Remove after one major release.

## 9. Acceptance criteria

- Public functions in `src/portfolio_analyzer/api` are typed and documented.
- Tests pass and coverage is not decreased.
- CI workflow is configured and green for PRs (mypy, ruff, pytest).
- At least one estimator and the optimizer use the new interfaces.
- Plugin discovery via entry points works with an example plugin.

## 10. Risks & mitigations

- Risk: refactor introduces regressions. Mitigation: incremental changes, tests, CI.
- Risk: breaking third-party users. Mitigation: compatibility shims and deprecation schedule.
- Risk: performance regressions. Mitigation: benchmark and optimize hot paths.

## 11. Alternatives considered

- Keep current layout and only add tests/docs — rejected: insufficient extensibility.
- Add plugin system without typing — rejected: harder to guarantee correctness.

## 12. Implementation tasks (first PR)

- [ ] Add `core/interfaces.py` and `core/models.py` with typed ABCs and dataclasses.
- [ ] Add unit tests for interface compliance.
- [ ] Add `data/schema.py` with DataFrame validators.
- [ ] Add `api/__init__.py` exposing `optimize_portfolio`.
- [ ] Add GH Actions workflow: test + lint + type check.
- [ ] Add `docs/` skeleton and update README.
- [ ] Add entry_points in `pyproject.toml` for CLI.

## 13. Quality gates (quick triage)

- Build: verify package imports cleanly.
- Lint/Typecheck: `ruff`/`mypy` config added; new code must pass.
- Tests: run `pytest` and add interface tests.
- Smoke test: run `optimize_portfolio` on a fixture and confirm sane output.

## 14. Appendix: Example migration snippet (conceptual)

- Create `core/interfaces.py` and make existing estimators subclass `BaseReturnEstimator`. Add adapter wrappers to preserve import paths.

## 15. Next steps

- I can implement Phase 1 now: create `core/interfaces.py` and `core/models.py` and run the tests to establish a baseline. Ask me to proceed when ready.

---

### Requirements coverage

- Create `docs/` directory and add the RFC markdown: Done (this file).

### Completion summary

The RFC has been added to `docs/RFC-redesign.md`. If you'd like, I can now create the first-phase interface files in `src/portfolio_analyzer/core/` and run the test suite to verify no regressions.

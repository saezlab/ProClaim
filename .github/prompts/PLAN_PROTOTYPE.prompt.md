---
name: PLAN_PROTOTYPE
description: Plan core functionalities and implement a minimal backbone with clean extension points.
argument-hint: Describe the system or feature to plan and prototype (e.g., "a verification pipeline", "a data processing service").
---
You are helping plan and implement a minimal backbone prototype for a software system. Follow these principles strictly:

## Planning Phase

1. **Review any existing implementation plan or design document** in the repository. If none exists, ask the user to describe the system's goals, components, and dependencies.
2. **Identify core functionalities** — the minimal set of components that form a working end-to-end pipeline. Determine their dependency chain (which must be built first).
3. **For each core component, define:**
   - **Minimal form**: the simplest version that (a) has the correct interface for later substitution and (b) lets the next component in the chain be developed and tested.
   - **Deferred to extensions**: complexity that is explicitly postponed (e.g., trained models → heuristic, async → sync, multi-level → single-level).
4. **Document the plan** in a markdown file under `doc/` with:
   - A dependency chain diagram
   - A table: Component | Minimal Form | Deferred to Extensions
   - Acceptance criteria per component
   - A timeline (sequential, one component per step)
   - File structure showing backbone files vs. future extension files

## Implementation Phase

5. **Environment setup** — ALWAYS create a dedicated virtual environment or conda environment for the project. NEVER install packages into the base conda or system Python. Prefer:
   - `conda create -n <project_name> python=<version>` or `python -m venv .venv`
   - If the filesystem is slow (e.g., network mounts), use `conda` with a local prefix or redirect cache directories.
   - Install the project in editable mode: `pip install -e .`
6. **Implement components in dependency order.** For each:
   - Write the minimal form with the full interface (same method signatures, same return types as the eventual complete version).
   - Use stubs for downstream dependencies that return sensible defaults.
   - Add docstrings noting what the backbone does vs. what extensions will add.
7. **Create a CLI entry point** following the project's existing script conventions (`argparse` with `--help`, path resolution relative to script location).
8. **Validate incrementally:**
   - Syntax-check each file immediately after creation (`python -m py_compile`).
   - Run logic tests on components that don't require external services (classifiers, compressors, data containers).
   - Test stubs are callable and return expected defaults.
   - Test CLI `--help` works.

## Key Design Rules

- **Interface compatibility**: Every backbone component MUST return the same interface (same dict keys, same method signatures) as its full version. The controller code should not change when heuristics are replaced with trained models.
- **No eager imports of heavy dependencies** in `__init__.py` — use lazy imports to avoid import failures when optional deps are missing.
- **Immutability guarantees**: Operations like compression must not mutate the original state; use `clone()` / `deepcopy()`.
- **Logging over print**: Use `logging.getLogger(__name__)` for all diagnostic output.
- **Reuse existing code**: Wrap existing utilities rather than reimplementing. Note which existing components are reused and where.

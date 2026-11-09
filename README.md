# ArgDebate

ArgDebate is a relation-preserving, route-aware selection policy for
multi-agent debate. The reference implementation extracts arguments and
support and attack relations from debate records, constructs and evaluates a
quantitative bipolar argumentation graph when the graph path is used, and
provides non-graph routes and fallback logic.

## Interactive research platform

[Open the research planning platform and mechanism demonstrations](https://argdebate-graph.app.workbuddy.host/?sharecode=UZOTEP76TA).
The landing page provides a multi-agent research planning workspace and an
ArgDebate argumentation-graph demonstration, among other interactive tools.
The platform is an external application; the code snapshot and historical
evaluation records have the distinct roles described below.

## Code and evidence scope

`code_release/` contains an inspectable reference implementation, comparison
selectors, experiment drivers, configuration examples, and tests. It is not
certified as the exact executable used for every historical evaluation cell:
the cell-specific runtime commit or code hash was not retained. Some routing,
convergence, and relation-handling behavior in this snapshot differs from the
paper-facing protocol. See [Implementation conformance](IMPLEMENTATION_CONFORMANCE.md)
before interpreting or reusing the drivers.

This repository contains code, not the frozen evaluation data or a numerical
replay package. The tests below exercise the snapshot; passing them does not
reproduce the reported study results or establish historical run-to-code
identity.

## Repository layout

- [`code_release/src/argumentation/`](code_release/src/argumentation/): argument
  extraction, quality scoring, graph construction, and propagation.
- [`code_release/src/debate/`](code_release/src/debate/): debate management,
  conflict and deadlock handling, and fallback strategies.
- [`code_release/src/agents/`](code_release/src/agents/): language-model agent
  wrapper.
- [`code_release/src/baselines/`](code_release/src/baselines/): comparison
  selectors.
- [`code_release/experiments/`](code_release/experiments/): dataset drivers,
  experiment utilities, and configuration examples.
- [`code_release/tests/`](code_release/tests/): offline tests and a separate
  live-provider smoke script.

## Installation and offline tests

Python 3.11 was used to verify the offline tests. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd code_release
PYTHONPATH=. python -m pytest -q tests --ignore=tests/test_2agent_resolve.py
```

The test command does not require credentials or model calls. Dependency
installation may require network access. `tests/test_2agent_resolve.py` is a
separate live-provider smoke script and is excluded from this command.

## Configuration for live use

Some experiment drivers require separately supplied datasets. Live-provider
paths require a compatible model endpoint and credentials, and may make
billable provider calls. For the single-endpoint client, copy the template
inside `code_release/` and set local values:

```bash
cp .env.example .env
```

This template is intended to be copied while the current directory is
`code_release/`. The multi-endpoint registry template and its scope are
described in [`code_release/experiments/config/`](code_release/experiments/config/).
Keep credentials and local endpoint files out of version control.

## Citing the code

Repository: <https://github.com/thunderzhu001/ArgDebate>. For a specific code
version, cite the GitHub URL of the commit used as well as the repository URL.

## License

The code is released under the [MIT License](LICENSE). Third-party libraries,
datasets, model weights, and provider outputs retain their own terms.

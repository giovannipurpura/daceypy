# Contributing to DACEyPy

## Getting set up

DACEyPy is a ctypes wrapper around the DACE C++ core. The compiled core is
**bundled** as prebuilt shared libraries under `daceypy/lib/`, one per
platform, so there is no C++ toolchain to install:

```bash
git clone https://github.com/giovannipurpura/daceypy
cd daceypy
pip install -e ".[test]"
```

The `test` extra pulls in pytest plus the matplotlib and scipy that the
documented examples need; the `lint` extra pins the exact ruff and mypy that
CI uses, so `pip install -e ".[test,lint]"` reproduces the full check set.

Supported Python versions are listed in `pyproject.toml` (currently 3.9 and
later). There is no upper bound on numpy; both ends of the supported range are
exercised in CI.

## Running the checks

Everything CI runs, you can run locally:

```bash
pytest                       # the suite, minus the "veryslow" examples
pytest -m "not slow"         # fast feedback, a few seconds
pytest -m veryslow           # the examples that take tens of minutes
ruff check daceypy tests docs
mypy
```

Examples are executed as smoke tests, so a matplotlib backend is needed;
`MPLBACKEND=Agg` keeps them headless.

## What a good contribution looks like

### Tests

New behaviour needs tests, and so does a bug fix — a fix without a test that
fails before it is not finished. The suite distinguishes two kinds:

- **Value tests** ask whether the computed number is right, ideally against a
  closed form known independently of DACEyPy.
- **Contract tests** ask whether the shape, type, scale, frame or
  interpretation of a result is the one the interface promises. A number can
  be correct while the contract around it is wrong, and a round trip that
  "closes" will not catch that.

Most of the bugs that took longest to find in this project were contract bugs.

The test environment imposes two constraints:

- The DACE core keeps the polynomial order and the number of variables in
  **global** state. Re-initialising invalidates every DA object created
  before, so tests must not share DA objects across a `DA.init`. Use the
  fixtures in `tests/conftest.py`.
- Some code paths need `DA.getTO() < DA.getMaxOrder()`, because they estimate
  the norm of the terms one order above the truncation order.

### Style

`ruff` and `mypy` run in CI and must pass. Three things to know, the first
two encoded in `pyproject.toml`:

- Import order is **load-bearing** in `daceypy/__init__.py` (the submodules
  import from the partially initialised package) and in the examples (which
  must run `daceypy_import_helper` before importing `daceypy`). Do not sort
  imports there.
- `mypy` exempts seven modules that carry an annotation backlog. Removing a
  module from that list is welcome. Adding one is not.
- `mypy` is sensitive to the **numpy version**, because numpy ships its own
  stubs: code that type-checks against numpy 1.x can fail against 2.x. CI
  installs the newest numpy for the lint job, so run `mypy` against a recent
  numpy before pushing, not only against whatever the working environment has.

### Commits

Write commit messages that explain **why**, not just what. If a change looks
wrong but is deliberate, say so in the message and leave a comment in the
code — see `ADS.canSplit` for an example of a non-strict comparison that
exists on purpose and has been "fixed" by mistake before.

## How pull requests are handled

Open a pull request against `master`.

- Contributions are reviewed against the numbers they produce. Code that
  produces values used downstream gets read line by line; code that only
  produces reports or plots is reviewed more lightly, because its errors are
  visible in the output.
- A contributor's PR is **squashed into one or two commits** on `master`, to
  keep the history linear and readable.
- **Authorship is preserved.** The squashed commit keeps the original author
  via `git commit --author=...`, using the contributor's GitHub `noreply`
  address, which is guaranteed to be linked to their profile — the address
  used in the original commits may not be registered on the account.
- If a PR needs substantial rework, the merge is done in a dedicated branch
  with a real `git merge` (not `--squash`), so the original commits remain
  consultable during review, and the squash happens only at the end.

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). Backwards
compatible additions bump the minor version; fixes bump the patch version.
Every user-visible change goes in `CHANGELOG.md`.

Releasing is currently manual: bump `daceypy/_version.py`, update the
changelog, build with `python -m build`, verify with `twine check`, upload to
TestPyPI, then to PyPI, then tag and create the GitHub release. CI builds the
distribution and verifies that the wheel installs into a clean environment on
every push, so a release should never be the first time that is tried.

## The bundled DACE core

`daceypy/lib/README.md` records which upstream DACE commit the bundled
binaries were built from, and how to rebuild them. Changing the core means
rebuilding for **every** bundled platform (Windows x86/x64/ARM64, Linux
i686/x86_64/aarch64, macOS x86_64/ARM64), so it is not a casual change.

## Contributors

DACEyPy is maintained by **Giovanni Purpura**.

The following people have contributed code:

- **Michele Maestrini** — the Adaptive Domain Splitting library and its
  tutorial (#2), online ADS and further ADS features together with the
  integrator type hinting (#5), and a fix for issue #7 (#8). Also reported
  issue #6, the C++-parity behaviour of `ADS.canSplit`.
- **Andrea De Vittori** — the optimised DA propagator, the ADS utilities and
  the optimised ADS integrator (#9, and #12 superseded by #13).
- **Matteo Capitanio** — fix for the auxiliary function `_eliminate` causing
  DA matrix inversion errors (#10).
- **Eremey Valetov** — fix for the typo in the `RK78_DP` A-matrix entry
  `A[10, 6]` (#11).

DACEyPy wraps [DACE](https://github.com/dacelib/dace), the Differential
Algebra Computational Toolbox; see `NOTICE` for attribution of the bundled
core.

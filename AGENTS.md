# AGENTS.md

## Project

`dml` is a small, dependency-free C++20 library for deep learning, written from scratch.
It targets four architectures, in this order of implementation:

1. **SLP** — single-layer perceptron (affine transform + activation)
2. **MLP** — multi-layer perceptron (stack of dense layers trained by backpropagation)
3. **RNN** — vanilla recurrent network, trained by BPTT
4. **LSTM** — long short-term memory, with gated cells

The goal is clarity over performance: this is a teaching/learning library, not a
production ML framework. Only the C++ standard library is required at runtime.
GoogleTest is used for testing and CMake for building.

## Layering

The library is two layers, and dependencies point strictly downward:

- `math/` — `Matrix<T>` and `Vector<T>`. Pure numerics: shape, transpose, elementwise
  operations, matrix multiplication, reductions. Knows nothing about neural networks.
- `nn/` — `Tensor<T>`, `Layer<T>`, `Optimizer<T>`, and the concrete architectures.
  May use `math/`; `math/` must never depend on `nn/`.

`Tensor<T>` wraps a `Matrix<T>` for data and a second `Matrix<T>` for the gradient.
Activations and intermediate values are plain `Matrix<T>`; only trainable parameters
carry a gradient. This keeps `math/` free of any training-related concept.

`include/dml/` is the public API. `include/dml/dml.hpp` is the umbrella header that
pulls in the public surface; include it rather than reaching into individual headers
unless you specifically need one.

## Contracts

These two contracts are the extension points of the library. Getting them right matters
more than any individual layer.

### Layer contract

Every architecture implements:

- `forward(input) -> output` — pure computation. Must not mutate parameters or
  optimizer state.
- `backward(grad_output) -> grad_input` — computes gradients and writes them into each
  parameter's gradient buffer. **Never applies the parameter update itself.**

### Optimizer contract

The parameter *update rule* is a strategy that the user selects. `Optimizer<T>`
implementations own whatever per-parameter state they need (Adam's first and second
moments, momentum buffers, etc.) and expose a way to apply one update across a set of
parameters, plus a way to reset that state.

Keep the two concerns separate:

| Concern                              | Lives in                | Swappable by user |
| ------------------------------------ | ----------------------- | ----------------- |
| Gradient *derivation* (how ∂L/∂W)    | the layer's `backward()` | No                |
| Parameter *update* (how ΔW is applied) | the `Optimizer<T>` impl | Yes               |

A network must work with **any** `Optimizer<T>` implementation. Never hard-code SGD,
Adam, momentum, or any other update rule inside a layer or architecture. Likewise,
never put update logic inside `backward()` — it would break gradient accumulation and
make optimizers impossible to swap.

Gradient derivation is layer-specific math and is not a user-selectable choice. There
is no autodiff engine and no finite-difference fallback in this library; each
architecture derives its own gradients analytically.

## Conventions

- C++20. Concepts, ranges, `std::span`, `std::format`, and `std::numbers` are available.
- All library code lives in `namespace dml`; use `dml::math` and `dml::nn` for the two
  layers.
- Naming: `snake_case` for functions and variables, trailing underscore for data
  members (`w_`, not `m_w` and not `w`). `PascalCase` for types. `kPascalCase` for
  constants.
- Numeric types are floating point (`float`/`double`). `Matrix<T>` and `Vector<T>` are
  constrained with `requires std::floating_point<T>`; integer types are not supported
  for math values.
- Header guards: `#ifndef DML_<PATH>_HPP` … `#endif  // DML_<PATH>_HPP`. The `DML_`
  prefix is required on every guard.
- Public headers are included as `#include <dml/...>`.
- Templates are header-only: implementations live in the header, marked `inline` or
  `constexpr`. Do not add a `.cpp` file to implement a template — `src/` holds only
  non-template translation units.
- Prefer `std::span` over pointer/length pairs, `std::vector` over `new`/`delete`, and
  `std::optional` over sentinel values.

## Comments

Comments explain intent, not mechanics. Never restate what the code already says.

- Comment the *why*: the reason a choice was made, a non-obvious invariant, a
  numerical caveat, a precondition.
- No comment that narrates the next line ("// loop over rows", "// increment i").
- Public API declarations carry a short doc comment. The rest needs none.
- No `// TODO`, `// FIXME`, or placeholder comments. Open an issue instead.
- If a comment stops matching the code, fix or delete it in the same change.

## Formatting

Formatting is enforced by `clang-format` using the checked-in `.clang-format`
(LLVM base style, 2-space indent, 100-column limit). Do not hand-format code, and do
not restate formatting rules in code comments — run the formatter instead:

```bash
cmake --build build --target format     # rewrite files in place
cmake --build build --target check-format # verify only, non-zero exit on drift
```

If `clang-format` is not found, install it with `brew install llvm`. CMake also searches
the usual Homebrew (`/opt/homebrew/opt/llvm/bin`) and Xcode Command Line Tools
(`/Library/Developer/CommandLineTools/usr/bin`) locations, so no `PATH` change is needed
in most cases. CI pins `clang-format` to an exact version via pip; to match it byte for
byte locally, run `pipx install clang-format==23.1.3`.

## Build

Requires CMake >= 3.20 and a C++20 compiler (Apple clang 21 or GCC 12+). The minimum
CMake version is deliberately low: the library must build on the CMake that ships with
Ubuntu 22.04 (3.22) and Debian 12 (3.25), not only on the newest release.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

| Option                 | Default  | Meaning                                     |
| ---------------------- | -------- | ------------------------------------------- |
| `DML_BUILD_TYPE`       | `STATIC` | `STATIC` or `SHARED`                       |
| `DML_USE_TESTS`        | `ON`     | Build the GoogleTest suite                  |
| `CMAKE_BUILD_TYPE`     | empty    | Standard CMake build type (`Release`/`Debug`) |

`CMAKE_EXPORT_COMPILE_COMMANDS` is always `ON`, so `build/compile_commands.json` is
generated and can be pointed at by clangd and other LSP tooling.

## Test-driven workflow

Write the test first, then the code that satisfies it.

1. **Red** — add a test derived from the mathematics or from the contract above, not
   from an implementation you already have in mind. Run it and confirm it fails for
   the *right* reason: missing behavior, not a typo or compile error in the test.
2. **Green** — write the least code that makes it pass. No behavior that no test asks
   for.
3. **Verify** — run the whole suite, not just the new test:
   `ctest --test-dir build --output-on-failure`. It is fast, so always run it.
4. **Repeat** — one behavior per iteration.

- Never weaken an assertion to make a test pass. If a test is wrong, change it
  deliberately and say why. An assertion relaxed to reach green is worse than no test.
- A test that already passes before the implementation exists is not testing anything.
- Anything with gradients gets a finite-difference gradient check (central difference,
  `ε ≈ 1e-5`) next to its expected-value test. It catches sign and factor errors that
  hand-written expectations miss.

## Test

Tests live in `tests/`, one file per unit, named `test_<unit>.cpp`. GoogleTest v1.18.0
is pulled in at configure time via `FetchContent`, so the **first configure requires
network access**; afterwards it is cached in `build/_deps/`.

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run a single suite or test with GTest's naming, which is `Suite.Test`:

```bash
ctest --test-dir build -R TestVector --output-on-failure
```

Tests are registered via `gtest_discover_tests`, so CTest test names match the GTest
suite names exactly.

To add a new unit:

1. Create `tests/test_<unit>.cpp`.
2. Add `<unit>` to `DML_TEST_SUITES` in `tests/CMakeLists.txt`. One line, one file —
   the executable, the link, the warning flags, and the CTest registration are all
   derived from it.
3. Run the suite and make sure it passes.

`enable_testing()` must stay in the **top-level** `CMakeLists.txt`, not in
`tests/CMakeLists.txt`. If it is only in the tests subdirectory, `ctest` run from the
build root reports `No tests were found!!!` while `ctest --test-dir build/tests` works.
The root call is what registers the tests with the top-level CTest file.

Never leave a failing, empty, or skipped test in the tree.

## Git workflow

`main` holds released versions. `dev` holds work in progress. Nothing lands without a
pull request.

```
main     ●──────────────●────────        tags: v0.1.0, v0.2.0
          ↑             ↑
          └─── squash ──┘ release PR
                     ╱
dev        ●──●──●──●──●──●──●──●──●
```

Branches:

- `main` — released code only. Never commit to it directly.
- `dev` — integration branch. All work merges here.
- `feature/<name>` — one change, branched off `dev`.
- `fix/<name>` — one bug fix, branched off `dev`.

Flow:

1. Branch off `dev`: `git switch -c feature/<name> dev`
2. Commit in small steps, keeping the suite green at every commit. Use Conventional
   Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
3. Push and open a PR against `dev`.
4. CI must be green before a PR is merged. Run the same checks locally first — the
   suite is fast.
5. Release: open a PR from `dev` into `main` and squash-merge it, then tag `vX.Y.Z` on
   the merge commit. `main` is protected, so a release is a pull request like any other.

Rules:

- Never commit to `main` or `dev` directly. Always use a feature branch and a PR.
- A merge is accepted only when CI is green.
- Merge feature PRs into `dev` with a merge commit, so the granular history survives.
  Squash-merge release PRs into `main`, so the release line stays readable.
- The local `pre-push` hook runs the same checks as CI. Bypass with `--no-verify` only
  with a stated reason.
- Never force-push to `main` or `dev`.

## Working agreements

- Keep the library dependency-free; do not add third-party runtime dependencies.
- Prefer clear, explicit code over cleverness — this library is meant to be read.
- When a change affects a contract described above, update this file in the same commit.

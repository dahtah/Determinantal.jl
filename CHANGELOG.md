# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Breaking

- Minimum Julia version raised from 1.6 to 1.10.
- `cardinal` renamed to `size_statistics` (same `(mean, std)` return value);
  calls to `cardinal` now error.
- `MarginalDPP(V)` no longer throws when the kernel has eigenvalues > 1;
  they are clipped to 1 with a warning instead.
- `nystrom_approx(K, ind)` now requires `ind::AbstractVector`; new methods take
  raw data and a kernel object: `nystrom_approx(x, ker, ind)` / `(x, ker, m)`.
- Dropped dependencies `LoopVectorization` and `NearestNeighbors`, added
  `NLsolve`; compat bumps `KernelFunctions` 0.10 → 0.11 and `StatsBase` 0.33 → 0.34.
- Removed unexported internals (`rff_fused`, `EllEnsIE`, `condition_inclusion`,
  `isample`, `iesample`); use the exported `condition_on` instead.

### Added

- Partial support for sampling from Hermitian projection ensembles.
- Conditioning of DPPs on inclusion/exclusion of items via the exported
  `condition_on(K; incl, excl)`.
- `match_prob` for rescaling an L-ensemble or marginal DPP to match target
  inclusion probabilities.
- Void probabilities and a moment-generating function for DPP sizes.
- Rescaling to match given inclusion probabilities (`rescale!` improvements).
- Partial support for sampling from Hermitian projection ensembles.
- A Literate.jl example notebook (recommender system built on a DPP).
- Aqua.jl and CompatHelperLocal checks, plus a JET.jl static-analysis test
  driven by a representative workload (`src/workload.jl`).

### Fixed

- Numerical problems in the likelihood computation for marginal kernels.
- Sampling from projection ensembles now uses `Random.default_rng()`, so
  results are reproducible under `Random.seed!`.
- `EllEnsemble` construction no longer errors on matrices whose eigendecomposition
  returns complex values (reported by JET.jl).
- Various documentation and CI fixes.

### Changed

- Explicit imports throughout the package (ExplicitImports.jl compliance).
- Dependencies now use caret-style compat bounds.
- Source and test files are formatted with JuliaFormatter.jl.

## [0.2.0] - 2025

- Initial registered release under the name Determinantal.jl (formerly DPP.jl).

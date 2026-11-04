# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- Partial support for sampling from Hermitian projection ensembles.
- Conditioning of marginal-kernel DPPs (`condition_on`) and the `EllEnsIE` type
  for conditioning L-ensembles on inclusion/exclusion of items.
- `match_prob` / `match_prob_nl` for match probabilities between samples.
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

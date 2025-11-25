# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2025-11-25
- Renamed the `get_bin` helper to `get_bin_indices` for clarity.
- Added `get_differentiable_significance` so applications/examples no longer repeat the same significance logic.
- `restore()` now raises a clear error when no checkpoint is found instead of continuing.
- Improved documentation.
- Made the covariance off-diagonal damping factor configurable via `cov_offdiag_damping`.
- New bump-hunt continuum reweighting helper to handle the sideband fit used for window reweights.

## [0.1.0] - 2025-08-29
- Initial public release.

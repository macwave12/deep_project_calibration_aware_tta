# T03 — O(N) LOO temperature, results-identical

Status: review · Owner: engineering agent · Depends on: —

## Goal
Remove the O(N²) cost of `_estimate_temperature` (it re-scans the whole score history for
381 temperatures every step) without changing any chosen temperature.

## Design
Keep, per grid temperature T, a running sum of `max softmax(s_t / T)` over past samples;
mean confidence = sum / count. The search then costs O(|grid| · C) per step. The LOO
estimate itself is O(M²) in cache size M ≤ 3C, unchanged.

## Acceptance
- Property test on random histories and estimates: the new search returns the same T as
  `utils/loo_temperature.search_temperature` on the full history (exact, ties included;
  document any float-summation tolerance).
- `CalTDA` with temperature on, on a synthetic stream: identical outputs old vs new.
- Old function kept for the test; new path used by `CalTDA`.

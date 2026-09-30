# Status

Last updated: 2026-09-30.

## Completed

### Full corpus dense SAM3 batch
290/291 ok · Tools-excluded mean ~0.96 / median **1.0**.

### AG→Manual targeted re-detect (11 IDs, Manual thr 0.70)
| recording | before → after |
|-----------|----------------|
| LE07UF1_08 | 0.79 → **0.97** |
| LE07UF1_01 | 0.37 → **0.70** |
| ER08ER23_03 | 0.39 → **0.65** |
| KJ03JM25_08 | 0.68 → **0.76** |
| AL07EJ17_05 | 0.72 → **0.78** |
| LE07UF1_04 | 0.75 → 0.77 |
| LE07UF1_06 | 0.22 → 0.27 |
| KI05KO01_02 | 0.39 → 0.42 |
| KJ03JM25_06 | 0.04 → 0.04 (stuck) |
| ER10WE06_06 | 0.18 → 0.18 (stuck) |
| LE07UF1_05 | 0.67 → 0.66 |

Helped several; two hard Manual-over-AG scenes unchanged.

## Current goal
1. `rebuild-qc` to restore full corpus CSV (subset run overwrote it).
2. Accept remaining hard outliers OR drop `instruction manual` on those IDs only.
3. Move to dwell/sequence analysis on the usable majority.

## Not yet
- Safe Tools AOI detector coverage
- Process-step / dwell-transition analysis

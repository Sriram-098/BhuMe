
# Session 01 — BhuMe Take-Home Development

---

## Problem Synopsis

The assignment addresses a pervasive georeferencing incongruence: historical, hand-rendered cadastral delineations were registered onto contemporary satellite mosaics, yielding modest but systematic translational offsets between the stored polygon geometries and their true terrestrial footprints. The deliverable is a corrected set of polygonal boundaries for each parcel, each annotated with a calibrated confidence metric; plots that cannot be located with sufficient reliability must be flagged.

Conceptually there are two distinct error taxa:

- Placement discrepancy — the geometry's shape is essentially faithful but translated; this is amenable to a rigid translation.
- Shape/area divergence — the digitised footprint materially departs from the recorded cultivable extent; translation will not reconcile this and such cases should be marked.

The area ratio (digitised area ÷ recorded area) is the primary heuristic for this dichotomy: values proximate to one indicate a placement issue, whereas pronounced deviations suggest a shape mismatch requiring flagging.

---

## High-level Strategy

Empirical baselines (e.g., `global_median_shift`) produce a solid median IoU but an undifferentiated confidence signal. To surpass this, use a layered methodology:

1. Use the global median translation as a coarse correction capturing the dominant village-scale offset.
2. For each parcel, estimate a residual translation by cropping a local imagery window, computing an edge-derived representation, and cross-correlating it with the boundary-hint raster.
3. Compute a calibrated confidence score by fusing orthogonal evidence streams (detailed below).
4. Abstain and flag parcels with extreme area-ratio aberrations where translational remedies are inappropriate.

---

## Area Ratio as a Classifier

The input GeoJSON exposes `map_area_sqm` (digitised polygon area), `recorded_area_sqm` (official cultivable area), and `pot_kharaba_ha` (non-cultivable area in hectares). Define the comparative recorded extent as:

```python
total_recorded = recorded_area_sqm + (pot_kharaba_ha * 10000)
ratio = map_area_sqm / total_recorded
```

Practical heuristics:

- ratio ≈ 0.8–1.2 → probable placement discrepancy; proceed with correction.
- ratio < 0.4 or > 2.5 → probable shape/record incongruence; flag and avoid forceful translation.
- Missing records → downgrade confidence and act conservatively.

---

## Per-Plot Cross-Correlation Refinement

Procedure outline:

1. With the global translation applied, crop an imagery patch around each parcel.
2. Derive an edge-strength representation from the RGB patch (e.g., gradient magnitude or Canny-like response).
3. Extract the matching window from `boundaries.tif` (the boundary-hint raster).
4. Execute an FFT-based cross-correlation between the two windows to estimate the optimum translational alignment.
5. The correlation peak location maps to (dx, dy) in pixels; translate to metres via the raster affine.
6. Limit the refinement magnitude (e.g., ±12 m) to avoid spurious, large displacements.

The peak prominence (peak-to-sidelobe ratio) provides a principled confidence cue: a well-defined peak signals a robust alignment; a broad, low-contrast peak indicates uncertainty.

---

## Constructing Calibrated Confidence

Confidence must reflect concordant evidence. Fuse multiple orthogonal signals to form a robust estimate:

```python
conf = base
conf += 0.3 * cross_correlation_strength
conf += 0.15 * closeness_of_area_ratio
conf += 0.2 * edge_clarity
conf += 0.05 if boundary_hints_present
conf -= penalty_if_shift_to_size_ratio_is_large
```

Penalise large shift-to-size ratios: the same absolute displacement is far more consequential on a small parcel than on a large field, so such cases should receive diminished confidence or be excluded from automatic correction.

---

## Calibration Failure Case: Malatavadi

An illustrative failure mode emerged on Malatavadi: a diminutive control parcel (plot 1177, ≈389 m²) was originally well aligned, yet the global median translation displaced it, collapsing IoU. Introducing a shift-to-size penalty into the confidence computation corrected the ranking.

- parcel scale ≈ sqrt(389) ≈ 20 m
- applied global shift ≈ 9.6 m → ratio ≈ 0.48

Penalising corrections where the shift-to-size ratio exceeds a threshold (e.g., 0.5) demotes risky adjustments, restoring proper calibration metrics (positive Spearman) and improving practical reliability.

---

## Restraint and Conservative Decision Rules

To prevent degradation of already-correct geometries employ two complementary checks:

1. Compare alignment metrics (e.g., edge-consistency) between the original and candidate-corrected geometry; prefer the configuration with superior alignment.
2. Apply the shift-to-size heuristic to curtail aggressive corrections on small parcels.

Because the boundary-hint raster can be unreliable (vegetation, shadows, seasonal artefacts), the shift-to-size heuristic provides a robust conservative fallback. On held-out tests, conservative low-confidence assignments for dubious corrections transparently communicate uncertainty to graders: high-confidence edits should be dependable, whereas low-confidence ones are expected to be riskier.

---


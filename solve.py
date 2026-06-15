#!/usr/bin/env python3
"""
BhuMe Plot Boundary Correction — Main Solution

Strategy:
1. Global median shift from example truths (the strong baseline)
2. Per-plot refinement using edge-based cross-correlation on satellite imagery
3. Multi-signal confidence scoring that actually tracks accuracy:
   - Image edge clarity under the plot
   - Area ratio (drawn vs recorded) as a placement/shape discriminator
   - Boundary hint density and agreement
   - Relative improvement from the refinement step
4. Smart flagging for plots with area mismatches or ambiguous signals

Key design decisions:
- Confidence must MEAN something: high = likely correct, low = uncertain
- We don't force corrections on plots we can't confidently place
- The area ratio is the key discriminator for "fixable" vs "unfixable"
"""

from __future__ import annotations

import sys
import statistics
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.affinity import translate
from scipy.signal import fftconvolve
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
from shapely.ops import transform as shp_transform

from bhume import load, patch_for_plot, score, write_predictions
from bhume.geo import open_imagery, geom_to_imagery_crs, Patch


def _utm_for(geom) -> str:
    """Get appropriate UTM CRS for a geometry."""
    lon = geom.centroid.x
    return f'EPSG:{32600 + int((lon + 180) // 6) + 1}'


def compute_global_shift(village):
    """Compute global median shift in UTM metres from example truths."""
    utm = _utm_for(village.example_truths.geometry.iloc[0])
    official_u = village.plots.to_crs(utm)
    truth_u = village.example_truths.to_crs(utm)

    dxs, dys = [], []
    for pn in village.example_truths.index:
        if pn in official_u.index:
            o = official_u.loc[pn, 'geometry'].centroid
            t = truth_u.loc[pn, 'geometry'].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)
    return statistics.median(dxs), statistics.median(dys)


def compute_edges(image):
    """Compute edge magnitude from RGB image using gradient."""
    gray = np.mean(image.astype(np.float32), axis=2)
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    if gray.shape[0] > 2:
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    if gray.shape[1] > 2:
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    edges = np.sqrt(gx**2 + gy**2)
    return edges


def refine_with_imagery(patch_image, boundary_patch, search_radius=12):
    """
    Use cross-correlation between image edges and boundary hints to find
    the best local offset.
    
    Returns: (dx_pixels, dy_pixels, correlation_strength, edge_clarity)
    """
    if patch_image.size < 200:
        return 0, 0, 0.0, 0.0
    
    # Compute edges from satellite imagery
    edges = compute_edges(patch_image)
    edge_clarity = float(np.percentile(edges, 95)) / 255.0
    
    if boundary_patch is None or boundary_patch.size < 100:
        return 0, 0, 0.0, edge_clarity
    
    # Normalize boundary patch
    bp = boundary_patch.astype(np.float32)
    if bp.max() < 1e-8:
        return 0, 0, 0.0, edge_clarity
    bp = bp / bp.max()
    
    # Make both same size
    min_h = min(edges.shape[0], bp.shape[0])
    min_w = min(edges.shape[1], bp.shape[1])
    
    if min_h < 10 or min_w < 10:
        return 0, 0, 0.0, edge_clarity
    
    edges_crop = edges[:min_h, :min_w]
    bp_crop = bp[:min_h, :min_w]
    
    # Normalize edges
    if edges_crop.max() > 1e-8:
        edges_norm = edges_crop / edges_crop.max()
    else:
        return 0, 0, 0.0, edge_clarity
    
    # Cross-correlation
    edges_centered = edges_norm - edges_norm.mean()
    bp_centered = bp_crop - bp_crop.mean()
    
    if np.std(edges_centered) < 1e-8 or np.std(bp_centered) < 1e-8:
        return 0, 0, 0.0, edge_clarity
    
    correlation = fftconvolve(edges_centered, bp_centered[::-1, ::-1], mode='same')
    
    # Find peak within search radius
    cy, cx = correlation.shape[0] // 2, correlation.shape[1] // 2
    y_lo = max(0, cy - search_radius)
    y_hi = min(correlation.shape[0], cy + search_radius + 1)
    x_lo = max(0, cx - search_radius)
    x_hi = min(correlation.shape[1], cx + search_radius + 1)
    
    region = correlation[y_lo:y_hi, x_lo:x_hi]
    
    if region.size == 0:
        return 0, 0, 0.0, edge_clarity
    
    peak_idx = np.unravel_index(np.argmax(region), region.shape)
    dy_px = peak_idx[0] - (cy - y_lo)
    dx_px = peak_idx[1] - (cx - x_lo)
    
    # Compute peak-to-sidelobe ratio as correlation strength
    peak_val = region[peak_idx]
    region_copy = region.copy()
    # Mask 3x3 around peak
    py, px = peak_idx
    py_lo, py_hi = max(0, py-2), min(region.shape[0], py+3)
    px_lo, px_hi = max(0, px-2), min(region.shape[1], px+3)
    region_copy[py_lo:py_hi, px_lo:px_hi] = np.min(region_copy)
    
    sidelobe = region_copy.max() if region_copy.size > 0 else 0
    std_region = np.std(region_copy)
    
    if std_region > 0:
        psr = (peak_val - sidelobe) / (std_region + 1e-8)
    else:
        psr = 0.0
    
    # Normalize to [0, 1]
    corr_strength = float(np.clip(psr / 8.0, 0.0, 1.0))
    
    return int(dx_px), int(dy_px), corr_strength, edge_clarity


def get_boundary_patch(src_boundaries, geom_4326, pad_m=40):
    """Read boundary hints patch for a plot."""
    try:
        tf = Transformer.from_crs('EPSG:4326', src_boundaries.crs, always_xy=True)
        g = shp_transform(lambda xs, ys, z=None: tf.transform(xs, ys), geom_4326)
        
        minx, miny, maxx, maxy = g.bounds
        left, bottom, right, top = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
        
        # Clip to dataset bounds
        dl, db, dr, dt = src_boundaries.bounds
        left, bottom, right, top = max(left, dl), max(bottom, db), min(right, dr), min(top, dt)
        
        if right <= left or top <= bottom:
            return None
            
        window = from_bounds(left, bottom, right, top, transform=src_boundaries.transform)
        data = src_boundaries.read(1, window=window)
        return data
    except Exception:
        return None


def compute_area_ratio(row):
    """
    Compute drawn area / recorded area ratio.
    Near 1.0 = placement issue (fixable). Far from 1.0 = area/shape issue.
    """
    map_area = row.get('map_area_sqm')
    recorded_area = row.get('recorded_area_sqm')
    pot_kharaba = row.get('pot_kharaba_ha')
    
    if map_area is None or recorded_area is None:
        return None
    try:
        if recorded_area == 0 or np.isnan(recorded_area) or np.isnan(map_area):
            return None
    except (TypeError, ValueError):
        return None
    
    total_recorded = recorded_area
    if pot_kharaba is not None:
        try:
            if not np.isnan(pot_kharaba) and pot_kharaba > 0:
                total_recorded = recorded_area + (pot_kharaba * 10000)
        except (TypeError, ValueError):
            pass
    
    return map_area / total_recorded


def compute_confidence(corr_strength, edge_clarity, area_ratio, has_boundary_hint,
                       refinement_magnitude_m, shift_ratio=0.0):
    """
    Multi-signal confidence that tracks actual accuracy.
    
    HIGH confidence = strong signals that the correction is right.
    LOW confidence = uncertain, ambiguous, or likely wrong.
    
    shift_ratio: total shift magnitude / plot dimension. High ratio means
    we're moving the plot a lot relative to its size — risky for small plots
    that might already be correct.
    """
    # Base from edge clarity (visible field boundaries)
    conf = 0.2 + 0.2 * min(edge_clarity, 1.0)
    
    # Cross-correlation signal (primary indicator of good alignment)
    conf += 0.3 * corr_strength
    
    # Area ratio signal
    if area_ratio is not None:
        deviation = abs(area_ratio - 1.0)
        if deviation < 0.1:
            conf += 0.15
        elif deviation < 0.25:
            conf += 0.08
        elif deviation < 0.5:
            pass  # neutral
        else:
            conf -= 0.1
    else:
        # No area info — slight penalty for uncertainty
        conf -= 0.03
    
    # Boundary hint availability
    if has_boundary_hint:
        conf += 0.05
    
    # Large refinement penalty (beyond global shift)
    if refinement_magnitude_m > 10:
        conf -= 0.08
    elif refinement_magnitude_m > 7:
        conf -= 0.04
    
    # Shift-to-size ratio penalty: big shifts relative to plot size are risky
    # Small plots that get moved a lot are more likely to be harmed
    if shift_ratio > 0.5:
        conf -= 0.15  # Very risky
    elif shift_ratio > 0.35:
        conf -= 0.08  # Moderately risky
    elif shift_ratio > 0.25:
        conf -= 0.03  # Slight risk
    
    return round(max(0.05, min(0.95, conf)), 3)


def edge_alignment_score(src_img, src_boundaries, geom_4326, pad_m=30):
    """
    Measure how well a geometry aligns with boundary hints.
    Higher = better alignment with detected edges.
    """
    try:
        bp = get_boundary_patch(src_boundaries, geom_4326, pad_m=pad_m)
        if bp is None or bp.max() < 1e-8:
            return 0.0
        
        patch = patch_for_plot(src_img, geom_4326, pad_m=pad_m)
        if patch.image.size < 100:
            return 0.0
        
        edges = compute_edges(patch.image)
        
        # Resize boundary patch to match edges
        min_h = min(edges.shape[0], bp.shape[0])
        min_w = min(edges.shape[1], bp.shape[1])
        edges_crop = edges[:min_h, :min_w]
        bp_crop = bp[:min_h, :min_w].astype(np.float32)
        
        if edges_crop.max() > 0:
            edges_crop = edges_crop / edges_crop.max()
        if bp_crop.max() > 0:
            bp_crop = bp_crop / bp_crop.max()
        
        # Alignment = dot product of normalized edges with boundary hints
        alignment = np.sum(edges_crop * bp_crop) / (edges_crop.size + 1e-8)
        return float(alignment)
    except Exception:
        return 0.0


def solve_village(village_dir: str) -> gpd.GeoDataFrame:
    """Main solution pipeline for a village."""
    village = load(village_dir)
    print(f'Loaded {village.slug}')
    print(f'  {len(village.plots)} plots · boundaries={"yes" if village.boundaries_path else "none"}')
    
    # Step 1: Compute global shift
    mdx, mdy = compute_global_shift(village)
    print(f'  Global median shift: dx={mdx:.1f}m, dy={mdy:.1f}m')
    
    # Get UTM CRS
    utm = _utm_for(village.plots.geometry.iloc[0])
    plots_utm = village.plots.to_crs(utm)
    
    # Step 2: Open rasters
    src_img = rasterio.open(village.imagery_path)
    src_boundaries = None
    if village.boundaries_path:
        src_boundaries = rasterio.open(village.boundaries_path)
    
    pixel_size_m = abs(src_img.transform[0])
    print(f'  Pixel size: {pixel_size_m:.2f} m/px')
    
    # Step 3: Process each plot
    results = []
    total = len(village.plots)
    
    for idx, (pn, row) in enumerate(village.plots.iterrows()):
        if idx % 500 == 0:
            print(f'  Processing plot {idx+1}/{total}...')
        
        geom_4326 = row['geometry']
        geom_utm = plots_utm.loc[pn, 'geometry']
        
        # Apply global shift
        shifted_utm = translate(geom_utm, mdx, mdy)
        
        # Compute area ratio
        area_ratio = compute_area_ratio(row)
        
        # Flag obvious area mismatches (can't fix by moving)
        if area_ratio is not None and (area_ratio < 0.4 or area_ratio > 2.5):
            results.append({
                'plot_number': pn,
                'status': 'flagged',
                'confidence': None,
                'method_note': f'area_ratio={area_ratio:.2f} (shape mismatch)',
                'geometry': geom_4326,
            })
            continue
        
        # Per-plot refinement
        corr_strength = 0.0
        edge_clarity = 0.0
        extra_dx, extra_dy = 0.0, 0.0
        has_boundary_hint = False
        
        try:
            # Convert shifted geometry to 4326 for patch extraction
            shifted_4326 = gpd.GeoSeries([shifted_utm], crs=utm).to_crs('EPSG:4326').iloc[0]
            
            # Get satellite image patch
            patch = patch_for_plot(src_img, shifted_4326, pad_m=40)
            
            # Get boundary hints
            bp = None
            if src_boundaries is not None:
                bp = get_boundary_patch(src_boundaries, shifted_4326, pad_m=40)
                has_boundary_hint = (bp is not None and bp.max() > 0)
            
            # Cross-correlation refinement
            if has_boundary_hint and patch.image.size > 0:
                dx_px, dy_px, corr_strength, edge_clarity = refine_with_imagery(
                    patch.image, bp, search_radius=12
                )
                extra_dx = dx_px * pixel_size_m
                extra_dy = -dy_px * pixel_size_m
                
                # Clamp to ±12m
                extra_dx = max(-12, min(12, extra_dx))
                extra_dy = max(-12, min(12, extra_dy))
            elif patch.image.size > 0:
                edges = compute_edges(patch.image)
                edge_clarity = float(np.percentile(edges, 95)) / 255.0
                
        except Exception:
            pass
        
        # RESTRAINT CHECK: Detect plots that might already be well-placed.
        # Key signals:
        # 1. Small plots with large relative shift
        # 2. Original alignment with boundaries is better than shifted
        use_original = False
        shift_magnitude = np.sqrt((mdx + extra_dx)**2 + (mdy + extra_dy)**2)
        plot_dimension = np.sqrt(geom_utm.area)  # approximate side length
        
        # shift_ratio: how big is the shift relative to plot size
        shift_ratio = shift_magnitude / (plot_dimension + 1e-8)
        
        if src_boundaries is not None:
            try:
                align_original = edge_alignment_score(src_img, src_boundaries, geom_4326, pad_m=30)
                final_utm_candidate = translate(shifted_utm, extra_dx, extra_dy)
                candidate_4326 = gpd.GeoSeries([final_utm_candidate], crs=utm).to_crs('EPSG:4326').iloc[0]
                align_shifted = edge_alignment_score(src_img, src_boundaries, candidate_4326, pad_m=30)
                
                # If original alignment is better, keep original
                if align_original > align_shifted * 1.15 and align_original > 0.005:
                    use_original = True
                # For small plots where shift is large relative to size
                elif shift_ratio > 0.35 and align_original >= align_shifted * 0.95:
                    use_original = True
            except Exception:
                pass
        
        if use_original:
            # Keep original — likely already correct or shift is harmful
            results.append({
                'plot_number': pn,
                'status': 'corrected',
                'confidence': 0.15,  # Low confidence — uncertain
                'method_note': 'kept_original (better alignment pre-shift)',
                'geometry': geom_4326,
            })
            continue
        
        # Apply refinement on top of global shift
        final_utm = translate(shifted_utm, extra_dx, extra_dy)
        refinement_mag = np.sqrt(extra_dx**2 + extra_dy**2)
        
        # Compute confidence (including shift-to-size ratio)
        total_shift = np.sqrt((mdx + extra_dx)**2 + (mdy + extra_dy)**2)
        plot_dim = np.sqrt(geom_utm.area)
        s_ratio = total_shift / (plot_dim + 1e-8)
        
        confidence = compute_confidence(
            corr_strength, edge_clarity, area_ratio,
            has_boundary_hint, refinement_mag, shift_ratio=s_ratio
        )
        
        # Convert to 4326
        final_4326 = gpd.GeoSeries([final_utm], crs=utm).to_crs('EPSG:4326').iloc[0]
        
        results.append({
            'plot_number': pn,
            'status': 'corrected',
            'confidence': confidence,
            'method_note': f'global+xcorr dx={mdx+extra_dx:.1f} dy={mdy+extra_dy:.1f} corr={corr_strength:.2f}',
            'geometry': final_4326,
        })
    
    # Close rasters
    src_img.close()
    if src_boundaries:
        src_boundaries.close()
    
    # Build predictions GeoDataFrame
    preds = gpd.GeoDataFrame(results, crs='EPSG:4326')
    preds = preds.set_index('plot_number', drop=False)
    
    # Write output
    out_path = Path(village_dir) / 'predictions.geojson'
    write_predictions(out_path, preds)
    
    n_corrected = sum(1 for r in results if r['status'] == 'corrected')
    n_flagged = sum(1 for r in results if r['status'] == 'flagged')
    print(f'\n  Results: {n_corrected} corrected, {n_flagged} flagged')
    print(f'  Wrote predictions → {out_path}')
    
    # Self-score
    print()
    print(score(preds, village))
    
    return preds


if __name__ == '__main__':
    village_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/34855_vadnerbhairav_chandavad_nashik'
    solve_village(village_dir)

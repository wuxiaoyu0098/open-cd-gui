# -*- coding: utf-8 -*-
"""
Streaming big GeoTIFF change detection with ChangeDINO (no full-image loading).

Key features:
- Read tif1/tif2 by windows (tiles), no ds.read() of full image
- If grid mismatch, reproject tif2 to tif1 window grid on-the-fly (tile-level reproject)
- Batch inference on GPU, overlap-weighted stitching
- Optional memmap for huge stitched buffers to avoid RAM blow-up
- Postprocess mask (fill holes + remove small objects)
- Polygonize mask to shapefile with area(mu) filtering and contained-polygons removal
- (Optional) second streaming pass to suppress vegetation greening false positives (tile-level ExG)

Author: adapted for your pipeline
"""

import os
import re
import time
import argparse
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes

import geopandas as gpd
from shapely.geometry import shape

from model.create_ChangeDINO import create_model


# ----------------------------
# Utils: weights selection
# ----------------------------
def extract_epoch_from_weight_filename(weight_filename: str):
    """
    从权重文件名中提取 epoch 号，兼容多种命名风格，例如：
    - SECOND_ruian_plus_mobilenetv2_e050_iou06012.pth  -> 50
    - best-model-epoch=000-step=000165-val_iou_1=0.5124.pt -> 0
    - model-epoch-37.pt / epoch_37_xxx.pt -> 37
    """
    # pattern 1: _e050_
    m = re.search(r"_e(\d+)_", weight_filename)
    if m:
        return int(m.group(1))

    # pattern 2: epoch=000 / epoch:000 / epoch-000 / epoch_000
    m = re.search(r"(?:^|[^a-zA-Z])epoch(?:=|:|-|_)?(\d+)", weight_filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    # pattern 3: -ep37- / _ep37_
    m = re.search(r"(?:^|[^a-zA-Z])ep(?:=|:|-|_)?(\d+)", weight_filename, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def _parse_int_set(expr: str):
    if expr is None:
        return None
    expr = str(expr).strip()
    if not expr:
        return None
    out = set()
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = [x.strip() for x in p.split("-", 1)]
            ia, ib = int(a), int(b)
            lo, hi = (ia, ib) if ia <= ib else (ib, ia)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(p))
    return out


from typing import Optional

def _resolve_weight_list(checkpoint_dir: str, weight_args, epochs_expr: Optional[str]):
    epochs = _parse_int_set(epochs_expr)

    # explicit
    if weight_args:
        resolved = []
        for w in weight_args:
            w = str(w).strip()
            if not w:
                continue
            cand = w if os.path.isabs(w) or os.path.sep in w else os.path.join(checkpoint_dir, w)
            if not os.path.isfile(cand):
                raise FileNotFoundError(f"Weight file not found: {cand}")
            resolved.append((cand, os.path.basename(cand)))
        resolved.sort(key=lambda x: x[1])
        return resolved

    # scan dir
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    files = []       # .pth/.pt
    ckpt_files = []  # .ckpt
    for fn in os.listdir(checkpoint_dir):
        if fn.endswith(".pth") or fn.endswith(".pt"):
            files.append((os.path.join(checkpoint_dir, fn), fn))
        elif fn.endswith(".ckpt"):
            ckpt_files.append((os.path.join(checkpoint_dir, fn), fn))
    files.sort(key=lambda x: x[1])
    ckpt_files.sort(key=lambda x: x[1])

    # If there are .pt/.pth use them directly; otherwise fallback to .ckpt candidates.
    candidates = files if files else ckpt_files
    if not candidates:
        raise ValueError(f"No .pth/.pt/.ckpt files found in {checkpoint_dir}")

    if epochs is not None:
        filtered = []
        for p, fn in candidates:
            ep = extract_epoch_from_weight_filename(fn)
            if ep is not None and ep in epochs:
                filtered.append((p, fn))
        if not filtered:
            # help debugging by printing what epochs we can parse
            parsed = []
            for _, fn in candidates:
                e = extract_epoch_from_weight_filename(fn)
                if e is not None:
                    parsed.append(e)
            parsed = sorted(set(parsed))
            raise ValueError(
                f"No weight files matched epochs='{epochs_expr}' under {checkpoint_dir}. "
                f"Parsed epochs from filenames: {parsed[:50]}{'...' if len(parsed) > 50 else ''}"
            )
        candidates = filtered

    # Convert only selected ckpt files (if selected set contains ckpts only)
    need_convert = all(str(fn).endswith(".ckpt") for _, fn in candidates)
    if need_convert:
        tmp_opt = build_opt(checkpoint_dir, name="__ckpt_convert__", gpu_ids=[])
        tmp_model = create_model(tmp_opt)
        out_dir = os.path.join(checkpoint_dir, "_ckpt_converted")
        converted = []
        for p, fn in candidates:
            pt_path = convert_lightning_ckpt_to_pt(p, model=tmp_model, out_dir=out_dir)
            converted.append((pt_path, os.path.basename(pt_path)))
        del tmp_model
        candidates = sorted(converted, key=lambda x: x[1])

    return candidates


def convert_lightning_ckpt_to_pt(ckpt_path: str, model, out_dir: str) -> str:
    """
    Convert Lightning .ckpt (dict with key 'state_dict') into a .pt file compatible with this script:
    dict with key 'network' storing model.model's state_dict.
    Cached under out_dir.
    """
    ckpt_path = str(ckpt_path)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(ckpt_path)
    out_path = os.path.join(out_dir, os.path.splitext(base)[0] + ".pt")
    if os.path.isfile(out_path):
        print(f"[CKPT->PT] Using cached: {out_path}", flush=True)
        return out_path

    print(f"[CKPT->PT] Converting: {ckpt_path} -> {out_path}", flush=True)
    raw = torch.load(ckpt_path, map_location=model.device, weights_only=False)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise ValueError(f"Not a Lightning checkpoint (missing 'state_dict'): {ckpt_path}")
    sd = raw["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Invalid 'state_dict' in ckpt: {ckpt_path}")

    target_sd = model.model.state_dict()
    target_keys = set(target_sd.keys())
    prefixes = [
        "model.",
        "model.model.",
        "model_wrapper.model.",
        "model_wrapper.model.model.",
        "",
    ]

    def _score_prefix(pfx: str) -> int:
        n = 0
        for k in sd.keys():
            kk = k[len(pfx):] if pfx and k.startswith(pfx) else k
            if kk in target_keys:
                n += 1
        return n

    best_pfx = max(prefixes, key=_score_prefix)
    best_score = _score_prefix(best_pfx)
    print(f"[CKPT->PT] prefix='{best_pfx}' match_keys={best_score}/{len(target_keys)}", flush=True)

    converted = {}
    for k, v in sd.items():
        kk = k[len(best_pfx):] if best_pfx and k.startswith(best_pfx) else k
        if kk in target_keys:
            converted[kk] = v

    out = {
        "network": converted,
        "epoch": raw.get("epoch", extract_epoch_from_weight_filename(base)),
        "meta": {"source_ckpt": ckpt_path},
    }
    torch.save(out, out_path)
    print(f"[CKPT->PT] Saved: {out_path} (keys={len(converted)})", flush=True)
    return out_path


# ----------------------------
# Model loader
# ----------------------------
def build_opt(checkpoint_dir, name, gpu_ids):
    opt = SimpleNamespace()
    opt.gpu_ids = gpu_ids
    opt.lr = 5e-4
    opt.weight_decay = 5e-4
    opt.checkpoint_dir = checkpoint_dir
    opt.name = name

    opt.backbone = "mobilenetv2"
    opt.fpn = "fpn"
    opt.fpn_channels = 128
    opt.deform_groups = 4
    opt.gamma_mode = "SE"
    opt.beta_mode = "contextgatedconv"
    opt.n_layers = [1, 1, 1, 1]
    opt.extract_ids = [5, 11, 17, 23]
    opt.alpha = 0.25
    opt.gamma = 4
    opt.num_epochs = 100

    opt.load_pretrain = False
    return opt


def load_model(checkpoint_dir, name, weight_file, gpu_ids):
    opt = build_opt(checkpoint_dir, name, gpu_ids=gpu_ids)

    print("=" * 60, flush=True)
    print("[MODEL] Device info", flush=True)
    print(f"gpu_ids={opt.gpu_ids}", flush=True)
    print(f"torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}", flush=True)
        print(f"current_device={torch.cuda.current_device()}", flush=True)
        try:
            print(f"device_name(0)={torch.cuda.get_device_name(0)}", flush=True)
        except Exception:
            pass
    print("=" * 60, flush=True)

    model = create_model(opt)  # should not auto-load best.pth due to load_pretrain=False
    if not os.path.isfile(weight_file):
        raise FileNotFoundError(f"Weight file not found: {weight_file}")

    print(f"[MODEL] Loading checkpoint: {weight_file}", flush=True)
    ckpt = torch.load(weight_file, map_location=model.device, weights_only=False)

    # Support:
    # - ChangeDINO .pth/.pt checkpoints: dict with key 'network'
    # - exported pt dict: key 'state_dict'
    # - raw state_dict dict
    state_dict = None
    if isinstance(ckpt, dict):
        if "network" in ckpt:
            state_dict = ckpt["network"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # raw state_dict dict (param_name -> tensor)
            any_tensor = any(hasattr(v, "shape") for v in ckpt.values())
            if any_tensor:
                state_dict = ckpt

        print(f"[CKPT] keys: {list(ckpt.keys())}", flush=True)
        for k in ["epoch", "iter", "best", "metric", "acc", "miou"]:
            if k in ckpt:
                print(f"[CKPT] {k}={ckpt[k]}", flush=True)
        if isinstance(ckpt.get("meta", None), dict):
            meta = ckpt["meta"]
            for k in ["source_pth", "epoch", "iter", "best", "metric", "acc", "miou"]:
                if k in meta:
                    print(f"[PT] meta.{k}={meta[k]}", flush=True)

    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format. Expect dict with 'network' or 'state_dict', or a raw state_dict dict."
        )

    model.model.load_state_dict(state_dict, strict=False)
    model.eval()

    # torch.compile: JIT compile for faster inference (first batch slower, rest much faster)
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
        print("[MODEL] torch.compile enabled (mode=reduce-overhead)", flush=True)
    except Exception as e:
        print(f"[MODEL] torch.compile not available, skipping: {e}", flush=True)

    print(f"[MODEL] model.device = {model.device}", flush=True)
    return model, model.device


# ----------------------------
# CRS / transform checks
# ----------------------------
def _crs_equal(crs1, crs2) -> bool:
    if crs1 is None or crs2 is None:
        return False  # safer: if missing CRS, treat as mismatch
    try:
        e1, e2 = crs1.to_epsg(), crs2.to_epsg()
        if e1 is not None and e2 is not None:
            return e1 == e2
    except Exception:
        pass
    return crs1 == crs2


def _transform_close(t1, t2, rtol=1e-6, atol=1e-3) -> bool:
    t1t = (t1.a, t1.b, t1.c, t1.d, t1.e, t1.f)
    t2t = (t2.a, t2.b, t2.c, t2.d, t2.e, t2.f)
    return np.allclose(t1t, t2t, rtol=rtol, atol=atol)


# ----------------------------
# Tile reading (window-based)
# ----------------------------
def _read_tile_from_ds(ds, win: Window, tile_size: int, indexes=(1, 2, 3)):
    """
    Read window from dataset and pad to tile_size using reflect.
    Return: np.ndarray [tile_size, tile_size, C]
    """
    # clip to bounds
    x0 = max(0, int(win.col_off))
    y0 = max(0, int(win.row_off))
    x1 = min(ds.width, int(win.col_off + win.width))
    y1 = min(ds.height, int(win.row_off + win.height))

    C = len(indexes)
    dtype = np.dtype(ds.dtypes[0])

    if x1 <= x0 or y1 <= y0:
        return np.zeros((tile_size, tile_size, C), dtype=dtype)

    clipped = Window(x0, y0, x1 - x0, y1 - y0)
    arr = ds.read(indexes=indexes, window=clipped)  # [C,h,w]
    arr = np.transpose(arr, (1, 2, 0))  # [h,w,C]

    h, w, _ = arr.shape

    # pad bottom/right
    if h < tile_size or w < tile_size:
        pad_y = tile_size - h
        pad_x = tile_size - w
        arr = np.pad(arr, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")

    return arr[:tile_size, :tile_size, :]


def _read_tile2_aligned_to_tile1(ds2, ds1_transform, ds1_crs, win: Window, tile_size: int, indexes=(1, 2, 3)):
    """
    Reproject ds2 -> ds1 window grid (tile_size x tile_size), on-the-fly.
    Return: np.ndarray [tile_size, tile_size, C]
    """
    C = len(indexes)
    dtype = np.dtype(ds2.dtypes[0])
    dst = np.zeros((C, tile_size, tile_size), dtype=dtype)

    dst_tr = window_transform(win, ds1_transform)

    for i, b in enumerate(indexes):
        reproject(
            source=rasterio.band(ds2, b),
            destination=dst[i],
            src_transform=ds2.transform,
            src_crs=ds2.crs,
            dst_transform=dst_tr,
            dst_crs=ds1_crs,
            dst_width=tile_size,
            dst_height=tile_size,
            resampling=Resampling.bilinear,
        )

    return np.transpose(dst, (1, 2, 0))


# ----------------------------
# Overlap weight map
# ----------------------------
def make_weight_map(tile_size: int, min_w: float = 0.85) -> np.ndarray:
    center = tile_size // 2
    y, x = np.ogrid[:tile_size, :tile_size]
    dist = np.sqrt((y - center) ** 2 + (x - center) ** 2)
    max_dist = np.sqrt(center ** 2 + center ** 2)
    normalized = dist / (max_dist + 1e-8)
    w = min_w + (1.0 - min_w) * (1.0 - normalized)
    return np.clip(w, min_w, 1.0).astype(np.float32)


# ----------------------------
# Optional: ExG suppress (second pass)
# ----------------------------
def _exg_index(tile_rgb: np.ndarray) -> np.ndarray:
    x = tile_rgb.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return (2.0 * g - r - b).astype(np.float32)


def suppress_green_growth_second_pass(
    full_pred: np.ndarray,
    ds1,
    ds2,
    tile_size: int,
    overlap: int,
    exg_delta_thr: float,
    exg2_min_thr: float,
):
    """
    Second streaming pass: for each tile region, if ExG2-ExG1 large and ExG2 high,
    suppress predicted change pixels (set to 0). Does NOT read full image.
    """
    H, W = ds1.height, ds1.width
    stride = tile_size - overlap

    need_reproject_t2 = not (_crs_equal(ds1.crs, ds2.crs) and _transform_close(ds1.transform, ds2.transform) and (ds1.width, ds1.height) == (ds2.width, ds2.height))

    need_reproject_t2 = False
    suppressed = 0
    total = 0

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            win = Window(x, y, tile_size, tile_size)
            t1 = _read_tile_from_ds(ds1, win, tile_size, indexes=(1, 2, 3))
            if not need_reproject_t2:
                t2 = _read_tile_from_ds(ds2, win, tile_size, indexes=(1, 2, 3))
            else:
                t2 = _read_tile2_aligned_to_tile1(ds2, ds1.transform, ds1.crs, win, tile_size, indexes=(1, 2, 3))

            # valid area (avoid padded outside ds bounds)
            h_valid = min(tile_size, H - y)
            w_valid = min(tile_size, W - x)

            exg1 = _exg_index(t1[:h_valid, :w_valid, :])
            exg2 = _exg_index(t2[:h_valid, :w_valid, :])
            growth = (exg2 - exg1) > float(exg_delta_thr)
            green2 = exg2 > float(exg2_min_thr)
            suppress = growth & green2

            patch = full_pred[y:y + h_valid, x:x + w_valid]
            before = int(patch.sum())
            patch[suppress] = 0
            after = int(patch.sum())
            full_pred[y:y + h_valid, x:x + w_valid] = patch

            suppressed += (before - after)
            total += (h_valid * w_valid)

    ratio = suppressed / max(1, total)
    print(f"[SUPPRESS_GREEN] suppressed_pixels={suppressed} ratio={ratio:.6f}", flush=True)
    return full_pred


# ----------------------------
# Postprocess mask
# ----------------------------
def postprocess_mask(mask, remove_holes_area=200000, remove_objects_min_size=800, fill_all_holes=False):
    if mask.max() > 1:
        mask_cv = (mask > 0).astype(np.uint8) * 255
    else:
        mask_cv = mask.astype(np.uint8) * 255

    # fill holes
    if fill_all_holes:
        h, w = mask_cv.shape
        marker = np.zeros((h + 2, w + 2), dtype=np.uint8)
        marker[1:h + 1, 1:w + 1] = mask_cv
        mask_inv = 255 - marker
        cv2.floodFill(mask_inv, None, (0, 0), 255)
        mask_processed = 255 - mask_inv
        mask_processed = mask_processed[1:h + 1, 1:w + 1]
    else:
        mask_processed = mask_cv.copy()
        mask_inv = 255 - mask_processed
        contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if hierarchy[0][i][3] >= 0 and area < remove_holes_area:
                    cv2.drawContours(mask_processed, [contour], -1, 255, -1)

    # remove small objects
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_processed)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= remove_objects_min_size:
            cv2.drawContours(mask_final, [contour], -1, 255, -1)

    return (mask_final > 0).astype(np.uint8)


# ----------------------------
# Polygonize & save
# ----------------------------
def save_mask_to_shapefile(mask, transform, crs, out_shp_path, change_value=1, min_area_mu=1.0):
    mask_bool = (mask == change_value)
    gen = shapes(mask.astype(np.int16), mask=mask_bool, transform=transform)

    geoms = []
    vals = []
    for geom, val in gen:
        if val == change_value:
            geoms.append(shape(geom))
            vals.append(int(val))

    os.makedirs(os.path.dirname(out_shp_path), exist_ok=True)
    remove_shapefile(out_shp_path)

    if not geoms:
        print("[SHP] no change polygons, saving empty shp", flush=True)
        gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=crs)
        gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return

    gdf = gpd.GeoDataFrame({"value": vals}, geometry=geoms, crs=crs)

    # area calc
    is_geographic = False
    try:
        is_geographic = gdf.crs.is_geographic
    except Exception:
        pass

    if is_geographic:
        bounds = gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        utm_zone = int(np.floor((center_lon + 180) / 6) + 1)
        epsg_code = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
        gdf_proj = gdf.to_crs(epsg=epsg_code)
        areas_m2 = gdf_proj.geometry.area
    else:
        areas_m2 = gdf.geometry.area

    gdf["area_m2"] = areas_m2
    gdf["area_mu"] = areas_m2 / 666.67

    min_area_m2 = float(min_area_mu) * 666.67
    before = len(gdf)
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()
    after = len(gdf)
    print(f"[SHP] polygons: {before} -> {after} after area filter >= {min_area_mu} mu", flush=True)

    # remove contained polygons
    if len(gdf) > 1:
        gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)

        if is_geographic:
            gdf_chk = gdf.to_crs(epsg=epsg_code)
        else:
            gdf_chk = gdf

        contained = []
        for i in range(len(gdf)):
            gi = gdf_chk.iloc[i].geometry
            ai = gdf.iloc[i]["area_m2"]
            for j in range(i):
                gj = gdf_chk.iloc[j].geometry
                try:
                    if gi.within(gj) or gi.covered_by(gj):
                        contained.append(i)
                        break
                    if gi.intersects(gj):
                        inter = gi.intersection(gj)
                        if not inter.is_empty:
                            r = inter.area / max(1e-9, ai)
                            if r >= 0.90:
                                contained.append(i)
                                break
                except Exception:
                    pass

        if contained:
            print(f"[SHP] removing {len(contained)} contained polygons", flush=True)
            gdf = gdf.drop(gdf.index[contained]).reset_index(drop=True)

    gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[SHP] saved: {out_shp_path}", flush=True)


def remove_shapefile(path: str):
    stem, _ = os.path.splitext(path)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"):
        sidecar = stem + ext
        if os.path.exists(sidecar):
            os.remove(sidecar)


# ----------------------------
# Stitch buffers (RAM / memmap)
# ----------------------------
def _pad_len(L, tile_size, stride):
    if L <= tile_size:
        return tile_size
    n = (L - tile_size + stride - 1) // stride
    return n * stride + tile_size


def _maybe_memmap(shape, dtype, memmap_dir, name):
    if memmap_dir is None:
        return np.zeros(shape, dtype=dtype), None
    os.makedirs(memmap_dir, exist_ok=True)
    path = os.path.join(memmap_dir, f"{name}.dat")
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    arr[:] = 0
    return arr, path


# ----------------------------
# Core: streamed inference on datasets
# ----------------------------
def run_inference_on_datasets_streamed(
    model,
    device,
    ds1,
    ds2,
    tile_size=512,
    overlap=128,
    batch_size=16,
    use_amp=True,
    binarize_mode="logit_diff",
    prob_threshold=0.7,
    logit_threshold=0.5,
    smooth_overlap=False,
    smooth_kernel_size=3,
    amp_fallback_fp32=True,
    memmap_dir=None,
):
    assert tile_size > 0
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile_size")

    H, W = ds1.height, ds1.width
    H_pad, W_pad = _pad_len(H, tile_size, stride), _pad_len(W, tile_size, stride)

    # Decide memmap automatically if huge (heuristic)
    # float32 buffers: pred_sum + weight; int32 vote
    # If H_pad*W_pad too big, use memmap to avoid RAM blow.
    Npix = int(H_pad * W_pad)
    auto_memmap = (memmap_dir is not None)
    if memmap_dir is None and Npix >= 12000 * 12000:
        # auto turn on memmap in system temp dir if extremely large
        memmap_dir = os.path.join(os.path.abspath(os.getenv("TEMP", ".")), "cd_memmap")
        auto_memmap = True
        print(f"[MEMMAP] auto enabled at {memmap_dir} (H_pad*W_pad={Npix})", flush=True)

    full_pred_sum, p_sum_path = _maybe_memmap((H_pad, W_pad), np.float32, memmap_dir if auto_memmap else None, "pred_sum")
    full_weight, w_path = _maybe_memmap((H_pad, W_pad), np.float32, memmap_dir if auto_memmap else None, "weight")
    full_vote_cnt, v_path = _maybe_memmap((H_pad, W_pad), np.int32,   memmap_dir if auto_memmap else None, "vote")

    weight_map = make_weight_map(tile_size, min_w=0.85) if overlap > 0 else np.ones((tile_size, tile_size), np.float32)

    # speed flags
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # normalize const on GPU
    mean = torch.tensor([0.430, 0.411, 0.296], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.213, 0.156, 0.143], device=device).view(1, 3, 1, 1)

    def _prep_batch(batch_tiles):
        arr = np.stack(batch_tiles, axis=0).astype(np.float32)  # B,H,W,3
        # NOTE: if your tif is uint16 reflectance, adjust scaling here
        arr = arr / 255.0
        ten = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if device.type == "cuda":
            try:
                ten = ten.pin_memory()
            except RuntimeError:
                pass
        ten = ten.to(device, non_blocking=True)
        ten = (ten - mean) / std
        ten = ten.contiguous(memory_format=torch.channels_last)
        return ten

    # Decide whether ds2 must be reprojected/resampled onto ds1 grid tile-by-tile.
    # We can handle CRS mismatch by tile-level reproject in _read_tile2_aligned_to_tile1().
    crs_mismatch = not _crs_equal(ds1.crs, ds2.crs)
    grid_mismatch = not (_transform_close(ds1.transform, ds2.transform) and (ds1.width, ds1.height) == (ds2.width, ds2.height))
    need_reproject_t2 = crs_mismatch or grid_mismatch
    if crs_mismatch:
        print(f"[CRS] mismatch -> will reproject tif2 to tif1 CRS on-the-fly\n  ds1: {ds1.crs}\n  ds2: {ds2.crs}", flush=True)
    print(f"[GRID] need_reproject_t2={need_reproject_t2} (crs_mismatch={crs_mismatch}, grid_mismatch={grid_mismatch})", flush=True)

    # total tiles
    n_tiles_y = (H_pad + stride - 1) // stride
    n_tiles_x = (W_pad + stride - 1) // stride
    total_tiles = int(n_tiles_y * n_tiles_x)

    model.eval()
    t0 = time.time()
    last_print = 0
    progress_every = 200
    fallback_batches = 0

    # --- helper: read one tile pair ---
    def _read_tile_pair(y, x):
        win = Window(x, y, tile_size, tile_size)
        t1 = _read_tile_from_ds(ds1, win, tile_size, indexes=(1, 2, 3))
        if not need_reproject_t2:
            t2 = _read_tile_from_ds(ds2, win, tile_size, indexes=(1, 2, 3))
        else:
            t2 = _read_tile2_aligned_to_tile1(ds2, ds1.transform, ds1.crs, win, tile_size, indexes=(1, 2, 3))
        return t1, t2, y, x

    # --- helper: process one batch of tiles ---
    def _process_batch(buf_t1, buf_t2, buf_pos):
        nonlocal fallback_batches
        t1_tensor = _prep_batch(buf_t1)
        t2_tensor = _prep_batch(buf_t2)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (use_amp and device.type == "cuda")
            else torch.cpu.amp.autocast(enabled=False)
        )
        with autocast_ctx:
            logits = model.inference(t1_tensor, t2_tensor)

        if amp_fallback_fp32 and use_amp and device.type == "cuda":
            if not torch.isfinite(logits).all():
                fallback_batches += 1
                logits = model.inference(t1_tensor.float(), t2_tensor.float())

        if logits.shape[-2:] != (tile_size, tile_size):
            logits = F.interpolate(logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False)

        if binarize_mode == "prob":
            score_t = torch.softmax(logits, dim=1)[:, 1]
        else:
            score_t = logits[:, 1] - logits[:, 0]

        score_t = torch.nan_to_num(score_t, nan=0.0, posinf=0.0, neginf=0.0)
        score = score_t.float().cpu().numpy()

        for b, (yy, xx) in enumerate(buf_pos):
            h_valid = min(tile_size, H_pad - yy)
            w_valid = min(tile_size, W_pad - xx)
            p = score[b, :h_valid, :w_valid]
            w = weight_map[:h_valid, :w_valid]
            full_pred_sum[yy:yy + h_valid, xx:xx + w_valid] += p * w
            full_weight[yy:yy + h_valid, xx:xx + w_valid] += w
            full_vote_cnt[yy:yy + h_valid, xx:xx + w_valid] += 1

    tile_idx = 0
    buf_t1, buf_t2, buf_pos = [], [], []
    with torch.inference_mode():
        for y in range(0, H_pad, stride):
            for x in range(0, W_pad, stride):
                t1, t2, yy, xx = _read_tile_pair(y, x)
                tile_idx += 1

                # skip blank/nodata tiles (both images must have valid pixels)
                if t1.max() == 0 or t2.max() == 0:
                    continue

                buf_t1.append(t1)
                buf_t2.append(t2)
                buf_pos.append((yy, xx))

                if tile_idx - last_print >= progress_every:
                    elapsed = max(1e-6, time.time() - t0)
                    tps = tile_idx / elapsed
                    remain = max(0, total_tiles - tile_idx)
                    eta = remain / max(1e-6, tps)
                    print(f"[PROGRESS] {tile_idx}/{total_tiles} tiles  tps={tps:.2f}  ETA={eta/60:.1f} min", flush=True)
                    last_print = tile_idx

                if len(buf_t1) >= batch_size:
                    _process_batch(buf_t1, buf_t2, buf_pos)
                    buf_t1, buf_t2, buf_pos = [], [], []

        # last partial batch
        if buf_t1:
            _process_batch(buf_t1, buf_t2, buf_pos)

    if fallback_batches > 0:
        print(f"[AMP] fallback FP32 batches: {fallback_batches}", flush=True)

    # average & binarize
    mask = full_weight > 0
    avg = np.zeros((H_pad, W_pad), dtype=np.float32)
    avg[mask] = full_pred_sum[mask] / full_weight[mask]
    thr = prob_threshold if binarize_mode == "prob" else logit_threshold
    full_pred = np.zeros((H_pad, W_pad), dtype=np.uint8)
    full_pred[mask] = (avg[mask] > thr).astype(np.uint8)

    # stats
    pos_ratio = float(full_pred[:H, :W].mean())
    avg_min = float(np.nanmin(avg[mask])) if np.any(mask) else 0.0
    avg_mean = float(np.nanmean(avg[mask])) if np.any(mask) else 0.0
    avg_max = float(np.nanmax(avg[mask])) if np.any(mask) else 0.0
    print(f"[MASK] mode={binarize_mode} thr={thr} avg(min/mean/max)={avg_min:.4f}/{avg_mean:.4f}/{avg_max:.4f} pos_ratio={pos_ratio:.4f}", flush=True)

    # optional overlap smoothing
    if smooth_overlap and overlap > 0:
        overlap_mask = full_vote_cnt > 1
        if np.any(overlap_mask):
            k = max(1, int(smooth_kernel_size))
            kernel = np.ones((k, k), np.uint8)
            pred_cv = (full_pred * 255).astype(np.uint8)
            sm = cv2.morphologyEx(pred_cv, cv2.MORPH_OPEN, kernel)
            sm = cv2.morphologyEx(sm, cv2.MORPH_CLOSE, kernel)
            full_pred[overlap_mask] = (sm[overlap_mask] > 127).astype(np.uint8)

    # return cropped to original H,W
    return full_pred[:H, :W], ds1.transform, ds1.crs


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser("Stream big TIF prediction with ChangeDINO (no full image in RAM)")
    parser.add_argument("--tif1", type=str, default=r"E:\随州\安居镇\clip\2018.tif")
    parser.add_argument("--tif2", type=str, default=r"E:\随州\安居镇\clip\2020.tif")
    parser.add_argument("--checkpoint_dir", type=str, default=r'D:\Projects\change_detection\ChangeDino_dynamic_all_elements\checkpoints\SECOND_nogg_ruian_change2242_nochange2242\_ckpt_converted')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--out_base_dir", type=str, default=r'D:\Projects\change_detection\open-cd-main\results\changedino\bcd\big_tif\e33_1')

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--gpu_ids", type=str, default="0", help="e.g. '0' or '0,1' or '' for CPU")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--binarize_mode", type=str, default="logit_diff", choices=["logit_diff", "prob"])
    parser.add_argument("--prob_threshold", type=float, default=0.7)
    parser.add_argument("--logit_threshold", type=float, default=0.5)

    parser.add_argument("--smooth_overlap", action="store_true")
    parser.add_argument("--smooth_kernel_size", type=int, default=3)

    parser.add_argument("--remove_holes_area", type=int, default=200000)
    parser.add_argument("--remove_objects_min_size", type=int, default=800)
    parser.add_argument("--fill_all_holes", action="store_true")

    parser.add_argument("--min_area_mu", type=float, default=0.1)

    # CRS handling
    parser.add_argument(
        "--strict_crs",
        action="store_true",
        help="If set, abort when tif1/tif2 CRS differ. Default: auto reproject tif2 onto tif1 grid per-tile.",
    )

    # weight selection
    parser.add_argument("--weights", nargs="*", default=None)
    parser.add_argument("--epochs", type=str, default='33')

    # memmap (optional)
    parser.add_argument("--memmap_dir", type=str, default=None, help="If set, use memmap buffers in this dir to reduce RAM.")

    # optional suppress green growth (second pass)
    parser.add_argument("--suppress_green_growth", action="store_true")
    parser.add_argument("--exg_delta_thr", type=float, default=0.10)
    parser.add_argument("--exg2_min_thr", type=float, default=0.05)
    parser.add_argument(
        "--auto_align_if_size_mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If tif1/tif2 size mismatch, run 1.align_tif.py first and continue with aligned files.",
    )
    parser.add_argument(
        "--align_script",
        type=str,
        default="/mnt/wuxy/change_detection/ChangeDINO_dynamic_DDP/1.align_tif.py",
        help="Path to alignment script (1.align_tif.py).",
    )

    args = parser.parse_args()

    print(f'tif1: {args.tif1}')
    print(f'tif2: {args.tif2}')
    print(f'out_base_dir: {args.out_base_dir}')

    gpu_ids = []
    if args.gpu_ids.strip():
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip().isdigit()]

    use_amp = (not args.no_amp)

    weight_files = _resolve_weight_list(args.checkpoint_dir, args.weights, args.epochs)
    print(f"[WEIGHTS] {len(weight_files)} files", flush=True)
    for _, fn in weight_files:
        print(f"  - {fn}", flush=True)

    os.makedirs(args.out_base_dir, exist_ok=True)

    # If size mismatch, auto-align by invoking 1.align_tif.py, then use aligned files for prediction.
    with rasterio.open(args.tif1) as _ds1_chk, rasterio.open(args.tif2) as _ds2_chk:
        size_mismatch = (_ds1_chk.width, _ds1_chk.height) != (_ds2_chk.width, _ds2_chk.height)
    if size_mismatch:
        if not args.auto_align_if_size_mismatch:
            raise ValueError(
                f"Size mismatch:\n  tif1: {_ds1_chk.width}x{_ds1_chk.height}\n"
                f"  tif2: {_ds2_chk.width}x{_ds2_chk.height}\n"
                "Enable --auto_align_if_size_mismatch or align them manually first."
            )

        aligned_dir = os.path.join(args.out_base_dir, "_aligned_inputs")
        os.makedirs(aligned_dir, exist_ok=True)
        base1 = os.path.splitext(os.path.basename(args.tif1))[0]
        base2 = os.path.splitext(os.path.basename(args.tif2))[0]
        aligned_tif1 = os.path.join(aligned_dir, f"{base1}_aligned_t1.tif")
        aligned_tif2 = os.path.join(aligned_dir, f"{base2}_aligned_t2.tif")

        if not (os.path.isfile(aligned_tif1) and os.path.isfile(aligned_tif2)):
            print(
                f"[ALIGN] size mismatch detected, running align script:\n"
                f"  tif1: {args.tif1}\n  tif2: {args.tif2}",
                flush=True,
            )
            cmd = [
                sys.executable,
                args.align_script,
                "--tif1",
                args.tif1,
                "--tif2",
                args.tif2,
                "--out_tif1",
                aligned_tif1,
                "--out_tif2",
                aligned_tif2,
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"[ALIGN] reuse cached aligned files in {aligned_dir}", flush=True)

        args.tif1 = aligned_tif1
        args.tif2 = aligned_tif2
        print(f"[ALIGN] use aligned tif1={args.tif1}", flush=True)
        print(f"[ALIGN] use aligned tif2={args.tif2}", flush=True)

    with rasterio.open(args.tif1) as ds1, rasterio.open(args.tif2) as ds2:
        # basic CRS check
        if ds1.crs is None or ds2.crs is None:
            raise ValueError("CRS is None for one of the inputs. Please define CRS before running.")
        if (ds1.width, ds1.height) != (ds2.width, ds2.height):
            raise ValueError(
                f"Size mismatch:\n"
                f"  tif1: {ds1.width}x{ds1.height}\n"
                f"  tif2: {ds2.width}x{ds2.height}\n"
                f"Please align/resample first."
            )
        # Print alignment status if already aligned (same CRS + very close transform + same size)
        if _crs_equal(ds1.crs, ds2.crs) and _transform_close(ds1.transform, ds2.transform) and (ds1.width, ds1.height) == (ds2.width, ds2.height):
            print("[ALIGN] inputs already aligned (CRS/transform/size match)", flush=True)
        if (not _crs_equal(ds1.crs, ds2.crs)) and args.strict_crs:
            raise ValueError(f"CRS mismatch (strict_crs=True):\n  tif1: {ds1.crs}\n  tif2: {ds2.crs}")
        if not _crs_equal(ds1.crs, ds2.crs):
            print(f"[CRS] mismatch detected, auto mode enabled (use --strict_crs to abort)\n  tif1: {ds1.crs}\n  tif2: {ds2.crs}", flush=True)

        # loop weights
        for i, (wpath, wfn) in enumerate(weight_files, 1):
            print("\n" + "=" * 80)
            print(f"[RUN] {i}/{len(weight_files)}  {wfn}")
            print("=" * 80)

            ep = extract_epoch_from_weight_filename(wfn)
            ep_folder = f"e{ep}" if ep is not None else "best"
            out_dir = os.path.join(args.out_base_dir, ep_folder)

            # skip if already exists
            if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
                print(f"[SKIP] output exists: {out_dir}", flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)

            model, device = load_model(args.checkpoint_dir, args.name, wpath, gpu_ids=gpu_ids)

            # infer
            pred_mask, transform, crs = run_inference_on_datasets_streamed(
                model=model,
                device=device,
                ds1=ds1,
                ds2=ds2,
                tile_size=args.tile_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                use_amp=use_amp,
                binarize_mode=args.binarize_mode,
                prob_threshold=args.prob_threshold,
                logit_threshold=args.logit_threshold,
                smooth_overlap=args.smooth_overlap,
                smooth_kernel_size=args.smooth_kernel_size,
                memmap_dir=args.memmap_dir,
            )

            # optional second pass suppression (no full image)
            if args.suppress_green_growth:
                pred_mask = suppress_green_growth_second_pass(
                    full_pred=pred_mask,
                    ds1=ds1,
                    ds2=ds2,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    exg_delta_thr=args.exg_delta_thr,
                    exg2_min_thr=args.exg2_min_thr,
                )

            # postprocess
            print("[POST] postprocess mask...", flush=True)
            pred_mask = postprocess_mask(
                pred_mask,
                remove_holes_area=args.remove_holes_area,
                remove_objects_min_size=args.remove_objects_min_size,
                fill_all_holes=args.fill_all_holes,
            )

            # vectorize
            out_shp = os.path.join(out_dir, "change.shp")

            print("[SHP] polygonize + save...", flush=True)
            save_mask_to_shapefile(pred_mask, transform, crs, out_shp, change_value=1, min_area_mu=args.min_area_mu)

            print(f"[OK] done: {out_shp}", flush=True)

    print("[DONE] all weights processed", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[TIME] {time.time() - t0:.1f} sec", flush=True)

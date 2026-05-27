# coding=utf-8
"""
Change3D BCD: 大图流式推理 + 后处理 + polygonize 输出 shp

参考逻辑来源：
  - ChangeDino_dynamic_all_elements/predict_big_pt_flow.py

本脚本假设输入为两张带地理信息的 GeoTIFF：
  - tif1: time1 (pre) RGB
  - tif2: time2 (post) RGB

模型输入：
  - TorchScript .pt：torch.jit.load 后可直接调用 model(pre, post)
  - pre/post: [B,3,H,W] in_height/in_width（默认 512）
  - 输出：prob [B,1,H,W]，阈值 prob_threshold => mask(0/1)

后处理：
  - fill holes / remove small objects（OpenCV）

输出：
  - shapefile（ESRI Shapefile）包含变化区域多边形
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import tempfile
import gc
from os.path import join as osp
from pathlib import Path
from typing import Dict, Optional, Tuple

from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes

import geopandas as gpd
import pandas as pd
from shapely.geometry import LinearRing, MultiPolygon, Polygon, shape


def _crs_equal(crs1, crs2) -> bool:
    if crs1 is None or crs2 is None:
        return False
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


def _pad_len(L: int, tile_size: int, stride: int) -> int:
    if L <= tile_size:
        return tile_size
    n = (L - tile_size + stride - 1) // stride
    return n * stride + tile_size


def _maybe_memmap(shape: Tuple[int, ...], dtype, memmap_dir: Optional[str], name: str):
    if memmap_dir is None:
        return np.zeros(shape, dtype=dtype), None
    # Some callers may pass empty string "" meaning "not set".
    # In that case, fall back to a default temp directory.
    if isinstance(memmap_dir, str) and memmap_dir.strip() == "":
        memmap_dir = osp(os.path.abspath(os.getenv("TEMP", tempfile.gettempdir())), "cd_memmap")
    os.makedirs(memmap_dir, exist_ok=True)
    path = os.path.join(memmap_dir, f"{name}.dat")
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    gb = nbytes / (1024**3)

    # 优化：使用 seek + write 创建稀疏文件，比逐块写零快得多
    # 新文件内容默认为零，无需显式初始化
    import struct

    # 先创建正确大小的文件（通过 seek + write 单字节）
    with open(path, "wb") as f:
        if nbytes > 0:
            f.seek(nbytes - 1)
            f.write(b"\x00")

    # 以 r+ 模式打开 memmap，避免重新初始化
    arr = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    print(f"[MEMMAP] created {name}.dat shape={shape} ~{gb:.2f} GiB", flush=True)
    return arr, path


def _cleanup_memmap_paths(memmap_paths, remove_parent_dir=False):
    if not memmap_paths:
        return
    gc.collect()
    uniq_paths = []
    seen = set()
    for p in memmap_paths:
        if p and p not in seen:
            uniq_paths.append(p)
            seen.add(p)
    for p in uniq_paths:
        try:
            os.remove(p)
            print(f"[MEMMAP] removed {p}", flush=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[MEMMAP] failed remove {p}: {e}", flush=True)
    if remove_parent_dir:
        parent = os.path.dirname(uniq_paths[0])
        if parent:
            try:
                os.rmdir(parent)
                print(f"[MEMMAP] removed dir {parent}", flush=True)
            except OSError:
                pass


def make_weight_map(tile_size: int, min_w: float = 0.85) -> np.ndarray:
    center = tile_size // 2
    y, x = np.ogrid[:tile_size, :tile_size]
    dist = np.sqrt((y - center) ** 2 + (x - center) ** 2)
    max_dist = np.sqrt(center**2 + center**2)
    normalized = dist / (max_dist + 1e-8)
    w = min_w + (1.0 - min_w) * (1.0 - normalized)
    return np.clip(w, min_w, 1.0).astype(np.float32)


def _read_tile_from_ds(ds: rasterio.io.DatasetReader, win: Window, tile_size: int, indexes):
    """
    Read window and pad reflect to tile_size.
    Return: np.ndarray [tile_size, tile_size, C] (C=len(indexes))
    """
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
    if h < tile_size or w < tile_size:
        pad_y = tile_size - h
        pad_x = tile_size - w
        arr = np.pad(arr, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
    return arr[:tile_size, :tile_size, :]


def _read_tile2_aligned_to_tile1(
    ds2: rasterio.io.DatasetReader,
    ds1_transform,
    ds1_crs,
    win: Window,
    tile_size: int,
    indexes,
):
    """
    Reproject ds2 tile to ds1 tile grid on-the-fly.
    Return: np.ndarray [tile_size, tile_size, C]
    """
    from rasterio.windows import transform as window_transform

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


def postprocess_mask(mask, remove_holes_area=200000, remove_objects_min_size=800, fill_all_holes=False):
    """
    mask: uint8/float, 0/1 or 0/255
    return: uint8 {0,1}
    """
    if mask.max() > 1:
        mask_cv = (mask > 0).astype(np.uint8) * 255
    else:
        mask_cv = mask.astype(np.uint8) * 255

    # fill holes
    if fill_all_holes:
        h, w = mask_cv.shape
        marker = np.zeros((h + 2, w + 2), dtype=np.uint8)
        marker[1 : h + 1, 1 : w + 1] = mask_cv
        mask_inv = 255 - marker
        cv2.floodFill(mask_inv, None, (0, 0), 255)
        mask_processed = 255 - mask_inv
        mask_processed = mask_processed[1 : h + 1, 1 : w + 1]
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


def _chaikin(coords, iterations=1, weight=0.8):
    smoothed = coords[:]
    for _ in range(max(0, int(iterations))):
        if len(smoothed) < 4:
            break
        new_coords = []
        for i in range(len(smoothed) - 1):
            p1 = smoothed[i]
            p2 = smoothed[i + 1]
            q1 = (weight * p1[0] + (1 - weight) * p2[0], weight * p1[1] + (1 - weight) * p2[1])
            q2 = ((1 - weight) * p1[0] + weight * p2[0], (1 - weight) * p1[1] + weight * p2[1])
            new_coords.extend([q1, q2])
        if new_coords:
            new_coords.append(new_coords[0])
        smoothed = new_coords
    return smoothed


def _smooth_ring(ring: LinearRing, chaikin_iterations: int, chaikin_weight: float) -> LinearRing:
    coords = list(ring.coords)
    if len(coords) < 4:
        return ring
    smoothed = _chaikin(coords, iterations=chaikin_iterations, weight=chaikin_weight)
    if len(smoothed) < 4:
        return ring
    try:
        return LinearRing(smoothed)
    except Exception:
        return ring


def _process_polygon_boundary(
    poly: Polygon,
    dp_tolerance_m: float,
    enable_chaikin: bool,
    chaikin_iterations: int,
    chaikin_weight: float,
) -> Polygon:
    if poly.is_empty:
        return poly

    p = poly
    if dp_tolerance_m > 0:
        p = p.simplify(float(dp_tolerance_m), preserve_topology=True)
        if p.is_empty:
            return poly

    if not enable_chaikin:
        return p

    try:
        ext = _smooth_ring(p.exterior, chaikin_iterations, chaikin_weight)
        holes = []
        for interior in p.interiors:
            holes.append(_smooth_ring(LinearRing(interior.coords), chaikin_iterations, chaikin_weight))
        p2 = Polygon(ext, holes=[list(h.coords) for h in holes if len(h.coords) >= 4])
        if not p2.is_valid:
            p2 = p2.buffer(0)
        return p2 if not p2.is_empty else p
    except Exception:
        return p


def _process_geometry_boundary(
    geom,
    dp_tolerance_m: float,
    enable_chaikin: bool,
    chaikin_iterations: int,
    chaikin_weight: float,
):
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return _process_polygon_boundary(geom, dp_tolerance_m, enable_chaikin, chaikin_iterations, chaikin_weight)
    if isinstance(geom, MultiPolygon):
        processed = [
            _process_polygon_boundary(p, dp_tolerance_m, enable_chaikin, chaikin_iterations, chaikin_weight)
            for p in geom.geoms
        ]
        processed = [p for p in processed if p is not None and not p.is_empty]
        if not processed:
            return geom
        return MultiPolygon(processed)
    return geom


def save_mask_to_shapefile(
    mask,
    transform,
    crs,
    out_shp_path,
    change_value=1,
    min_area_mu=1.0,
    overlap_ratio=0.90,
    dp_tolerance_m=1.0,
    disable_chaikin=False,
    chaikin_iterations=1,
    chaikin_weight=0.8,
    postprocess: bool = True,
):
    mask_bool = mask == change_value
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

    if not bool(postprocess):
        # Raw polygonize output (no geometry fix/simplify/smooth/area-filter/containment removal).
        gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        print(f"[SHP] saved (raw, no postprocess): {out_shp_path}", flush=True)
        return

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
    else:
        gdf_proj = gdf

    # geometry fix + boundary postprocess (DP + Chaikin)
    try:
        gdf_proj["geometry"] = gdf_proj.geometry.make_valid()
    except Exception:
        gdf_proj["geometry"] = gdf_proj.geometry.buffer(0)
    gdf_proj = gdf_proj[gdf_proj.geometry.notna()].copy()
    gdf_proj = gdf_proj[gdf_proj.geometry.is_empty == False].copy()
    if gdf_proj.empty:
        print("[SHP] all polygons invalid after geometry fix, saving empty shp", flush=True)
        gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=crs)
        gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return

    gdf_proj["geometry"] = gdf_proj.geometry.apply(
        lambda geom: _process_geometry_boundary(
            geom,
            dp_tolerance_m=float(dp_tolerance_m),
            enable_chaikin=not bool(disable_chaikin),
            chaikin_iterations=int(chaikin_iterations),
            chaikin_weight=float(chaikin_weight),
        )
    )
    gdf_proj = gdf_proj[gdf_proj.geometry.notna()].copy()
    gdf_proj = gdf_proj[gdf_proj.geometry.is_empty == False].copy()
    if gdf_proj.empty:
        print("[SHP] all polygons removed after boundary processing, saving empty shp", flush=True)
        gdf = gpd.GeoDataFrame({"value": []}, geometry=[], crs=crs)
        gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return

    areas_m2 = gdf_proj.geometry.area

    gdf_proj["area_m2"] = areas_m2
    gdf_proj["area_mu"] = areas_m2 / 666.67

    min_area_m2 = float(min_area_mu) * 666.67
    before = len(gdf_proj)
    gdf_proj = gdf_proj[gdf_proj["area_m2"] >= min_area_m2].copy()
    after = len(gdf_proj)
    print(f"[SHP] polygons: {before} -> {after} after area filter >= {min_area_mu} mu", flush=True)

    # remove contained polygons
    if len(gdf_proj) > 1:
        gdf_proj = gdf_proj.sort_values("area_m2", ascending=False).reset_index(drop=True)
        gdf_chk = gdf_proj

        contained = []
        for i in range(len(gdf_proj)):
            gi = gdf_chk.iloc[i].geometry
            ai = gdf_proj.iloc[i]["area_m2"]
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
                            if r >= float(overlap_ratio):
                                contained.append(i)
                                break
                except Exception:
                    pass

        if contained:
            print(f"[SHP] removing {len(contained)} contained polygons", flush=True)
            gdf_proj = gdf_proj.drop(gdf_proj.index[contained]).reset_index(drop=True)

    if is_geographic:
        gdf = gdf_proj.to_crs(crs)
    else:
        gdf = gdf_proj
    gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[SHP] saved: {out_shp_path}", flush=True)


def remove_shapefile(path: str):
    stem, _ = os.path.splitext(path)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"):
        sidecar = stem + ext
        if os.path.exists(sidecar):
            os.remove(sidecar)


def save_multiclass_mask_to_shapefile(
    mask,
    transform,
    crs,
    out_shp_path,
    num_classes: int,
    export_class_ids=None,
    min_area_mu=1.0,
    overlap_ratio=0.90,
    dp_tolerance_m=1.0,
    disable_chaikin=False,
    chaikin_iterations=1,
    chaikin_weight=0.8,
    remove_holes_area=200000,
    remove_objects_min_size=800,
    fill_all_holes=False,
    postprocess: bool = True,
):
    """
    mask: uint8 class-id map [H,W], with background class 0 recommended.
    Export all classes in export_class_ids into ONE shp (same file), with attribute column `value`.
    """
    if export_class_ids is None:
        export_class_ids = list(range(1, int(num_classes)))

    os.makedirs(os.path.dirname(out_shp_path), exist_ok=True)
    remove_shapefile(out_shp_path)

    gdfs = []
    with tempfile.TemporaryDirectory() as td:
        any_written = False
        for cid in export_class_ids:
            cid = int(cid)
            bin_mask = (mask == cid).astype(np.uint8)
            if bin_mask.max() == 0:
                continue

            if bool(postprocess):
                bin_mask = postprocess_mask(
                    bin_mask,
                    remove_holes_area=remove_holes_area,
                    remove_objects_min_size=remove_objects_min_size,
                    fill_all_holes=fill_all_holes,
                )
            if bin_mask.max() == 0:
                continue

            mask_val = (bin_mask * cid).astype(np.uint8)
            any_written = True
            tmp_shp = osp(td, f"tmp_class_{cid}.shp")

            save_mask_to_shapefile(
                mask=mask_val,
                transform=transform,
                crs=crs,
                out_shp_path=tmp_shp,
                change_value=cid,
                min_area_mu=min_area_mu,
                overlap_ratio=overlap_ratio,
                dp_tolerance_m=dp_tolerance_m,
                disable_chaikin=disable_chaikin,
                chaikin_iterations=chaikin_iterations,
                chaikin_weight=chaikin_weight,
                postprocess=bool(postprocess),
            )

            gdf_part = gpd.read_file(tmp_shp)
            # Keep `value` column as class id.
            gdfs.append(gdf_part)

        if not any_written:
            print("[SHP] no polygons found for any class, saving empty shp", flush=True)
            gdf_empty = gpd.GeoDataFrame({"value": []}, geometry=[], crs=crs)
            gdf_empty.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
            return

        combined = pd.concat(gdfs, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=crs)
        combined.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        print(f"[SHP] multiclass saved: {out_shp_path} (classes={export_class_ids})", flush=True)


def run_inference_streamed_to_mask(
    model,
    device,
    ds1,
    ds2,
    tile_size: int,
    overlap: int,
    batch_size: int,
    prob_threshold: float,
    input_divisor: float,
    memmap_dir: Optional[str] = None,
    cleanup_memmap: bool = True,
):
    """
    Stream tiles: overlap-weighted stitching -> return mask [H,W] uint8 {0,1}
    """
    assert tile_size > 0
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile_size")

    H, W = ds1.height, ds1.width
    H_pad, W_pad = _pad_len(H, tile_size, stride), _pad_len(W, tile_size, stride)

    Npix = int(H_pad * W_pad)
    auto_memmap = memmap_dir is not None
    if memmap_dir is None and Npix >= 12000 * 12000:
        auto_memmap = True
        memmap_dir = osp(os.path.abspath(os.getenv("TEMP", ".")), "cd_memmap")
        print(f"[MEMMAP] auto enabled at {memmap_dir} (H_pad*W_pad={Npix})", flush=True)

    # buffers
    full_pred_sum, pred_sum_path = _maybe_memmap((H_pad, W_pad), np.float32, memmap_dir if auto_memmap else None, "pred_sum")
    full_weight, weight_path = _maybe_memmap((H_pad, W_pad), np.float32, memmap_dir if auto_memmap else None, "weight")

    weight_map = make_weight_map(tile_size, min_w=0.85) if overlap > 0 else np.ones((tile_size, tile_size), np.float32)

    # choose RGB indexes
    def _rgb_indexes(ds):
        if ds.count >= 3:
            return (1, 2, 3)
        if ds.count == 2:
            return (1, 2, 2)
        # count == 1
        return (1, 1, 1)

    idx1 = _rgb_indexes(ds1)
    idx2 = _rgb_indexes(ds2)

    crs_mismatch = not _crs_equal(ds1.crs, ds2.crs)
    grid_mismatch = not (_transform_close(ds1.transform, ds2.transform) and (ds1.width, ds1.height) == (ds2.width, ds2.height))
    need_reproject_t2 = crs_mismatch or grid_mismatch
    print(f"[GRID] need_reproject_t2={need_reproject_t2} (crs_mismatch={crs_mismatch}, grid_mismatch={grid_mismatch})", flush=True)

    # normalization: mean/std for BCDTransforms default: [0.5]*3, [0.5]*3
    mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def _prep_batch(batch_tiles_rgb):
        # batch_tiles_rgb: list[np.ndarray] each [tile,tile,3] (RGB uint/float)
        arr = np.stack(batch_tiles_rgb, axis=0).astype(np.float32)  # [B,H,W,3]
        arr = arr / float(input_divisor)
        ten = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
        # Important: move to the same device before normalize with mean/std.
        ten = ten.to(device=device, non_blocking=True)
        ten = (ten - mean) / std
        return ten

    def _read_tile_pair(y, x):
        win = Window(x, y, tile_size, tile_size)
        t1 = _read_tile_from_ds(ds1, win, tile_size, indexes=idx1)
        if not need_reproject_t2:
            t2 = _read_tile_from_ds(ds2, win, tile_size, indexes=idx2)
        else:
            t2 = _read_tile2_aligned_to_tile1(ds2, ds1.transform, ds1.crs, win, tile_size, indexes=idx2)
        return t1, t2, y, x

    n_tiles_y = (H_pad + stride - 1) // stride
    n_tiles_x = (W_pad + stride - 1) // stride
    total_tiles = int(n_tiles_y * n_tiles_x)

    model.eval()

    tile_idx = 0
    buf_t1, buf_t2, buf_pos = [], [], []
    t0 = time.time()
    last_print = 0
    progress_every = 200

    # --- 性能优化 ---
    use_amp = device.type == "cuda"
    _batch_buf_np = np.empty((batch_size, tile_size, tile_size, 3), dtype=np.float32)
    prefetch_size = batch_size * 2
    tile_queue = Queue(maxsize=prefetch_size)
    _SENTINEL = object()
    print(f"[PERF] use_amp={use_amp}, prefetch={prefetch_size}, batch_size={batch_size}", flush=True)

    def _prep_batch_fast(tiles_list):
        B = len(tiles_list)
        buf = _batch_buf_np[:B]
        for i, t in enumerate(tiles_list):
            np.copyto(buf[i], t, casting='unsafe')
        arr = buf[:B] / float(input_divisor)
        ten = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
        ten = ten.to(device=device, non_blocking=True)
        ten = (ten - mean) / std
        return ten

    def _flush_batch():
        nonlocal buf_t1, buf_t2, buf_pos
        if not buf_t1:
            return
        ten1 = _prep_batch_fast(buf_t1)
        ten2 = _prep_batch_fast(buf_t2)
        with torch.cuda.amp.autocast(enabled=use_amp):
            prob = model(ten1, ten2)  # [B,1,H,W]
        if prob.shape[-2:] != (tile_size, tile_size):
            prob = torch.nn.functional.interpolate(prob.float(), size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        score = prob[:, 0].float().cpu().numpy()  # [B,H,W]

        for b, (py, px) in enumerate(buf_pos):
            h_valid = min(tile_size, H_pad - py)
            w_valid = min(tile_size, W_pad - px)
            p = score[b, :h_valid, :w_valid]
            w = weight_map[:h_valid, :w_valid]
            full_pred_sum[py : py + h_valid, px : px + w_valid] += p * w
            full_weight[py : py + h_valid, px : px + w_valid] += w
        buf_t1, buf_t2, buf_pos = [], [], []

    def _producer():
        for y in range(0, H_pad, stride):
            for x in range(0, W_pad, stride):
                t1, t2, yy, xx = _read_tile_pair(y, x)
                ts = tile_size
                if (t1[0, 0, 0] <= 0 and t1[0, ts-1, 0] <= 0 and t1[ts-1, 0, 0] <= 0 and
                    t2[0, 0, 0] <= 0 and t2[0, ts-1, 0] <= 0 and t2[ts-1, 0, 0] <= 0):
                    if t1.max() <= 0 and t2.max() <= 0:
                        tile_queue.put(None)
                        continue
                tile_queue.put((t1, t2, yy, xx))
        tile_queue.put(_SENTINEL)

    reader_thread = Thread(target=_producer, daemon=True)
    reader_thread.start()

    with torch.inference_mode():
        while True:
            item = tile_queue.get()
            if item is _SENTINEL:
                _flush_batch()
                break
            if item is None:
                tile_idx += 1
                continue

            t1, t2, yy, xx = item
            tile_idx += 1
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
                _flush_batch()

        reader_thread.join(timeout=5)

    # average & binarize
    mask = full_weight > 0
    avg = np.zeros((H_pad, W_pad), dtype=np.float32)
    avg[mask] = full_pred_sum[mask] / full_weight[mask]

    full_pred = np.zeros((H_pad, W_pad), dtype=np.uint8)
    full_pred[mask] = (avg[mask] > float(prob_threshold)).astype(np.uint8)

    # stats
    pos_ratio = float(full_pred[:H, :W].mean())
    avg_min = float(np.nanmin(avg[mask])) if np.any(mask) else 0.0
    avg_mean = float(np.nanmean(avg[mask])) if np.any(mask) else 0.0
    avg_max = float(np.nanmax(avg[mask])) if np.any(mask) else 0.0
    print(f"[MASK] thr={prob_threshold} avg(min/mean/max)={avg_min:.4f}/{avg_mean:.4f}/{avg_max:.4f} pos_ratio={pos_ratio:.4f}", flush=True)
    memmap_paths = [pred_sum_path, weight_path]
    if auto_memmap and bool(cleanup_memmap):
        del full_pred_sum, full_weight
        _cleanup_memmap_paths(memmap_paths, remove_parent_dir=True)
    return full_pred[:H, :W], ds1.transform, ds1.crs


def run_inference_streamed_to_mask_multiclass(
    model,
    device,
    ds1,
    ds2,
    tile_size: int,
    overlap: int,
    batch_size: int,
    num_classes: int,
    input_divisor: float,
    memmap_dir: Optional[str] = None,
    memmap_fp16: bool = True,
    cleanup_memmap: bool = True,
):
    """
    Stream tiles with overlap-weighted stitching.
    Return: mask [H,W] uint8 class-id with values in [0..num_classes-1]
    """
    assert tile_size > 0
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile_size")

    H, W = ds1.height, ds1.width
    H_pad, W_pad = _pad_len(H, tile_size, stride), _pad_len(W, tile_size, stride)
    Npix = int(H_pad * W_pad)

    # allocate buffers (memmap optional)
    # NOTE: use stripe-level rolling fusion to avoid full-image pred_sum cache.
    auto_memmap = memmap_dir is not None
    if memmap_dir is None and Npix * int(num_classes) >= int(4e9 / 4):
        auto_memmap = True
        memmap_dir = osp(os.path.abspath(os.getenv("TEMP", ".")), "cd_memmap_mc")
        print(f"[MEMMAP] auto enabled at {memmap_dir} (H_pad*W_pad*C={Npix}*{num_classes})", flush=True)

    pred_sum_dtype = np.float16 if bool(memmap_fp16) else np.float32
    full_pred, pred_out_path = _maybe_memmap(
        (H_pad, W_pad),
        np.uint8,
        memmap_dir if auto_memmap else None,
        "pred_out_mc",
    )

    weight_map = make_weight_map(tile_size, min_w=0.85) if overlap > 0 else np.ones((tile_size, tile_size), np.float32)

    def _rgb_indexes(ds):
        if ds.count >= 3:
            return (1, 2, 3)
        if ds.count == 2:
            return (1, 2, 2)
        return (1, 1, 1)

    idx1 = _rgb_indexes(ds1)
    idx2 = _rgb_indexes(ds2)

    crs_mismatch = not _crs_equal(ds1.crs, ds2.crs)
    grid_mismatch = not (_transform_close(ds1.transform, ds2.transform) and (ds1.width, ds1.height) == (ds2.width, ds2.height))
    need_reproject_t2 = crs_mismatch or grid_mismatch
    print(f"[GRID] need_reproject_t2={need_reproject_t2} (crs_mismatch={crs_mismatch}, grid_mismatch={grid_mismatch})", flush=True)

    mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)

    def _prep_batch(batch_tiles_rgb):
        arr = np.stack(batch_tiles_rgb, axis=0).astype(np.float32)  # [B,H,W,3]
        arr = arr / float(input_divisor)
        ten = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [B,3,H,W]
        ten = ten.to(device=device, non_blocking=True)
        ten = (ten - mean) / std
        return ten

    def _read_tile_pair(y, x):
        win = Window(x, y, tile_size, tile_size)
        t1 = _read_tile_from_ds(ds1, win, tile_size, indexes=idx1)
        if not need_reproject_t2:
            t2 = _read_tile_from_ds(ds2, win, tile_size, indexes=idx2)
        else:
            t2 = _read_tile2_aligned_to_tile1(ds2, ds1.transform, ds1.crs, win, tile_size, indexes=idx2)
        return t1, t2, y, x

    n_tiles_y = (H_pad + stride - 1) // stride
    n_tiles_x = (W_pad + stride - 1) // stride
    total_tiles = int(n_tiles_y * n_tiles_x)

    model.eval()
    tile_idx = 0
    buf_t1, buf_t2, buf_pos = [], [], []
    t0 = time.time()
    last_print = 0
    progress_every = 200

    num_classes = int(num_classes)
    # rolling stripe cache: only keep active rows [active_start, active_end)
    active_start = 0
    active_end = 0
    active_sum = np.zeros((0, W_pad, num_classes), dtype=pred_sum_dtype)
    active_weight = np.zeros((0, W_pad), dtype=np.float32)
    den_floor = 1e-20

    # --- 性能优化：预分配 batch numpy buffer，避免每次 np.stack 重新分配 ---
    _batch_buf_np = np.empty((batch_size, tile_size, tile_size, 3), dtype=np.float32)

    # --- 性能优化：检测是否支持 FP16 推理 ---
    use_amp = device.type == "cuda"

    def _ensure_active_end(required_end: int):
        nonlocal active_end, active_sum, active_weight
        if required_end <= active_end:
            return
        grow = int(required_end - active_end)
        if grow <= 0:
            return
        sum_grow = np.zeros((grow, W_pad, num_classes), dtype=pred_sum_dtype)
        w_grow = np.zeros((grow, W_pad), dtype=np.float32)
        if active_sum.shape[0] == 0:
            active_sum = sum_grow
            active_weight = w_grow
        else:
            active_sum = np.concatenate([active_sum, sum_grow], axis=0)
            active_weight = np.concatenate([active_weight, w_grow], axis=0)
        active_end = required_end

    def _finalize_rows(finalize_to: int):
        nonlocal active_start, active_sum, active_weight
        finalize_to = max(active_start, min(finalize_to, H_pad))
        n = int(finalize_to - active_start)
        if n <= 0:
            return
        w_blk = active_weight[:n, :]
        s_blk = active_sum[:n, :, :].astype(np.float32, copy=False)
        valid = w_blk[:, :, np.newaxis] > 0
        den = np.maximum(w_blk[:, :, np.newaxis], den_floor)
        chunk = np.zeros((n, W_pad, num_classes), dtype=np.float32)
        np.divide(s_blk, den, out=chunk, where=valid)
        full_pred[active_start:finalize_to, :] = np.argmax(chunk, axis=-1).astype(np.uint8)
        active_sum = active_sum[n:, :, :]
        active_weight = active_weight[n:, :]
        active_start = finalize_to

    def _prep_batch_fast(tiles_list):
        """优化版：直接填充预分配 buffer，避免 np.stack 开销"""
        B = len(tiles_list)
        buf = _batch_buf_np[:B]
        for i, t in enumerate(tiles_list):
            np.copyto(buf[i], t, casting='unsafe')
        buf_slice = buf if B == batch_size else buf[:B]
        arr = buf_slice / float(input_divisor)
        ten = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))  # [B,3,H,W]
        ten = ten.to(device=device, non_blocking=True)
        ten = (ten - mean) / std
        return ten

    def _flush_buffers():
        nonlocal buf_t1, buf_t2, buf_pos
        if not buf_t1:
            return
        ten1 = _prep_batch_fast(buf_t1)
        ten2 = _prep_batch_fast(buf_t2)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(ten1, ten2)  # [B,C,H,W]
        if logits.shape[1] != num_classes:
            raise ValueError(f"Model output channels={logits.shape[1]} != num_classes={num_classes}")
        probs = torch.softmax(logits.float(), dim=1)
        if probs.shape[-2:] != (tile_size, tile_size):
            probs = torch.nn.functional.interpolate(probs, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
        score = probs.cpu().numpy()  # [B,C,tile,tile]

        for b, (py, px) in enumerate(buf_pos):
            h_valid = min(tile_size, H_pad - py)
            w_valid = min(tile_size, W_pad - px)
            p = score[b, :, :h_valid, :w_valid]  # [C,h,w]
            w = weight_map[:h_valid, :w_valid]
            add_blk = np.transpose(p, (1, 2, 0)) * w[:, :, None]
            if active_sum.dtype == np.float16:
                add_blk = add_blk.astype(np.float16)
            _ensure_active_end(py + h_valid)
            y0 = py - active_start
            y1 = y0 + h_valid
            active_sum[y0:y1, px : px + w_valid, :] += add_blk
            active_weight[y0:y1, px : px + w_valid] += w
        buf_t1, buf_t2, buf_pos = [], [], []

    # --- 性能优化：单线程预读取，通过队列让 I/O 和 GPU 推理 pipeline 化 ---
    prefetch_size = batch_size * 2  # 预读取队列深度
    tile_queue = Queue(maxsize=prefetch_size)
    _SENTINEL = object()
    print(f"[PERF] use_amp={use_amp}, prefetch={prefetch_size}, batch_size={batch_size}", flush=True)

    def _producer():
        """后台线程：顺序读取 tile 并送入队列，每行末尾插入行结束标记"""
        for y in range(0, H_pad, stride):
            for x in range(0, W_pad, stride):
                t1, t2, yy, xx = _read_tile_pair(y, x)
                ts = tile_size
                if (t1[0, 0, 0] <= 0 and t1[0, ts-1, 0] <= 0 and t1[ts-1, 0, 0] <= 0 and
                    t2[0, 0, 0] <= 0 and t2[0, ts-1, 0] <= 0 and t2[ts-1, 0, 0] <= 0):
                    if t1.max() <= 0 and t2.max() <= 0:
                        tile_queue.put(None)  # skipped tile
                        continue
                tile_queue.put((t1, t2, yy, xx))
            tile_queue.put(("ROW_END", y))  # 行结束标记
        tile_queue.put(_SENTINEL)  # 全部结束

    reader_thread = Thread(target=_producer, daemon=True)
    reader_thread.start()

    with torch.inference_mode():
        while True:
            item = tile_queue.get()
            if item is _SENTINEL:
                _flush_buffers()
                break

            # 行结束标记：flush + finalize
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "ROW_END":
                _flush_buffers()
                _finalize_rows(item[1] + stride)
                continue

            # 跳过的 tile
            if item is None:
                tile_idx += 1
                continue

            t1, t2, yy, xx = item
            tile_idx += 1
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
                _flush_buffers()

        reader_thread.join(timeout=5)

    print(
        f"[FUSE-STRIP] tiles done; finalizing remaining rows (H_pad,W_pad,C)=({H_pad},{W_pad},{num_classes})...",
        flush=True,
    )
    _finalize_rows(H_pad)
    print("[FUSE-STRIP] argmax done", flush=True)
    out = np.array(full_pred[:H, :W], copy=True)
    if auto_memmap and bool(cleanup_memmap):
        del full_pred
        _cleanup_memmap_paths([pred_out_path], remove_parent_dir=True)
    return out, ds1.transform, ds1.crs


# --------- deployment defaults (same style as predict_BCD.py) ----------
TILE_SIZE = 512
OVERLAP = 128
BATCH_SIZE = 24
INPUT_DIVISOR = 255.0

NUM_CLASSES = 31
EXPORT_CLASS_IDS = None  # None => 1..NUM_CLASSES-1

REMOVE_HOLES_AREA = 200000
REMOVE_OBJECTS_MIN_SIZE = 800
FILL_ALL_HOLES = True
MIN_AREA_MU = 0.1
OVERLAP_RATIO = 0.90
DP_TOLERANCE_M = 1.0
DISABLE_CHAIKIN = False
CHAIKIN_ITERATIONS = 1
CHAIKIN_WEIGHT = 0.8
POSTPROCESS = True

MEMMAP_FP16 = True
CLEANUP_MEMMAP = True

CUDA_DEVICE = 0

MODEL_MCD_PATH = None
DEFAULT_MCD_WEIGHT_BASENAME = "best-model-step=005899-val_iou=0.2454.pt"


def _default_mcd_weight_path() -> str:
    # predict_MCD.py: aqmi/change_detection/change3d/predict_MCD.py
    # -> repo root: ../../../../
    repo_root = Path(__file__).resolve().parents[3]
    return str(
        repo_root
        / "aqmi"
        / "change_detection"
        / "change3d"
        / "model_MCD"
        / DEFAULT_MCD_WEIGHT_BASENAME
    )


def _setup() -> str:
    """
    Deployment-style setup:
    - only resolve model weight path
    """
    global MODEL_MCD_PATH
    # Priority: explicit path from test/runtime env first, then MCD-specific env, then default.
    MODEL_MCD_PATH = (
        os.environ.get("MODEL_CHANGE3D_PATH")
        or os.environ.get("MODEL_MCD_PATH")
        or os.environ.get("CHANGE_DETECTION_CHANGE3D_MCD_MODEL_PATH")
    )
    if MODEL_MCD_PATH is None:
        MODEL_MCD_PATH = _default_mcd_weight_path()
    return MODEL_MCD_PATH


def _pick_tif_pair(input_folder: str) -> Tuple[str, str]:
    tif_files = [
        str(p)
        for p in Path(input_folder).iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    tif_files.sort()
    if len(tif_files) < 2:
        raise ValueError(f"need at least 2 tif files, found {len(tif_files)} in {input_folder}")

    by_base: Dict[str, Dict[str, str]] = {}
    for f in tif_files:
        stem = Path(f).stem.lower()
        if stem.endswith("_t1"):
            base = stem[: -len("_t1")]
            by_base.setdefault(base, {})["t1"] = f
        elif stem.endswith("_t2"):
            base = stem[: -len("_t2")]
            by_base.setdefault(base, {})["t2"] = f

    for base in sorted(by_base.keys()):
        pair = by_base[base]
        if "t1" in pair and "t2" in pair and pair["t1"] != pair["t2"]:
            return pair["t1"], pair["t2"]

    t1 = next((f for f in tif_files if "_t1" in f.lower() or "t1" in Path(f).stem.lower()), None)
    t2 = next((f for f in tif_files if "_t2" in f.lower() or "t2" in Path(f).stem.lower()), None)
    if t1 and t2 and t1 != t2:
        return t1, t2

    return tif_files[0], tif_files[1]


def _build_change_shp_name(tif1: str, tif2: str) -> str:
    stem1 = Path(tif1).stem
    stem2 = Path(tif2).stem

    def _strip_time_suffix(stem: str) -> str:
        low = stem.lower()
        if low.endswith("_t1") or low.endswith("_t2"):
            return stem[:-3]
        return stem

    def _split_region_year(stem: str) -> Tuple[str, Optional[str]]:
        s = _strip_time_suffix(stem)
        parts = s.split("_")
        if len(parts) >= 2 and re.fullmatch(r"\d{4}", parts[-1]):
            return "_".join(parts[:-1]), parts[-1]
        return s, None

    region1, year1 = _split_region_year(stem1)
    region2, year2 = _split_region_year(stem2)
    if year1 and year2:
        if region1 == region2 and region1:
            return f"change_{region1}_{year1}_{year2}.shp"
        common = os.path.commonprefix([region1, region2]).rstrip("_")
        if common:
            return f"change_{common}_{year1}_{year2}.shp"
        return f"change_{year1}_{year2}.shp"
    return "change_polygons.shp"


def process(input_folder: str, output_folder: str) -> bool:
    """
    Deployment entry (same form as predict_BCD.py):
    - validate input folder
    - resolve model by _setup()
    - run multiclass streamed inference
    - save final shp under output_folder
    """
    global MODEL_MCD_PATH

    if not os.path.exists(input_folder):
        return False
    if not os.path.isdir(input_folder):
        return False

    supported_files = []
    for ext in (".tif", ".tiff"):
        supported_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        supported_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    if not supported_files:
        return False

    os.makedirs(output_folder, exist_ok=True)

    if MODEL_MCD_PATH is None:
        _setup()
    if MODEL_MCD_PATH is None or not os.path.exists(MODEL_MCD_PATH):
        return False

    tif1, tif2 = _pick_tif_pair(input_folder)
    try:
        run_info = (
            f"tif1={tif1}\n"
            f"tif2={tif2}\n"
            f"model_mcd_pt={MODEL_MCD_PATH}\n"
        )
        print("[RUN] " + run_info.replace("\n", "  ").strip(), flush=True)
        with open(os.path.join(output_folder, "run_info.txt"), "w", encoding="utf-8") as f:
            f.write(run_info)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
        device = torch.device("cuda:0")

    try:
        model = torch.jit.load(MODEL_MCD_PATH, map_location=device)
        model.eval()
    except Exception:
        return False

    out_shp = os.path.join(output_folder, _build_change_shp_name(tif1, tif2))

    try:
        with rasterio.open(tif1) as ds1, rasterio.open(tif2) as ds2:
            if ds1.crs is None or ds2.crs is None:
                raise ValueError("CRS is None for one of the inputs.")
            if (ds1.width, ds1.height) != (ds2.width, ds2.height):
                raise ValueError("Input size mismatch. Please align/resample first.")

            pred_mask_mc, transform, crs = run_inference_streamed_to_mask_multiclass(
                model=model,
                device=device,
                ds1=ds1,
                ds2=ds2,
                tile_size=TILE_SIZE,
                overlap=OVERLAP,
                batch_size=BATCH_SIZE,
                num_classes=NUM_CLASSES,
                input_divisor=INPUT_DIVISOR,
                memmap_dir=None,
                memmap_fp16=bool(MEMMAP_FP16),
                cleanup_memmap=bool(CLEANUP_MEMMAP),
            )

        export_class_ids = EXPORT_CLASS_IDS if EXPORT_CLASS_IDS is not None else list(range(1, int(NUM_CLASSES)))
        save_multiclass_mask_to_shapefile(
            mask=pred_mask_mc,
            transform=transform,
            crs=crs,
            out_shp_path=out_shp,
            num_classes=NUM_CLASSES,
            export_class_ids=export_class_ids,
            min_area_mu=MIN_AREA_MU,
            overlap_ratio=OVERLAP_RATIO,
            dp_tolerance_m=DP_TOLERANCE_M,
            disable_chaikin=DISABLE_CHAIKIN,
            chaikin_iterations=CHAIKIN_ITERATIONS,
            chaikin_weight=CHAIKIN_WEIGHT,
            remove_holes_area=REMOVE_HOLES_AREA,
            remove_objects_min_size=REMOVE_OBJECTS_MIN_SIZE,
            fill_all_holes=FILL_ALL_HOLES,
            postprocess=bool(POSTPROCESS),
        )
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser("Change3D BCD: big tif streamed inference + shp")
    parser.add_argument("--tif1", type=str, default=r"E:\随州\平林镇\平林镇_2018_t1.tif", help="time1/pre GeoTIFF path (RGB)")
    parser.add_argument("--tif2", type=str, default=r"E:\随州\平林镇\平林镇_2020_t2.tif", help="time2/post GeoTIFF path (RGB)")
    # Backward-compatible single-pt option (used as MCD pt when --mcd_pt_path not provided)
    parser.add_argument("--pt_path", type=str, default=r"D:\Projects\change_detection\Change3D\checkpoint\\1PF_1111\best-model-step=012496-val_miou=0.1847.pt", help="exported TorchScript .pt (scripted)")
    parser.add_argument(
        "--mcd_pt_path",
        type=str,
        default="",
        help="MCD TorchScript .pt to predict multi-class logits (C channels). If empty, fallback to --pt_path.",
    )

    parser.add_argument("--out_dir", type=str, default=r"D:\Projects\change_detection\Change3D\outputs_1\MCD\\1PF_1111\step12496\平林镇", help="output dir for shp")
    parser.add_argument("--out_name", type=str, default="洪山镇_1111_s2496", help="shp name")

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=24)

    parser.add_argument("--input_divisor", type=float, default=255.0, help="divide raw tile by this before normalization")

    # Multi-class prediction
    parser.add_argument("--num_classes", type=int, default=31, help="Number of classes for MCD multi-class logits output")
    parser.add_argument("--export_class_ids", type=str, default="", help="Comma-separated class ids to export. Empty => export 1..num_classes-1")

    parser.add_argument("--remove_holes_area", type=int, default=200000)
    parser.add_argument("--remove_objects_min_size", type=int, default=800)
    parser.add_argument("--fill_all_holes",default=True)
    parser.add_argument("--min_area_mu", type=float, default=0.1)
    parser.add_argument("--overlap_ratio", type=float, default=0.90)
    parser.add_argument("--dp_tolerance_m", type=float, default=1.0, help="Douglas-Peucker tolerance (meters)")
    parser.add_argument("--disable_chaikin", action="store_true", help="disable Chaikin smoothing")
    parser.add_argument("--chaikin_iterations", type=int, default=1)
    parser.add_argument("--chaikin_weight", type=float, default=0.8)

    parser.add_argument(
        "--postprocess",
        type=int,
        default=1,
        help="Unified postprocess switch: 1 enables mask + vector postprocess; 0 exports raw polygonize result.",
    )

    parser.add_argument("--memmap_dir", type=str, default='', help="Use memmap to reduce RAM")
    parser.add_argument("--memmap_fp16", type=int, default=1, help="Use float16 for multiclass pred_sum memmap (1/0)")
    parser.add_argument("--cleanup_memmap", type=int, default=1, help="Delete memmap temp files after inference (1/0)")

    parser.add_argument("--strict_crs", action="store_true", help="abort if tif1/tif2 CRS differ")

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("args.device=cuda but CUDA is not available")
        n_gpu = torch.cuda.device_count()
        if int(args.gpu_id) < 0 or int(args.gpu_id) >= n_gpu:
            raise ValueError(f"Invalid --gpu_id={args.gpu_id}; available CUDA devices: 0..{n_gpu-1}")
        device = torch.device(f"cuda:{int(args.gpu_id)}")
    else:
        device = torch.device("cpu")
    print(f"[DEVICE] inference device: {device}", flush=True)

    mcd_pt = args.mcd_pt_path.strip() or args.pt_path
    if not mcd_pt:
        raise ValueError("Please provide --mcd_pt_path (or --pt_path as fallback).")

    print(f"[PT] loading MCD torchscript: {mcd_pt}", flush=True)
    mcd_model = torch.jit.load(mcd_pt, map_location=device)
    mcd_model.eval()

    if int(args.num_classes) <= 1:
        raise ValueError("This script is multiclass-only now. Please set --num_classes > 1.")

    tif1_base = osp(os.path.splitext(os.path.basename(args.tif1))[0])
    weight_tag = osp(os.path.splitext(os.path.basename(mcd_pt))[0])
    out_name = args.out_name.strip() or weight_tag
    out_shp = osp(args.out_dir, "change.shp")

    with rasterio.open(args.tif1) as ds1, rasterio.open(args.tif2) as ds2:
        if ds1.crs is None or ds2.crs is None:
            raise ValueError("CRS is None for one of the inputs. Please define CRS before running.")
        if (ds1.width, ds1.height) != (ds2.width, ds2.height):
            raise ValueError("Input size mismatch. Please align/resample first (as in predict_big_pt_flow.py).")
        if ds1.crs != ds2.crs and args.strict_crs:
            raise ValueError(f"CRS mismatch: tif1={ds1.crs} tif2={ds2.crs}")

        print(f"[TIF] tif1={args.tif1}", flush=True)
        print(f"[TIF] tif2={args.tif2}", flush=True)
        print(f"[RUN] tile_size={args.tile_size} overlap={args.overlap} batch_size={args.batch_size} num_classes={args.num_classes}", flush=True)

        pred_mask_mc, transform, crs = run_inference_streamed_to_mask_multiclass(
            model=mcd_model,
            device=device,
            ds1=ds1,
            ds2=ds2,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            input_divisor=args.input_divisor,
            memmap_dir=args.memmap_dir,
            memmap_fp16=bool(int(args.memmap_fp16)),
            cleanup_memmap=bool(int(args.cleanup_memmap)),
        )

    print("[POST] postprocess mask...", flush=True)

    export_class_ids = None
    if str(args.export_class_ids).strip():
        export_class_ids = [int(x) for x in str(args.export_class_ids).split(",") if x.strip()]
    else:
        export_class_ids = list(range(1, int(args.num_classes)))

    print("[SHP] multiclass polygonize + save...", flush=True)
    save_multiclass_mask_to_shapefile(
        mask=pred_mask_mc,
        transform=transform,
        crs=crs,
        out_shp_path=out_shp,
        num_classes=args.num_classes,
        export_class_ids=export_class_ids,
        min_area_mu=args.min_area_mu,
        overlap_ratio=args.overlap_ratio,
        dp_tolerance_m=args.dp_tolerance_m,
        disable_chaikin=args.disable_chaikin,
        chaikin_iterations=args.chaikin_iterations,
        chaikin_weight=args.chaikin_weight,
        remove_holes_area=args.remove_holes_area,
        remove_objects_min_size=args.remove_objects_min_size,
        fill_all_holes=args.fill_all_holes,
        postprocess=bool(int(args.postprocess)),
    )


if __name__ == "__main__":
    main()

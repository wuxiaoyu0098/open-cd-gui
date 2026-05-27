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
from os.path import join as osp
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes

import geopandas as gpd
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
    os.makedirs(memmap_dir, exist_ok=True)
    path = os.path.join(memmap_dir, f"{name}.dat")
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    gb = nbytes / (1024**3)

    # 优化：使用 seek + write 创建稀疏文件（默认内容为 0），避免 arr[:] = 0 的大规模写盘
    with open(path, "wb") as f:
        if nbytes > 0:
            f.seek(nbytes - 1)
            f.write(b"\x00")

    # 以 r+ 模式打开 memmap，避免重新初始化
    arr = np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    print(f"[MEMMAP] created {name}.dat shape={shape} ~{gb:.2f} GiB", flush=True)
    return arr, path


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


MU_TO_M2 = 666.67


def _to_area_crs(gdf: gpd.GeoDataFrame):
    is_geographic = False
    epsg_code = None
    try:
        is_geographic = bool(gdf.crs is not None and gdf.crs.is_geographic)
    except Exception:
        is_geographic = False
    if not is_geographic:
        return gdf, False, None

    bounds = gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2.0
    center_lat = (bounds[1] + bounds[3]) / 2.0
    utm_zone = int(np.floor((center_lon + 180.0) / 6.0) + 1)
    epsg_code = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone
    return gdf.to_crs(epsg=epsg_code), True, epsg_code


def _remove_contained(gdf: gpd.GeoDataFrame, gdf_chk: gpd.GeoDataFrame, overlap_ratio: float = 0.90) -> gpd.GeoDataFrame:
    if len(gdf) <= 1:
        return gdf
    contained = []
    for i in range(len(gdf)):
        gi = gdf_chk.iloc[i].geometry
        ai = float(gdf.iloc[i]["area_m2"])
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
                        if r >= overlap_ratio:
                            contained.append(i)
                            break
            except Exception:
                continue
    if contained:
        gdf = gdf.drop(gdf.index[contained]).reset_index(drop=True)
    return gdf


def _chaikin(coords, iterations: int = 1, weight: float = 0.8):
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


def _process_geometry_boundary(geom, dp_tolerance_m: float, enable_chaikin: bool, chaikin_iterations: int, chaikin_weight: float):
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


def save_mask_to_shapefile(mask, transform, crs, out_shp_path, change_value=1, min_area_mu=1.0):
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
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.is_empty == False].copy()
    if gdf.empty:
        gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    # Project first, then apply boundary smoothing/simplification in meter units.
    gdf_proj, is_geographic, _epsg_code = _to_area_crs(gdf)
    gdf_proj["geometry"] = gdf_proj.geometry.apply(
        lambda geom: _process_geometry_boundary(
            geom,
            dp_tolerance_m=0.95,
            enable_chaikin=True,
            chaikin_iterations=1,
            chaikin_weight=0.8,
        )
    )
    gdf_proj = gdf_proj[gdf_proj.geometry.notna()].copy()
    gdf_proj = gdf_proj[gdf_proj.geometry.is_empty == False].copy()
    if gdf_proj.empty:
        gdf_proj.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return

    areas_m2 = gdf_proj.geometry.area
    gdf_proj["area_m2"] = areas_m2.values
    gdf_proj["area_mu"] = gdf_proj["area_m2"] / MU_TO_M2

    min_area_m2 = float(min_area_mu) * MU_TO_M2
    before = len(gdf_proj)
    gdf_proj = gdf_proj[gdf_proj["area_m2"] >= min_area_m2].copy()
    after = len(gdf_proj)
    print(f"[SHP] polygons: {before} -> {after} after area filter >= {min_area_mu} mu", flush=True)

    if gdf_proj.empty:
        gdf_proj.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
        return

    gdf_proj = gdf_proj.sort_values("area_m2", ascending=False).reset_index(drop=True)
    gdf_proj = _remove_contained(gdf_proj, gdf_proj, overlap_ratio=0.90)

    if is_geographic and gdf.crs is not None:
        gdf_out = gdf_proj.to_crs(gdf.crs)
    else:
        gdf_out = gdf_proj

    gdf_out.to_file(out_shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[SHP] saved: {out_shp_path}", flush=True)


def remove_shapefile(path: str):
    stem, _ = os.path.splitext(path)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"):
        sidecar = stem + ext
        if os.path.exists(sidecar):
            os.remove(sidecar)


# --------- deployment defaults (same style as predict0915部署.py) ----------
# 推理参数不通过环境变量配置，保持脚本内部固定默认值（需要改就改这里常量）。
TILE_SIZE = 512
OVERLAP = 128
BATCH_SIZE = 16
PROB_THRESHOLD = 0.5
INPUT_DIVISOR = 255.0

REMOVE_HOLES_AREA = 200000
REMOVE_OBJECTS_MIN_SIZE = 800
FILL_ALL_HOLES = True
MIN_AREA_MU = 0.1

CUDA_DEVICE = 0

MODEL_CHANGE3D_PATH = None
DEFAULT_CHANGE3D_WEIGHT_BASENAME = "best-model-step=005899-val_iou=0.2454.pt"


def _default_change3d_weight_path() -> str:
    # predict_3d.py: aqmi/change_detection/change3d/predict_3d.py
    # -> repo root: ../../../../
    repo_root = Path(__file__).resolve().parents[3]
    return str(
        repo_root
        / "aqmi"
        / "change_detection"
        / "change3d"
        / "model"
        / DEFAULT_CHANGE3D_WEIGHT_BASENAME
    )


def _setup() -> str:
    """
    与 predict0915部署.py 相同逻辑：
    - 只负责解析模型权重路径（可用环境变量覆盖）
    - 不在这里设置推理参数
    """
    global MODEL_CHANGE3D_PATH

    # 兼容两套命名：优先 MODEL_CHANGE3D_PATH（类似 MODEL_DRY_LAND_PATH 的风格）
    MODEL_CHANGE3D_PATH = os.environ.get("MODEL_CHANGE3D_PATH") or os.environ.get("CHANGE_DETECTION_CHANGE3D_MODEL_PATH")

    if MODEL_CHANGE3D_PATH is None:
        MODEL_CHANGE3D_PATH = _default_change3d_weight_path()

    return MODEL_CHANGE3D_PATH


def _pick_tif_pair(input_folder: str) -> Tuple[str, str]:
    """
    Pick (t1, t2) from input folder.
    Prefer same basename prefix: 2020_t1.tif + 2020_t2.tif.
    """
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

    # fallback: first match by substring
    t1 = next((f for f in tif_files if "_t1" in f.lower() or "t1" in Path(f).stem.lower()), None)
    t2 = next((f for f in tif_files if "_t2" in f.lower() or "t2" in Path(f).stem.lower()), None)
    if t1 and t2 and t1 != t2:
        return t1, t2

    return tif_files[0], tif_files[1]


def _build_change_shp_name(tif1: str, tif2: str) -> str:
    """
    Build shp filename like:
      安居镇_2018_t1.tif + 安居镇_2020_t2.tif -> change_安居镇_2018_2020.shp
    Fallback to change_polygons.shp when parsing fails.
    """
    stem1 = Path(tif1).stem
    stem2 = Path(tif2).stem

    def _strip_time_suffix(stem: str) -> str:
        s = stem
        low = s.lower()
        if low.endswith("_t1") or low.endswith("_t2"):
            return s[:-3]
        return s

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

        # If region differs, keep common prefix when available.
        common = os.path.commonprefix([region1, region2]).rstrip("_")
        if common:
            return f"change_{common}_{year1}_{year2}.shp"
        return f"change_{year1}_{year2}.shp"

    return "change_polygons.shp"


def process(input_folder: str, output_folder: str) -> bool:
    """
    与 predict0915部署.py 的 process 相同风格：
    - 做输入检查
    - 如未设置模型路径则 _setup()
    - 检查权重存在
    - 固定默认推理参数（不从环境变量读取）
    - 输出写入 output_folder
    """
    global MODEL_CHANGE3D_PATH

    if not os.path.exists(input_folder):
        return False
    if not os.path.isdir(input_folder):
        return False

    # 与 predict0915部署.py 一样：先确认有 tif 文件
    supported_files = []
    for ext in (".tif", ".tiff"):
        supported_files.extend(list(Path(input_folder).glob(f"*{ext}")))
        supported_files.extend(list(Path(input_folder).glob(f"*{ext.upper()}")))
    if not supported_files:
        return False

    os.makedirs(output_folder, exist_ok=True)

    if MODEL_CHANGE3D_PATH is None:
        _setup()

    if MODEL_CHANGE3D_PATH is None or not os.path.exists(MODEL_CHANGE3D_PATH):
        return False

    tif1, tif2 = _pick_tif_pair(input_folder)
    # 输出本次使用的输入与权重，便于追溯
    try:
        run_info = (
            f"tif1={tif1}\n"
            f"tif2={tif2}\n"
            f"model_pt={MODEL_CHANGE3D_PATH}\n"
        )
        print("[RUN] " + run_info.replace("\n", "  ").strip(), flush=True)
        with open(os.path.join(output_folder, "run_info.txt"), "w", encoding="utf-8") as f:
            f.write(run_info)
    except Exception:
        # 不影响主流程
        pass

    # 与 predict0915部署.py 类似：固定 CUDA_DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
        device = torch.device("cuda:0")

    model = torch.jit.load(MODEL_CHANGE3D_PATH, map_location=device)
    model.eval()

    out_shp = os.path.join(output_folder, _build_change_shp_name(tif1, tif2))

    try:
        with rasterio.open(tif1) as ds1, rasterio.open(tif2) as ds2:
            if ds1.crs is None or ds2.crs is None:
                raise ValueError("CRS is None for one of the inputs.")
            if (ds1.width, ds1.height) != (ds2.width, ds2.height):
                raise ValueError("Input size mismatch. Please align/resample first.")

            pred_mask, transform, crs = run_inference_streamed_to_mask(
                model=model,
                device=device,
                ds1=ds1,
                ds2=ds2,
                tile_size=TILE_SIZE,
                overlap=OVERLAP,
                batch_size=BATCH_SIZE,
                prob_threshold=PROB_THRESHOLD,
                input_divisor=INPUT_DIVISOR,
                memmap_dir=None,
            )

        pred_mask = postprocess_mask(
            pred_mask,
            remove_holes_area=REMOVE_HOLES_AREA,
            remove_objects_min_size=REMOVE_OBJECTS_MIN_SIZE,
            fill_all_holes=FILL_ALL_HOLES,
        )

        save_mask_to_shapefile(
            mask=pred_mask,
            transform=transform,
            crs=crs,
            out_shp_path=out_shp,
            change_value=1,
            min_area_mu=MIN_AREA_MU,
        )
        return True
    except Exception as e:
        return False


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
    full_pred_sum, _ = _maybe_memmap((H_pad, W_pad), np.float32, memmap_dir if auto_memmap else None, "pred_sum")
    full_weight, _ = _maybe_memmap((H_pad, W_pad), np.float32, memmap_dir if auto_memmap else None, "weight")
    full_vote_cnt, _ = _maybe_memmap((H_pad, W_pad), np.int32, memmap_dir if auto_memmap else None, "vote")

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

    with torch.inference_mode():
        for y in range(0, H_pad, stride):
            for x in range(0, W_pad, stride):
                t1, t2, yy, xx = _read_tile_pair(y, x)
                tile_idx += 1

                # skip mostly-empty tiles
                # (若你数据 nodata 不为 0，可通过 input_divisor/阈值在代码里调整)
                if t1.max() <= 0 and t2.max() <= 0:
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
                    ten1 = _prep_batch(buf_t1)
                    ten2 = _prep_batch(buf_t2)
                    prob = model(ten1, ten2)  # [B,1,H,W]
                    if prob.shape[-2:] != (tile_size, tile_size):
                        prob = torch.nn.functional.interpolate(prob, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                    score = prob[:, 0].float().cpu().numpy()  # [B,H,W]

                    for b, (py, px) in enumerate(buf_pos):
                        h_valid = min(tile_size, H_pad - py)
                        w_valid = min(tile_size, W_pad - px)
                        p = score[b, :h_valid, :w_valid]
                        w = weight_map[:h_valid, :w_valid]
                        full_pred_sum[py : py + h_valid, px : px + w_valid] += p * w
                        full_weight[py : py + h_valid, px : px + w_valid] += w
                        full_vote_cnt[py : py + h_valid, px : px + w_valid] += 1

                    buf_t1, buf_t2, buf_pos = [], [], []

        # last partial batch
        if buf_t1:
            ten1 = _prep_batch(buf_t1)
            ten2 = _prep_batch(buf_t2)
            prob = model(ten1, ten2)
            if prob.shape[-2:] != (tile_size, tile_size):
                prob = torch.nn.functional.interpolate(prob, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
            score = prob[:, 0].float().cpu().numpy()
            for b, (py, px) in enumerate(buf_pos):
                h_valid = min(tile_size, H_pad - py)
                w_valid = min(tile_size, W_pad - px)
                p = score[b, :h_valid, :w_valid]
                w = weight_map[:h_valid, :w_valid]
                full_pred_sum[py : py + h_valid, px : px + w_valid] += p * w
                full_weight[py : py + h_valid, px : px + w_valid] += w
                full_vote_cnt[py : py + h_valid, px : px + w_valid] += 1

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

    return full_pred[:H, :W], ds1.transform, ds1.crs


def main():
    parser = argparse.ArgumentParser("Change3D BCD: big tif streamed inference + shp")
    parser.add_argument("--tif1", type=str, default="/mnt/wuxy/change_detection/Changedino_dynamic_all_elements/tif_data/随州/安居镇/clip/2018.tif", help="time1/pre GeoTIFF path (RGB)")
    parser.add_argument("--tif2", type=str, default="/mnt/wuxy/change_detection/Changedino_dynamic_all_elements/tif_data/随州/安居镇/clip/2020.tif", help="time2/post GeoTIFF path (RGB)")
    parser.add_argument(
        "--pt_path",
        type=str,
        default=_default_change3d_weight_path(),
        help="exported Change3D TorchScript .pt (scripted)",
    )

    parser.add_argument("--out_dir", type=str, default="outputs", help="output dir for shp")
    parser.add_argument("--out_name", type=str, default="安居镇_clip", help="optional shp name suffix")

    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--input_divisor", type=float, default=255.0, help="divide raw tile by this before normalization")

    parser.add_argument("--remove_holes_area", type=int, default=200000)
    parser.add_argument("--remove_objects_min_size", type=int, default=800)
    parser.add_argument("--fill_all_holes", default=True)
    parser.add_argument("--min_area_mu", type=float, default=0.1)

    parser.add_argument("--memmap_dir", type=str, default=None, help="Use memmap to reduce RAM")

    parser.add_argument("--strict_crs", action="store_true", help="abort if tif1/tif2 CRS differ")

    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")

    print(f"[PT] loading torchscript: {args.pt_path}", flush=True)
    model = torch.jit.load(args.pt_path, map_location=device)
    model.eval()

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
        print(f"[RUN] tile_size={args.tile_size} overlap={args.overlap} batch_size={args.batch_size} prob_thr={args.prob_threshold}", flush=True)

        pred_mask, transform, crs = run_inference_streamed_to_mask(
            model=model,
            device=device,
            ds1=ds1,
            ds2=ds2,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            prob_threshold=args.prob_threshold,
            input_divisor=args.input_divisor,
            memmap_dir=args.memmap_dir,
        )

    print("[POST] postprocess mask...", flush=True)
    pred_mask = postprocess_mask(
        pred_mask,
        remove_holes_area=args.remove_holes_area,
        remove_objects_min_size=args.remove_objects_min_size,
        fill_all_holes=args.fill_all_holes,
    )

    print("[SHP] polygonize + save...", flush=True)
    save_mask_to_shapefile(
        mask=pred_mask,
        transform=transform,
        crs=crs,
        out_shp_path=out_shp,
        change_value=1,
        min_area_mu=args.min_area_mu,
    )


if __name__ == "__main__":
    main()


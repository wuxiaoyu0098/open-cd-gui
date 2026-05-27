# Copyright (c) Open-CD. All rights reserved.
import argparse
import os
import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from rasterio.features import shapes
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

import opencd  # noqa: F401
import rasterio


os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


def pad_len(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return tile_size
    n = (length - tile_size + stride - 1) // stride
    return n * stride + tile_size


def make_weight_map(tile_size: int, min_weight: float = 0.85) -> np.ndarray:
    if tile_size <= 1:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    center = (tile_size - 1) / 2.0
    y, x = np.ogrid[:tile_size, :tile_size]
    dist = np.sqrt((y - center)**2 + (x - center)**2)
    max_dist = np.sqrt(center**2 + center**2)
    weight = min_weight + (1.0 - min_weight) * (1.0 - dist /
                                                (max_dist + 1e-8))
    return np.clip(weight, min_weight, 1.0).astype(np.float32)


def rgb_indexes(ds) -> Tuple[int, int, int]:
    if ds.count >= 3:
        return (1, 2, 3)
    if ds.count == 2:
        return (1, 2, 2)
    return (1, 1, 1)


def read_tile(ds, win: Window, tile_size: int,
              indexes: Sequence[int]) -> np.ndarray:
    x0 = max(0, int(win.col_off))
    y0 = max(0, int(win.row_off))
    x1 = min(ds.width, int(win.col_off + win.width))
    y1 = min(ds.height, int(win.row_off + win.height))
    dtype = np.dtype(ds.dtypes[0])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((tile_size, tile_size, len(indexes)), dtype=dtype)

    clipped = Window(x0, y0, x1 - x0, y1 - y0)
    arr = ds.read(indexes=indexes, window=clipped)
    arr = np.transpose(arr, (1, 2, 0))
    h, w = arr.shape[:2]
    if h < tile_size or w < tile_size:
        arr = np.pad(
            arr,
            ((0, tile_size - h), (0, tile_size - w), (0, 0)),
            mode='reflect')
    return arr[:tile_size, :tile_size, :]


def read_tile_reprojected(ds2, ref_transform, ref_crs, win: Window,
                          tile_size: int,
                          indexes: Sequence[int]) -> np.ndarray:
    from rasterio.windows import transform as window_transform

    dst = np.zeros((len(indexes), tile_size, tile_size), dtype=ds2.dtypes[0])
    dst_transform = window_transform(win, ref_transform)
    for i, band in enumerate(indexes):
        reproject(
            source=rasterio.band(ds2, band),
            destination=dst[i],
            src_transform=ds2.transform,
            src_crs=ds2.crs,
            dst_transform=dst_transform,
            dst_crs=ref_crs,
            dst_width=tile_size,
            dst_height=tile_size,
            resampling=Resampling.bilinear)
    return np.transpose(dst, (1, 2, 0))


def build_model(config: str, checkpoint: str, device: str):
    cfg = Config.fromfile(config)
    cfg.model.train_cfg = None
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.dataset_meta = getattr(model, 'dataset_meta',
                                 dict(classes=('unchanged', 'changed')))
    model.to(device)
    model.eval()
    return model


def predict_batch(model, tiles_a, tiles_b, device: str) -> np.ndarray:
    data_samples = []
    inputs = []
    for tile_a, tile_b in zip(tiles_a, tiles_b):
        h, w = tile_a.shape[:2]
        pair = np.concatenate([tile_a, tile_b], axis=2)
        tensor = torch.from_numpy(pair.transpose(2, 0, 1)).contiguous()
        inputs.append(tensor)
        sample = SegDataSample()
        sample.set_metainfo(
            dict(
                ori_shape=(h, w),
                img_shape=(h, w),
                pad_shape=(h, w),
                padding_size=[0, 0, 0, 0]))
        data_samples.append(sample)

    probs = []
    with torch.inference_mode():
        for tensor, sample in zip(inputs, data_samples):
            data = dict(inputs=[tensor], data_samples=[sample])
            output = model.test_step(data)[0]
            if 'seg_logits' in output:
                logit = output.seg_logits.data
                if logit.shape[0] == 1:
                    prob = torch.sigmoid(logit[0])
                else:
                    prob = torch.softmax(logit, dim=0)[1]
            else:
                pred = output.pred_sem_seg.data[0]
                prob = pred.float()
            probs.append(prob.detach().to('cpu').numpy().astype(np.float32))
    return np.stack(probs, axis=0)


def postprocess_mask(mask: np.ndarray,
                     remove_holes_area: int = 2000,
                     remove_objects_min_size: int = 100,
                     fill_all_holes: bool = False) -> np.ndarray:
    mask_cv = (mask > 0).astype(np.uint8) * 255
    if fill_all_holes:
        h, w = mask_cv.shape
        marker = np.zeros((h + 2, w + 2), dtype=np.uint8)
        flood = mask_cv.copy()
        cv2.floodFill(flood, marker, (0, 0), 255)
        mask_processed = mask_cv | cv2.bitwise_not(flood)
    else:
        mask_processed = mask_cv.copy()
        inv = 255 - mask_processed
        contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if hierarchy[0][i][3] >= 0 and area < remove_holes_area:
                    cv2.drawContours(mask_processed, [contour], -1, 255, -1)

    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_processed)
    for contour in contours:
        if cv2.contourArea(contour) >= remove_objects_min_size:
            cv2.drawContours(out, [contour], -1, 255, -1)
    return (out > 0).astype(np.uint8)


def save_mask_tif(path: str, mask: np.ndarray, ref_ds):
    profile = ref_ds.profile.copy()
    profile.update(count=1, dtype='uint8', nodata=0, compress='lzw')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_mask_shp(path: str, mask: np.ndarray, transform, crs):
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except Exception as exc:
        print(f'[SHP] skip shapefile output: {exc}', flush=True)
        return

    geoms = []
    vals = []
    for geom, val in shapes(mask.astype(np.int16), mask=mask == 1,
                            transform=transform):
        if val == 1:
            geoms.append(shape(geom))
            vals.append(1)

    gdf = gpd.GeoDataFrame({'value': vals}, geometry=geoms, crs=crs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    remove_shapefile(path)
    gdf.to_file(path, driver='ESRI Shapefile', encoding='utf-8')
    print(f'[SHP] saved: {path}', flush=True)


def remove_shapefile(path: str):
    stem, _ = os.path.splitext(path)
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.fix'):
        sidecar = stem + ext
        if os.path.exists(sidecar):
            os.remove(sidecar)


def run_big_tif(args):
    device = 'cuda:0' if args.device == 'cuda' and torch.cuda.is_available(
    ) else 'cpu'
    model = build_model(args.config, args.checkpoint, device)

    os.makedirs(args.out_dir, exist_ok=True)
    raw_tif = os.path.join(args.out_dir, 'raw_pred_mask.tif')
    post_tif = os.path.join(args.out_dir, 'post_mask.tif')
    shp_path = os.path.join(args.out_dir, 'change.shp')

    stride = args.tile_size - args.overlap
    if stride <= 0:
        raise ValueError('overlap must be smaller than tile_size')

    with rasterio.open(args.tif_a) as ds1, rasterio.open(args.tif_b) as ds2:
        h, w = ds1.height, ds1.width
        h_pad = pad_len(h, args.tile_size, stride)
        w_pad = pad_len(w, args.tile_size, stride)
        pred_sum = np.zeros((h_pad, w_pad), dtype=np.float32)
        weights = np.zeros((h_pad, w_pad), dtype=np.float32)
        weight_map = make_weight_map(args.tile_size)
        idx1 = rgb_indexes(ds1)
        idx2 = rgb_indexes(ds2)
        same_grid = (ds1.crs == ds2.crs and ds1.transform == ds2.transform
                     and ds1.width == ds2.width and ds1.height == ds2.height)

        positions = [(y, x) for y in range(0, h_pad, stride)
                     for x in range(0, w_pad, stride)]
        total = len(positions)
        tiles_a, tiles_b, tile_pos = [], [], []
        start = time.time()

        for idx, (y, x) in enumerate(positions, 1):
            win = Window(x, y, args.tile_size, args.tile_size)
            tile_a = read_tile(ds1, win, args.tile_size, idx1)
            if same_grid:
                tile_b = read_tile(ds2, win, args.tile_size, idx2)
            else:
                tile_b = read_tile_reprojected(ds2, ds1.transform, ds1.crs,
                                               win, args.tile_size, idx2)
            tiles_a.append(tile_a)
            tiles_b.append(tile_b)
            tile_pos.append((y, x))

            if len(tiles_a) >= args.batch_size or idx == total:
                probs = predict_batch(model, tiles_a, tiles_b, device)
                for prob, (py, px) in zip(probs, tile_pos):
                    valid_h = min(args.tile_size, h_pad - py)
                    valid_w = min(args.tile_size, w_pad - px)
                    pred_sum[py:py + valid_h, px:px + valid_w] += (
                        prob[:valid_h, :valid_w] *
                        weight_map[:valid_h, :valid_w])
                    weights[py:py + valid_h,
                            px:px + valid_w] += weight_map[:valid_h, :valid_w]
                tiles_a, tiles_b, tile_pos = [], [], []

            if idx == total or idx % args.progress_interval == 0:
                elapsed = max(time.time() - start, 1e-6)
                eta = (total - idx) / max(idx / elapsed, 1e-6)
                print(
                    f'[PROGRESS] {idx}/{total} tiles eta={eta / 60:.1f} min',
                    flush=True)

        avg = np.zeros_like(pred_sum, dtype=np.float32)
        valid = weights > 0
        avg[valid] = pred_sum[valid] / weights[valid]
        raw_mask = (avg[:h, :w] >= args.threshold).astype(np.uint8)
        save_mask_tif(raw_tif, raw_mask, ds1)
        print(f'[TIF] saved raw mask: {raw_tif}', flush=True)

        post_mask = postprocess_mask(
            raw_mask,
            remove_holes_area=args.remove_holes_area,
            remove_objects_min_size=args.remove_objects_min_size,
            fill_all_holes=args.fill_all_holes)
        save_mask_tif(post_tif, post_mask, ds1)
        print(f'[TIF] saved post mask: {post_tif}', flush=True)
        if args.save_shp:
            save_mask_shp(shp_path, post_mask, ds1.transform, ds1.crs)


def parse_args():
    parser = argparse.ArgumentParser('Open-CD big GeoTIFF inference')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tif-a', required=True, help='first temporal GeoTIFF')
    parser.add_argument('--tif-b', required=True, help='second temporal GeoTIFF')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--remove-holes-area', type=int, default=2000)
    parser.add_argument('--remove-objects-min-size', type=int, default=100)
    parser.add_argument('--fill-all-holes', action='store_true')
    parser.add_argument('--save-shp', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--progress-interval', type=int, default=20)
    return parser.parse_args()


if __name__ == '__main__':
    run_big_tif(parse_args())

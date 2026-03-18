#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations

import argparse
import glob
import os
import json
import math
from dataclasses import dataclass
from typing import List
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
import yaml

# =============================
# Constants
# =============================
GRID = 1216  # BEV resolution
NUM_CLASSES = 3
DOWN_RATIO = 4
CENTER_PEAK_THRESH = 0.4
MAX_Z_M = 5.5
DIM_SCALE = 10.0
DT = 0.1  # 10Hz 가정 (타임스탬프 없으므로 프레임*i*0.1s 사용)

COLORS = (
    (0, 255, 255), (0, 0, 255), (255, 0, 0), (255, 120, 0),
    (255, 120, 120), (0, 120, 0), (120, 255, 255), (120, 0, 255),
)

# =============================
# Boundary definition
# =============================
@dataclass(frozen=True)
class Boundary:
    minX: float = -25.0
    maxX: float = 75.0
    minY: float = -50.0
    maxY: float = 50.0
    minZ: float = -4.1
    maxZ: float = 1.4

    @property
    def discretization(self) -> float:
        """Return discretization scale in meters per grid cell."""
        return (self.maxX - self.minX) / GRID

    @property
    def z_range(self) -> float:
        return float(abs(self.maxZ - self.minZ))


BOUNDARY = Boundary()

# =============================
# PointCloud → BEV Map
# =============================

def remove_points(pcd: np.ndarray, boundary: Boundary = BOUNDARY) -> np.ndarray:
    """Remove points outside the boundary and normalize z."""
    mask = (
        (pcd[:, 0] >= boundary.minX) & (pcd[:, 0] <= boundary.maxX) &
        (pcd[:, 1] >= boundary.minY) & (pcd[:, 1] <= boundary.maxY) &
        (pcd[:, 2] >= boundary.minZ) & (pcd[:, 2] <= boundary.maxZ)
    )
    out = pcd[mask].copy()
    out[:, 2] -= boundary.minZ
    return out


def make_bev_map(pcd: np.ndarray, boundary: Boundary = BOUNDARY) -> np.ndarray:
    """Create 3-channel BEV map (height, intensity, density) in *strict compatibility*
    with the original `makeBEVMap` implementation.


    Notes on compatibility:
    - Uses grid size (GRID+1) during rasterization and crops to [:GRID, :GRID]
    - Applies uint8 quantization (×255 → uint8 → ÷255) for height/density/intensity,
    matching the original behavior exactly
    - Z mask: keep 0 ≤ z ≤ MAX_Z_M before rasterization
    - Channel order: [0]=intensity, [1]=height, [2]=density
    """
    # z-mask as in original code
    mask = (pcd[:, 2] >= 0.0) & (pcd[:, 2] <= MAX_Z_M)
    pts = pcd[mask].copy()


    # Meter→pixel projection with the same offsets as original
    H = W = GRID + 1
    d = boundary.discretization
    pts[:, 0] = np.floor(pts[:, 0] / d + H / 4).astype(np.int32)
    pts[:, 1] = np.floor(pts[:, 1] / d + W / 2).astype(np.int32)


    # Sort and unique by (x,y), keeping highest z first
    order = np.lexsort((-pts[:, 2], pts[:, 1], pts[:, 0]))
    pts = pts[order]
    xy = pts[:, :2].astype(np.int32)
    _, idx, counts = np.unique(xy, axis=0, return_index=True, return_counts=True)
    top = pts[idx]


    # Allocate maps with +1 then crop to GRID to mirror original
    hmap = np.zeros((H, W), dtype=np.float32)
    imap = np.zeros_like(hmap)
    dmap = np.zeros_like(hmap)


    rr, cc = top[:, 0].astype(int), top[:, 1].astype(int)


    # Height map (normalize by z-range, then uint8 quantization like original)
    zmax = boundary.z_range
    hvals = np.clip(top[:, 2] / zmax, 0.0, 1.0)
    hmap[rr, cc] = hvals


    # Intensity map (original cast to uint8 then /255.0)
    imap[rr, cc] = top[:, 3]


    # Density map (log-normalized counts, then quantized)
    dvals = np.minimum(1.0, np.log(counts + 1) / np.log(128))
    dmap[rr, cc] = dvals


    # --- uint8 quantization step to match the legacy output ---
    dmap = (dmap * 255).astype(np.uint8).astype(np.float32) / 255.0
    hmap = (hmap * 255).astype(np.uint8).astype(np.float32) / 255.0
    imap = (imap).astype(np.uint8).astype(np.float32) / 255.0


    bev = np.zeros((3, GRID, GRID), dtype=np.float32)
    bev[2] = dmap[:GRID, :GRID] # density → R
    bev[1] = hmap[:GRID, :GRID] # height → G
    bev[0] = imap[:GRID, :GRID] # intensity → B
    return bev

# =============================
# Post-processing utilities
# =============================

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)


def _nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    return feat.gather(1, ind)


def _transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


def _topk(scores: torch.Tensor, K: int = 40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.floor_divide(topk_ind, K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, _, _, _ = hm_cen.size()
    hm_cen = _sigmoid(hm_cen)
    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K)

    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds).view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs, ys = xs.view(batch_size, K, 1) + 0.5, ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds).view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds).view(batch_size, K, 1) * MAX_Z_M
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3) * DIM_SCALE
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # --- Debug: log first few decoded entries ---
    log_k = min(5, K)
    print("[decode] sample yaw entries:")
    for idx in range(log_k):
        print(f"  idx={idx} score={float(scores[0, idx, 0]):.3f} "
              f"x={float(xs[0, idx, 0]):.3f} y={float(ys[0, idx, 0]):.3f} "
              f"yaw_raw={float(torch.atan2(direction[0, idx, 0], direction[0, idx, 1])):.3f}")

    return torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)


def get_yaw(direction: np.ndarray) -> np.ndarray:
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def post_processing(dets: np.ndarray, peak_thresh=CENTER_PEAK_THRESH) -> List[np.ndarray]:
    """Filter and format detections.

    Returns:
        List of arrays per class: [score, x, y, z, h, w_px, l_px, yaw, cls]
    Notes:
        - Always returns NUM_CLASSES entries (empty arrays when no detections for a class)
        - x,y are scaled to pixel space by DOWN_RATIO (same as original)
        - width/length are converted from network scale to *pixel units* by (/100 * GRID)
          to exactly match the legacy implementation
    """
    results: List[np.ndarray] = [np.empty((0, 9), dtype=np.float32) for _ in range(NUM_CLASSES)]
    for cls in range(NUM_CLASSES):
        inds = (dets[:, -1] == cls)
        cls_dets = dets[inds]
        if len(cls_dets) == 0:
            continue
        cls_dets = cls_dets[cls_dets[:, 0] > peak_thresh]
        if len(cls_dets) == 0:
            continue
        cls_dets = cls_dets.copy()
        # x,y → pixel
        cls_dets[:, 1:3] *= DOWN_RATIO
        # w,l → pixel (match original: /100 * GRID)
        cls_dets[:, 5:6] = cls_dets[:, 5:6] / 100.0 * GRID
        cls_dets[:, 6:7] = cls_dets[:, 6:7] / 100.0 * GRID
        # yaw from direction vector
        yaw = get_yaw(cls_dets[:, 7:9])
        cls_dets = np.concatenate([cls_dets[:, :7], yaw, cls_dets[:, -1:]], axis=1)
        results[cls] = cls_dets
    return results

# =============================
# Debug helpers (추가)
# =============================

def _stats(t: torch.Tensor):
    t = t.detach().float()
    return dict(
        shape=tuple(t.shape),
        dtype=str(t.dtype).replace('torch.', ''),
        mean=float(t.mean()),
        std=float(t.std(unbiased=False)),
        min=float(t.min()),
        max=float(t.max()),
    )

@torch.no_grad()
def inspect_heads(hm_logits: torch.Tensor,
                  cen_offset_logits: torch.Tensor,
                  direction: torch.Tensor,
                  z_coor: torch.Tensor,
                  dim: torch.Tensor,
                  k: int = 10):
    """
    - hm_logits: (1,C,H,W) raw logits
    - cen_offset_logits: (1,2,H,W) raw
    - direction, z_coor, dim: (1,*,H,W)
    """
    assert hm_logits.dim() == 4 and hm_logits.size(0) == 1
    B, C, H, W = hm_logits.shape

    # print("\n[Head stats]")
    # print("  hm logits :", _stats(hm_logits))
    hm_sig = _sigmoid(hm_logits.clone())
    # print("  hm sigmoid:", _stats(hm_sig))
    hm_nms = _nms(hm_sig.clone())
    # print("  hm nms    :", _stats(hm_nms))

    # per-class count over threshold (after sigmoid, before NMS)
    th = CENTER_PEAK_THRESH
    counts = [(hm_sig[0, c] > th).sum().item() for c in range(C)]
    # print(f"  counts > {th} (per class):", counts)

    # ----- TopK BEFORE NMS (sigmoid) -----
    s1, inds1, cls1, ys1, xs1 = _topk(hm_sig, K=k)
    s1 = s1[0].cpu().numpy()
    cls1 = cls1[0].cpu().numpy()
    ys1 = ys1[0].cpu().numpy().astype(int)
    xs1 = xs1[0].cpu().numpy().astype(int)
    # print(f"\n[Top{k} @ sigmoid] (score, c, y, x)")
    # for i in range(k):
    #     print(f"  {i:02d}: {s1[i]:.6f}  c={cls1[i]}  y={ys1[i]}  x={xs1[i]}")

    # ----- TopK AFTER NMS -----
    s2, inds2, cls2, ys2, xs2 = _topk(hm_nms, K=k)
    # print(f"\n[Top{k} @ NMS] (score, c, y, x)")
    # for i in range(k):
    #     print(f"  {i:02d}: {float(s2[0,i]):.6f}  c={int(cls2[0,i])}  y={int(ys2[0,i])}  x={int(xs2[0,i])}")

    # ----- For TopK@NMS: gather other heads and compute decoded quantities -----
    # offset -> center subpixel, z, dim, yaw
    off = _transpose_and_gather_feat(_sigmoid(cen_offset_logits.clone()), inds2).view(B, k, 2)  # (1,k,2)
    dirv = _transpose_and_gather_feat(direction.clone(), inds2).view(B, k, 2)                   # (1,k,2)
    zz   = _transpose_and_gather_feat(z_coor.clone(), inds2).view(B, k, 1) * MAX_Z_M           # (1,k,1)
    dd   = _transpose_and_gather_feat(dim.clone(), inds2).view(B, k, 3) * DIM_SCALE            # (1,k,3)
    xs = xs2.view(B, k, 1) + off[:, :, 0:1]
    ys = ys2.view(B, k, 1) + off[:, :, 1:2]

    yaw = torch.atan2(dirv[:, :, 0:1], dirv[:, :, 1:2])  # atan2(x, y) per original
    # pixel conversions (post_processing와 동일)
    x_px = xs * DOWN_RATIO
    y_px = ys * DOWN_RATIO
    h_m  = dd[:, :, 0:1]
    w_m  = dd[:, :, 1:2]
    l_m  = dd[:, :, 2:1+2]
    w_px = (w_m / 100.0) * GRID
    l_px = (l_m / 100.0) * GRID

    # print(f"\n[Decoded @ Top{k} after NMS]")
    # print("  idx | score   c  y  x  | off(x,y)     | xy(px)         | z(m)   | h,w,l(m)        | w,l(px)         | yaw(rad)")
    # for i in range(k):
    #     print(f"  {i:02d} | {float(s2[0,i]):.6f}  {int(cls2[0,i])} {int(ys2[0,i])} {int(xs2[0,i])} | "
    #           f"{float(off[0,i,0]):+.3f},{float(off[0,i,1]):+.3f} | "
    #           f"{float(x_px[0,i,0]):7.2f},{float(y_px[0,i,0]):7.2f} | "
    #           f"{float(zz[0,i,0]):5.2f} | "
    #           f"{float(h_m[0,i,0]):5.2f},{float(w_m[0,i,0]):5.2f},{float(l_m[0,i,0]):5.2f} | "
    #           f"{float(w_px[0,i,0]):7.2f},{float(l_px[0,i,0]):7.2f} | "
    #           f"{float(yaw[0,i,0]):+6.3f}"
    #     )
        
# =============================
# Visualization
# =============================

def get_corners(x, y, w, l, yaw):
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return np.array([
        [x - w/2 * cos_y - l/2 * sin_y, y - w/2 * sin_y + l/2 * cos_y],
        [x - w/2 * cos_y + l/2 * sin_y, y - w/2 * sin_y - l/2 * cos_y],
        [x + w/2 * cos_y + l/2 * sin_y, y + w/2 * sin_y - l/2 * cos_y],
        [x + w/2 * cos_y - l/2 * sin_y, y + w/2 * sin_y + l/2 * cos_y],
    ])


def draw_rotated_box(img, x, y, w, l, yaw, color):
    corners = get_corners(x, y, w, l, yaw).astype(int)
    cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, color, 2)
    cv2.line(img, tuple(corners[0]), tuple(corners[3]), (255, 255, 0), 2)


def draw_predictions(img: np.ndarray, detections: np.ndarray) -> np.ndarray:
    for det in detections:
        _, x, y, _, h, w, l, yaw, cls = det
        draw_rotated_box(img, x, y, w, l, yaw, COLORS[int(cls)])
    return img


# model_path = "sfa_sim.onnx"
# scene_paths = sorted(glob.glob("at128/*.bin"))

# # Load ONNX model
# session = ort.InferenceSession(model_path)
# input_name = session.get_inputs()[0].name

# # Load LiDAR frame
# lidar_data = np.fromfile(scene_paths[100], dtype=np.float32).reshape(-1, 4)
# lidar_data = remove_points(lidar_data, BOUNDARY)
# bev = make_bev_map(lidar_data, BOUNDARY)
# bev_input = np.expand_dims(bev, axis=0).astype(np.float32)

# # np.save("bev_py.npy", bev_input)  # shape (1,3,1216,1216)
# # print("[PY] bev stats per-ch:", bev_input.min(axis=(0,2,3)), bev_input.max(axis=(0,2,3)))

# # Run inference
# output_names = [o.name for o in session.get_outputs()]
# # print(output_names)
# outputs = session.run(output_names, {input_name: bev_input})
# # print(outputs[0].shape)
# # print(np.mean(outputs[0]))
# # print(outputs[1].shape)
# # print(outputs[2].shape)
# # print(outputs[3].shape)
# # print(outputs[4].shape)

# hm_cen, cen_offset, direction, z_coor, dim = map(torch.tensor, outputs)
# # hm_cen, cen_offset = _sigmoid(hm_cen), _sigmoid(cen_offset)

# # ===== Debug: step stats + TopK (sigmoid & NMS) + gathered heads =====
# # inspect_heads(hm_cen, cen_offset, direction, z_coor, dim, k=10)

# # print("HM_CEN: ", hm_cen[:1])
# # print("CEN_OFFSET: ", cen_offset[:1])
# detections = decode(hm_cen, cen_offset, direction, z_coor, dim, K=50)
# detections = detections.cpu().numpy().astype(np.float32)
# detections = np.concatenate(detections, axis=0)
# detections = post_processing(detections, CENTER_PEAK_THRESH)

# # for result in detections:
# #     print(result[1],result[2])
# # Visualize
# bev_img = np.transpose(bev, (1, 2, 0))
# bev_img = cv2.resize(bev_img, (GRID, GRID))
# for cls_dets in detections:
#     bev_img = draw_predictions(bev_img, cls_dets)
# bev_img = cv2.rotate(bev_img, cv2.ROTATE_180)

# # plt.figure(figsize=(15,15))
# # plt.imshow(bev_img)
# # plt.title("BEV with Detections")
# # plt.show()



# =============================
# Tracking 통합
# =============================

from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData

#  BEV raster와 호환되는 px↔m 매퍼
#   make_bev_map에서 x_px = floor(x_m/d + H/4), y_px = floor(y_m/d + W/2)
#   => 역변환: x_m ≈ (v + 0.5 - H/4) * d,  y_m ≈ (u + 0.5 - W/2) * d
# ★★ px <-> ego(m) 매퍼 (이미지 y↓ ↔ ego y← 부호 보정 포함) ★★
d = BOUNDARY.discretization  # meters per pixel

# meters per pixel
d = BOUNDARY.discretization

def px_to_ego(v_px: float, u_px: float):
    """img (row=v, col=u) px -> ego(m)"""
    x_e = BOUNDARY.minX + (v_px + 0.5) * d
    y_img = BOUNDARY.minY + (u_px + 0.5) * d
    y_e = -y_img                          # ★ y 부호 반전
    return x_e, y_e

def ego_to_px(x_e: float, y_e: float):
    """ego(m) -> img (row=v, col=u) px"""
    v = (x_e - BOUNDARY.minX) / d - 0.5
    u = ((-y_e) - BOUNDARY.minY) / d - 0.5 # ★ y 부호 반전
    return int(round(v)), int(round(u))


# 디텍션 → FrameData 포장
# CLASS_TOKEN = {0:1, 1:2, 2:4}  # vehicle/pedestrian/cyclist

# 0: Ped(=2), 1: Car(=1), 2: Cyclist(=4)
CLASS_TOKEN = {0: 2, 1: 1, 2: 4}

# Detections Return value
# List of arrays per class: [score, x, y, z, h, w_px, l_px, yaw, cls]

DEBUG_DET = False

def detections_to_framedata(detections, timestamp, ego=None, cls_name=None):
    if ego is None:
        ego = np.eye(4, dtype=np.float32)
    det_arrays, det_types = [], []

    for cls_dets in detections:
        if len(cls_dets) == 0:
            continue
        for (score, x_px, y_px, z_m, h_m, w_px, l_px, yaw_px, cls_idx) in cls_dets:
            # 1) 위치: px -> ego(m)
            x_e, y_e = px_to_ego(y_px, x_px)     # v=y_px, u=x_px
            # 2) 크기: px -> m
            w_m = float(w_px) * d
            l_m = float(l_px) * d
            # 3) 각도: network yaw already lives in ego frame → just wrap to [-pi, pi]
            o_e = float(((yaw_px + np.pi) % (2 * np.pi)) - np.pi)

            # ★★★ 최종 배열 순서: [x, y, z, o, l, w, h, s] ★★★
            det_arrays.append(np.array(
                [x_e, y_e, float(z_m), o_e, l_m, w_m, float(h_m), float(score)],
                dtype=np.float32
            ))
            det_types.append(CLASS_TOKEN.get(int(cls_idx)))
            if DEBUG_DET:
                print(f"[detections_to_framedata] cls={cls_idx} yaw_px={yaw_px:.3f} -> o_e={o_e:.3f}")

    aux_info = {'is_key_frame': True}
    if cls_name is not None:
        aux_info['cls_name'] = cls_name

    return FrameData(
        dets=det_arrays,
        ego=ego,
        time_stamp=float(timestamp),
        pc=None,
        det_types=det_types,
        aux_info=aux_info
    )


# 트레일 폴리라인 → 폴리곤(채움용)
def stroke_polyline_to_polygon(pts, width=8):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2: return None
    half = width / 2.0
    left, right = [], []
    for i in range(len(pts)-1):
        p0, p1 = pts[i], pts[i+1]
        v = p1 - p0
        L = np.linalg.norm(v)
        if L < 1e-6: continue
        n = np.array([-v[1], v[0]], dtype=np.float32) / L * half
        left.append(p0 + n); right.append(p0 - n)
        if i == len(pts)-2:
            left.append(p1 + n); right.append(p1 - n)
    if not left or not right: return None
    poly = np.vstack([np.array(left), np.array(right[::-1])]).astype(np.int32)
    return poly.reshape(-1,1,2)

import copy


def make_cfg(src,
             giou_asso,
             giou_redund,
             max_age,
             min_hits,
             score_thr=0.45,
             nms_th=0.5,
             measurement_noise=None,
             post_nms_iou=None):
    cfg = copy.deepcopy(src)
    # 공통
    cfg['running']['asso'] = 'giou'
    cfg['running'].setdefault('asso_thres', {})
    cfg['running']['asso_thres']['giou'] = giou_asso          # Association threshold (1 - GIoU) 공간, ↑ 더 널널
    cfg.setdefault('redundancy', {}).setdefault('det_dist_threshold', {})
    cfg['redundancy']['det_dist_threshold']['giou'] = giou_redund  # Redundancy threshold (GIoU) 공간, ↓ 더 널널
    cfg['running']['max_age_since_update'] = max_age
    cfg['running']['min_hits_to_birth']   = min_hits
    cfg['running']['score_threshold']     = score_thr
    cfg.setdefault('data_loader', {}).setdefault('nms_thres', nms_th)
    cfg['data_loader']['nms_thres'] = nms_th

    if measurement_noise is not None:
        cfg['running']['measurement_noise'] = list(measurement_noise)
    else:
        cfg['running'].pop('measurement_noise', None)

    if post_nms_iou is not None:
        cfg['running']['post_nms_iou'] = float(post_nms_iou)
    else:
        cfg['running'].pop('post_nms_iou', None)

    return cfg

# =============================
# CLI
# =============================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SFA3D ONNX inference with SimpleTrack MOT.")
    parser.add_argument("--model-path", type=str, default="/home/a/Downloads/sfa_sim.onnx",
                        help="ONNX 모델 경로")
    parser.add_argument("--data-glob", type=str, default="../at128/*.bin",
                        help="입력 LiDAR .bin 파일 glob 패턴")
    parser.add_argument("--config-path", type=str, default="../configs/waymo_configs/vc_kf_giou.yaml",
                        help="SimpleTrack 설정 YAML 경로")
    parser.add_argument("--video-path", type=str, default=None,
                        help="지정 시 해당 경로로 영상(mp4/avi) 저장")
    parser.add_argument("--video-fps", type=float, default=1.0 / DT,
                        help="영상 저장 시 FPS (기본 10Hz)")
    parser.add_argument("--play", action="store_true",
                        help="키 대기 없이 자동으로 재생")
    parser.add_argument("--no-gui", action="store_true",
                        help="OpenCV 창 없이 저장/처리만 수행")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="테스트용으로 처리할 최대 프레임 수 제한")
    parser.add_argument("--debug-assoc", action="store_true",
                        help="연관(association) 단계 상세 출력")
    parser.add_argument("--debug-life", action="store_true",
                        help="트랙 생성/수명 디버그 출력")
    parser.add_argument("--debug-all", action="store_true",
                        help="association/lifecycle 디버그 모두 활성화")
    parser.add_argument("--debug-det", action="store_true",
                        help="detections_to_framedata 단계 yaw 변환 로그")
    return parser


# =============================
# Main
# =============================
def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    global DEBUG_DET
    DEBUG_DET = args.debug_det

    scene_paths = sorted(glob.glob(args.data_glob))
    if not scene_paths:
        raise FileNotFoundError(f"No .bin files found for pattern: {args.data_glob}")
    if args.max_frames is not None:
        scene_paths = scene_paths[:max(0, args.max_frames)]
    print(f"[DATA] Loaded {len(scene_paths)} frames from pattern '{args.data_glob}'")

    # 강제 재생 (영상 기록 시 대기 비활성화)
    pause_for_key = not args.play
    if args.video_path:
        pause_for_key = False

    # 1) ONNX 로드
    session = ort.InferenceSession(args.model_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print("[ONNX] inputs:", input_name, " outputs:", output_names)

    # 2) 트래커 로드 (설정 약간 관대하게 튠)
    with open(args.config_path, "r") as f:
        base_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Params (수동 튜닝)
    veh_meas_noise = [2.0, 2.0, 0.4, 0.08, 1.0, 1.0, 1.0]
    ped_meas_noise = [0.25, 0.25, 0.5, 0.08, 0.6, 0.6, 0.4]
    cyc_meas_noise = [0.22, 0.22, 0.45, 0.06, 0.55, 0.55, 0.35]

    cfg_veh = make_cfg(base_cfg, giou_asso=1.5, giou_redund=0.45, max_age=3, min_hits=3,
                       score_thr=0.45, nms_th=0.35, measurement_noise=veh_meas_noise, post_nms_iou=0.5)
    cfg_cyc = make_cfg(base_cfg, giou_asso=1.2, giou_redund=0.6, max_age=5, min_hits=3,
                       score_thr=0.40, nms_th=0.5, measurement_noise=cyc_meas_noise, post_nms_iou=0.5)
    cfg_ped = make_cfg(base_cfg, giou_asso=1.3, giou_redund=0.2, max_age=6, min_hits=3,
                       score_thr=0.35, nms_th=0.5, measurement_noise=ped_meas_noise, post_nms_iou=0.5)

    tracker_veh = MOTModel(cfg_veh)
    tracker_cyc = MOTModel(cfg_cyc)
    tracker_ped = MOTModel(cfg_ped)

    assoc_debug = args.debug_all or args.debug_assoc
    life_debug = args.debug_all or args.debug_life
    for tr in (tracker_veh, tracker_cyc, tracker_ped):
        tr.set_debug(association=assoc_debug, lifecycle=life_debug)
    # tracker_veh.set_debug(association=True,  lifecycle=True)   # 차량만 상세 로그 ON
    # tracker_ped.set_debug(association=False, lifecycle=False)  # 보행자 로그 OFF
    # tracker_cyc.set_debug(association=False, lifecycle=False)  # 자전거 로그 OFF
    print("[Tracker] initialized with class-wise configs")
    print("[veh]", cfg_veh['running']['asso_thres']['giou'], cfg_veh['redundancy']['det_dist_threshold']['giou'], cfg_veh['running']['max_age_since_update'])
    print("      measurement_noise:", veh_meas_noise)
    print("[cyc]", cfg_cyc['running']['asso_thres']['giou'], cfg_cyc['redundancy']['det_dist_threshold']['giou'], cfg_cyc['running']['max_age_since_update'])
    print("      measurement_noise:", cyc_meas_noise)
    print("[ped]", cfg_ped['running']['asso_thres']['giou'], cfg_ped['redundancy']['det_dist_threshold']['giou'], cfg_ped['running']['max_age_since_update'])
    print("      measurement_noise:", ped_meas_noise)
    if assoc_debug or life_debug:
        print(f"[Debug] association={assoc_debug} lifecycle={life_debug}")
    
    # 3) 뷰어
    show_gui = not args.no_gui
    if show_gui:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 900, 900)

    # 4) VideoWriter 준비 (선택)
    writer = None
    if args.video_path:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if args.video_path.lower().endswith('.mp4') else 'XVID'))
        writer = cv2.VideoWriter(args.video_path, fourcc, args.video_fps, (GRID, GRID))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {args.video_path}")
        print(f"[Video] Recording to {args.video_path} @ {args.video_fps:.2f} FPS")

    frame_period_ms = max(1, int(round(1000.0 / args.video_fps)))

    trail_px = defaultdict(lambda: deque(maxlen=120))  # ID -> 최근 px (u,v)
    last_m = {}
    total_dist_m = defaultdict(float)


    WIN_NAME = "Tracking"

    if show_gui:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 900, 900)
# ------------ Main Logic -------------------

    for i, bin_path in enumerate(scene_paths[:-1]):
        # --- LiDAR → BEV ---
        lidar = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        lidar = remove_points(lidar, BOUNDARY)
        bev = make_bev_map(lidar, BOUNDARY)
        bev_input = np.expand_dims(bev, axis=0).astype(np.float32)

        # --- Inference ---
        outputs = session.run(output_names, {input_name: bev_input})
        hm_cen, cen_offset, direction, z_coor, dim = map(torch.tensor, outputs)

        # --- Decode + Post-process ---
        dets = decode(hm_cen, cen_offset, direction, z_coor, dim, K=60)
        dets = dets.cpu().numpy().astype(np.float32)
        dets = np.concatenate(dets, axis=0)
        detections = post_processing(dets, CENTER_PEAK_THRESH)
        # print(f"cls detections len {len(detections[1])}")
        # print(f"ped detections len {len(detections[0])}")
        # print(f"cyc detections len {len(detections[2])}")

        
        # print(f"[DET] ped={detections[0]}, car={detections[1].shape[0]}, cyc={detections[2].shape[0]}")
        # print(detections)
        ts = i * DT
        # fd = detections_to_framedata(detections, ts, ego=np.eye(4, dtype=np.float32))
        # tracks = tracker.frame_mot(fd)

        I4 = np.eye(4, dtype=np.float32)

        # detections 
        # 0.0 : Ped 
        # 1.0 : Car 
        # 2.0 : Cyclist

        ped_list = [detections[0]] if len(detections) > 0 else []
        veh_list = [detections[1]] if len(detections) > 1 else []
        cyc_list = [detections[2]] if len(detections) > 2 else []

        # 클래스별 FrameData 생성
        fd_ped = detections_to_framedata(ped_list, ts, ego=I4, cls_name="Pedestrian")
        fd_veh = detections_to_framedata(veh_list, ts, ego=I4, cls_name="Car")
        fd_cyc = detections_to_framedata(cyc_list, ts, ego=I4, cls_name="Cyclist")

        # 클래스별 추적
        res_veh = tracker_veh.frame_mot(fd_veh)
        res_ped = tracker_ped.frame_mot(fd_ped)
        res_cyc = tracker_cyc.frame_mot(fd_cyc)

        # 최종 병합
        tracks = res_veh + res_ped + res_cyc
        # tracks = res_veh

        # --- 캔버스 준비 (BGR)
        canvas = np.transpose(bev, (1, 2, 0)).copy()
        canvas = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(canvas, (GRID, GRID), interpolation=cv2.INTER_NEAREST)
        print("-----------------------")

        # ===  디텍션 박스(흰색) 먼저 그리기 ===
        # WHITE = (255, 255, 255)
        # for cls_dets in detections:

        #     for det in cls_dets:
                
        #         _, x_px, y_px, _, h_m, w_px, l_px, yaw, cls_idx = det
        #         # det 좌표계는 이미 BEV 픽셀 기준 (u=x, v=y)
        #         if(cls_idx == 1):

        #             draw_rotated_box(
        #                 canvas,
        #                 int(x_px), int(y_px),
        #                 float(w_px), float(l_px),
        #                 float(yaw),
        #                 WHITE
        #             )
        # --- 트랙 박스 (+ 선택적으로 trail) ---
        for (bbox, tid, state_str, det_type) in tracks:
            v, u = ego_to_px(float(bbox.x), float(bbox.y))
            
            # 경계 체크: 화면 밖이면 스킵 (표시만 생략)
            if not (0 <= u < GRID and 0 <= v < GRID):
                continue

            # u = max(0, min(GRID - 1, u))
            # v = max(0, min(GRID - 1, v))
            w_px = float(bbox.w) / d
            l_px = float(bbox.l) / d
            yaw_draw = -float(getattr(bbox, 'o', 0.0))

            color = COLORS[tid % len(COLORS)]
            draw_rotated_box(canvas, u, v, w_px, l_px, -yaw_draw, color)

        canvas_show = cv2.rotate(canvas, cv2.ROTATE_180)

        if writer is not None:
            writer.write(canvas_show)

        if show_gui:
            # (선택) 윈도우 타이틀에 프레임 번호만 업데이트
            try:
                cv2.setWindowTitle(WIN_NAME, f"Tracking {i+1}")  # OpenCV 4.5+에서 동작
            except Exception:
                pass

            cv2.imshow(WIN_NAME, canvas_show)                 # ← 항상 같은 창 이름으로 갱신
            wait_ms = 0 if pause_for_key else frame_period_ms
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), 27):
                print("[INFO] Early termination requested.")
                break
            elif key in (ord('s'),) and pause_for_key:
                os.makedirs("results_custom/frames", exist_ok=True)
                cv2.imwrite(f"results_custom/frames/frame_{i:04d}.png", canvas_show)

        elif not args.video_path:
            # GUI 없이 실행하는 경우 진행률만 보여주기
            if i % 10 == 0 or i == len(scene_paths) - 1:
                print(f"[Progress] frame {i+1}/{len(scene_paths)}")

    if writer is not None:
        writer.release()
        print(f"[DONE] Video saved -> {args.video_path}")

    if show_gui:
        cv2.destroyAllWindows()
    print("[DONE] Tracking run")


if __name__ == "__main__":
    main()

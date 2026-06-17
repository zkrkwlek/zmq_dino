"""
pipeline_debug_visualizer.py  -- Matplotlib OO API (thread-safe, no pyplot)
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from anchor_debug_visualizer import (make_distinct_colors, patch_to_pixel, draw_patch_mask)
except ImportError:
    from dino.anchor_debug_visualizer import (make_distinct_colors, patch_to_pixel, draw_patch_mask)

# ── 내부 유틸 ─────────────────────────────────────────────────────

def _norm_color(c):
    """(R,G,B) 0-255 -> 0-1"""
    return tuple(v/255 for v in c)

def _save_fig(fig: Figure, save_path: str, dpi: int = 100):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(save_path, dpi=dpi)
        fig.clf()
        print(f"[viz] saved: {save_path}")

def _img_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float [0,1]"""
    return img_bgr[:, :, ::-1].astype(np.float32) / 255.0

def _from_vom(vom_row: np.ndarray, grid_shape, patch_size: int):
    """vom bool row -> pixel centroid (cx, cy)"""
    H_p, W_p = grid_shape
    idx = np.where(vom_row)[0]
    if len(idx) == 0:
        return W_p * patch_size // 2, H_p * patch_size // 2
    cy = int((( idx // W_p).mean() + 0.5) * patch_size)
    cx = int(((idx  % W_p).mean() + 0.5) * patch_size)
    return cx, cy

def _patch_overlay_ax(ax, img_rgb, patches_idx, grid_shape, patch_size, color_rgb, alpha=0.45):
    """img_rgb 위에 패치 마스크 오버레이 후 ax에 imshow"""
    overlay = img_rgb.copy()
    H_p, W_p = grid_shape
    H, W = img_rgb.shape[:2]
    mask = np.zeros((H, W, 3), dtype=np.float32)
    for n in patches_idx:
        r, c = n // W_p, n % W_p
        y0, x0 = r * patch_size, c * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        mask[y0:y1, x0:x1] = color_rgb
    blended = overlay * (1 - alpha) + mask * alpha
    ax.imshow(np.clip(blended, 0, 1))

# ── Phase 1 ──────────────────────────────────────────────────────

def phase1_attention(img_bgr, attn, grid_shape, patch_size=14, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    img_rgb = _img_to_rgb(img_bgr)
    an = attn.cpu().float().numpy().reshape(H_p, W_p)
    an = (an - an.min()) / (an.max() - an.min() + 1e-8)
    import cv2
    heat_bgr  = cv2.applyColorMap((cv2.resize(an, (W, H)) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat_rgb  = heat_bgr[:, :, ::-1].astype(np.float32) / 255.0
    blended   = img_rgb * 0.45 + heat_rgb * 0.55
    grid_big  = cv2.resize(cv2.applyColorMap((an * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                           (W, H), interpolation=cv2.INTER_NEAREST)
    grid_rgb  = grid_big[:, :, ::-1].astype(np.float32) / 255.0

    fig = Figure(figsize=(W * 3 / 100, H / 100))
    axes = fig.subplots(1, 3)
    for ax, im, title in zip(axes,
                             [img_rgb, blended, grid_rgb],
                             ["Original", "CLS Attention", f"Grid [{H_p}x{W_p}]"]):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.suptitle("PHASE 1 -- CLS Attention", fontsize=8, color="gold")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase1_xfeat_patches(img_bgr, kp, bind_mat, grid_shape, patch_size=14, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    img_rgb  = _img_to_rgb(img_bgr)
    kp_np    = kp.cpu().numpy()
    bind_np  = bind_mat.cpu().numpy()
    kpp      = bind_np.sum(axis=1)
    mx       = max(kpp.max(), 1)

    canvas_overlay = img_rgb.copy()
    for n in range(H_p * W_p):
        r, c = n // W_p, n % W_p
        y0, x0 = r * patch_size, c * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        t = kpp[n] / mx
        canvas_overlay[y0:y1, x0:x1] = (
            canvas_overlay[y0:y1, x0:x1] * 0.5 +
            np.array([0.0, t * 0.7, t * 0.3]) * 0.5
        )

    fig = Figure(figsize=(W * 2 / 100, H / 100))
    ax0, ax1 = fig.subplots(1, 2)
    ax0.imshow(np.clip(canvas_overlay, 0, 1))
    ax0.scatter(kp_np[:, 0], kp_np[:, 1], s=4, c="cyan", alpha=0.7, linewidths=0)
    ax0.set_title(f"XFeat kp={len(kp_np)}", fontsize=7)
    ax0.axis("off")

    step = max(1, len(kpp) // 60)
    colors = [(0, v / mx * 0.7, v / mx * 0.3) for v in kpp[::step]]
    ax1.bar(range(len(kpp[::step])), kpp[::step], color=colors, width=1.0)
    ax1.set_title(f"kp/patch  N={H_p*W_p}  M={len(kp_np)}", fontsize=7)
    ax1.tick_params(labelsize=5)

    fig.suptitle("PHASE 1 -- XFeat Binding", fontsize=8, color="gold")
    fig.tight_layout()
    _save_fig(fig, save_path)

# ── Phase 2 ──────────────────────────────────────────────────────

def phase2_memory_pool(pool_vecs, source_labels=None, selected_mask=None, save_path=None):
    import torch.nn.functional as F
    M = pool_vecs.shape[0]
    if M == 0:
        return
    vecs_np  = F.normalize(pool_vecs, p=2, dim=1).cpu().numpy()
    sel_np   = selected_mask.cpu().numpy() if selected_mask is not None else None
    n_src    = len(set(source_labels)) if source_labels else 1
    palette  = make_distinct_colors(max(n_src, 1))
    pt_cols  = [
        _norm_color(palette[source_labels[i] % len(palette)]) if source_labels else (0.7, 0.7, 0.7)
        for i in range(M)
    ]

    fig = Figure(figsize=(8, 4))
    ax0, ax1 = fig.subplots(1, 2)
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(vecs_np) if M >= 3 else np.zeros((M, 2))
        sel_idx = np.where(sel_np)[0] if sel_np is not None else []
        ax0.scatter(coords[:, 0], coords[:, 1], c=pt_cols, s=14, alpha=0.8, linewidths=0)
        if len(sel_idx):
            ax0.scatter(coords[sel_idx, 0], coords[sel_idx, 1],
                        s=60, facecolors="none", edgecolors="white", linewidths=1.5)
        ax0.set_title("Pool PCA (* = selected)", fontsize=7)
        ax0.tick_params(labelsize=5)
    except ImportError:
        ax0.text(0.5, 0.5, "sklearn unavailable", ha="center", va="center", transform=ax0.transAxes)

    info_text = (
        f"Pool size : {M}\n"
        f"Selected  : {int(sel_np.sum()) if sel_np is not None else M}\n"
        f"Sources   : {n_src}\n"
        f"Dim       : {pool_vecs.shape[1]}"
    )
    ax1.text(0.1, 0.6, info_text, transform=ax1.transAxes, fontsize=8,
             va="top", family="monospace", color="lightcyan")
    ax1.axis("off")

    fig.suptitle("PHASE 2 -- Memory Pool PCA", fontsize=8, color="gold")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase2_neighbor_frames(imgs_bgr, sims, selected_idx=None, save_path=None):
    if not imgs_bgr:
        return
    n = len(imgs_bgr)
    fig = Figure(figsize=(n * 3, 3))
    axes = fig.subplots(1, n)
    if n == 1:
        axes = [axes]
    for i, (im, sim, ax) in enumerate(zip(imgs_bgr, sims, axes)):
        is_sel = selected_idx is not None and i in selected_idx
        ax.imshow(_img_to_rgb(im))
        title = f"{'[SEL] ' if is_sel else ''}sim={sim:.3f}"
        ax.set_title(title, fontsize=7, color="lime" if is_sel else "white")
        if is_sel:
            for sp in ax.spines.values():
                sp.set_edgecolor("lime"); sp.set_linewidth(2.5)
        ax.axis("off")
    fig.suptitle("PHASE 2 -- Neighbor Frames", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)

# ── Phase 3 ──────────────────────────────────────────────────────

def phase3_anchors(img_bgr, sample, vom, pure, grid_shape, patch_size=14,
                   title_suffix="", save_path=None):
    K = vom.shape[0]
    vom_np  = vom.cpu().bool().numpy()
    pure_np = pure.cpu().bool().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _img_to_rgb(img_bgr)

    fig = Figure(figsize=(img_bgr.shape[1] * 2 / 100, img_bgr.shape[0] / 100))
    ax0, ax1 = fig.subplots(1, 2)

    ov_vom  = img_rgb.copy()
    ov_pure = img_rgb.copy()
    for k in range(K):
        col = _norm_color(palette[k])
        ptv = np.where(vom_np[k])[0]
        ptp = np.where(pure_np[k])[0]
        H_p, W_p = grid_shape
        H, W = img_bgr.shape[:2]
        for pts, ov, a in [(ptv, ov_vom, 0.40), (ptp, ov_pure, 0.50)]:
            mask = np.zeros_like(ov)
            for n in pts:
                r, c = n // W_p, n % W_p
                y0, x0 = r * patch_size, c * patch_size
                y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
                mask[y0:y1, x0:x1] = col
            ov[:] = ov * (1 - a) + mask * a

    ax0.imshow(np.clip(ov_vom, 0, 1))
    ax1.imshow(np.clip(ov_pure, 0, 1))
    for k in range(K):
        cx, cy = _from_vom(vom_np[k], grid_shape, patch_size)
        col = _norm_color(palette[k])
        for ax in (ax0, ax1):
            ax.plot(cx, cy, "o", ms=5, color=col)
            ax.text(cx + 4, cy - 4, str(k), fontsize=5, color="white")

    ax0.set_title(f"VOM  K={K}", fontsize=7)
    ax1.set_title("Pure (oc==1)", fontsize=7)
    for ax in (ax0, ax1):
        ax.axis("off")

    suf = f"  {title_suffix}" if title_suffix else ""
    fig.suptitle(f"PHASE 3 -- Anchors{suf}", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase3_cross_frame(img_bgr, vom, pure, valid_mask, grid_shape, patch_size=14, save_path=None):
    K = vom.shape[0]
    vom_np   = vom.cpu().bool().numpy()
    valid_np = valid_mask.cpu().bool().numpy()
    palette  = make_distinct_colors(max(K, 1))
    img_rgb  = _img_to_rgb(img_bgr)
    H, W     = img_bgr.shape[:2]
    H_p, W_p = grid_shape

    fig = Figure(figsize=(W * 2 / 100, H / 100))
    ax0, ax1 = fig.subplots(1, 2)

    ov_ok  = img_rgb.copy()
    ov_bad = img_rgb.copy()
    for k in range(K):
        pts = np.where(vom_np[k])[0]
        col = _norm_color(palette[k]) if valid_np[k] else (0.8, 0.2, 0.2)
        a   = 0.40
        ov  = ov_ok if valid_np[k] else ov_bad
        mask = np.zeros_like(ov)
        for n in pts:
            r, c = n // W_p, n % W_p
            y0, x0 = r * patch_size, c * patch_size
            y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
            mask[y0:y1, x0:x1] = col
        ov[:] = ov * (1 - a) + mask * a

    ax0.imshow(np.clip(ov_ok, 0, 1))
    ax1.imshow(np.clip(ov_bad, 0, 1))
    for k in range(K):
        cx, cy = _from_vom(vom_np[k], grid_shape, patch_size)
        col = _norm_color(palette[k]) if valid_np[k] else (0.9, 0.3, 0.3)
        ax = ax0 if valid_np[k] else ax1
        ax.plot(cx, cy, "o", ms=5, color=col)
        ax.text(cx + 4, cy - 4, str(k), fontsize=5, color="white")

    ax0.set_title(f"Valid {int(valid_np.sum())}/{K}", fontsize=7)
    ax1.set_title(f"Invalid {int((~valid_np).sum())}/{K}", fontsize=7)
    for ax in (ax0, ax1):
        ax.axis("off")

    fig.suptitle("PHASE 3 -- Cross-Frame Projection", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)

# ── Phase 4 ──────────────────────────────────────────────────────

def phase4_quality_filter(img_bgr, vom, pure, keep, grid_shape, patch_size=14,
                           quality_metrics=None, save_path=None):
    K = vom.shape[0]
    vom_np  = vom.cpu().bool().numpy()
    keep_np = keep.cpu().bool().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _img_to_rgb(img_bgr)
    H, W    = img_bgr.shape[:2]
    H_p, W_p = grid_shape

    fig = Figure(figsize=(W * 3 / 100, H / 100))
    ax0, ax1, ax2 = fig.subplots(1, 3)

    ov_k = img_rgb.copy()
    ov_r = img_rgb.copy()
    for k in range(K):
        pts = np.where(vom_np[k])[0]
        col = _norm_color(palette[k]) if keep_np[k] else (0.85, 0.2, 0.2)
        a   = 0.40
        ov  = ov_k if keep_np[k] else ov_r
        mask = np.zeros_like(ov)
        for n in pts:
            r, c = n // W_p, n % W_p
            y0, x0 = r * patch_size, c * patch_size
            y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
            mask[y0:y1, x0:x1] = col
        ov[:] = ov * (1 - a) + mask * a

    ax0.imshow(np.clip(ov_k, 0, 1))
    ax1.imshow(np.clip(ov_r, 0, 1))
    for k in range(K):
        cx, cy = _from_vom(vom_np[k], grid_shape, patch_size)
        col = _norm_color(palette[k]) if keep_np[k] else (0.9, 0.3, 0.3)
        ax  = ax0 if keep_np[k] else ax1
        ax.plot(cx, cy, "o", ms=5, color=col)
        ax.text(cx + 4, cy - 4, str(k), fontsize=5, color="white")
    ax0.set_title(f"유지 앵커 {int(keep_np.sum())}/{K}개", fontsize=7)
    ax1.set_title(f"제거 앵커 {int((~keep_np).sum())}/{K}개", fontsize=7)
    for ax in (ax0, ax1):
        ax.axis("off")

    if quality_metrics:
        key  = next(iter(quality_metrics))
        vals = np.asarray(quality_metrics[key], dtype=float)
    else:
        vals = pure.sum(dim=1).cpu().numpy().astype(float)
        key  = "pure count"
    bar_cols = [_norm_color(palette[k]) if keep_np[k] else (0.85, 0.2, 0.2) for k in range(K)]
    ax2.bar(range(K), vals, color=bar_cols)
    ax2.set_title(key, fontsize=7)
    ax2.tick_params(labelsize=5)

    fig.suptitle("PHASE 4 -- Quality Filter", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase4_multiresponse(img_bgr, I_n, grid_shape, patch_size=14,
                          clean_patch_mask=None, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    img_rgb = _img_to_rgb(img_bgr)
    I_np = I_n.cpu().numpy()
    mx   = int(I_np.max()) if len(I_np) > 0 else 0

    canvas = img_rgb.copy()
    for n in range(H_p * W_p):
        r, c   = n // W_p, n % W_p
        y0, x0 = r * patch_size, c * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        ov = int(I_np[n]) if n < len(I_np) else 0
        if ov == 0:
            col = np.array([0.3, 0.3, 0.3])
        elif ov == 1:
            col = np.array([0.15, 0.82, 0.24])
        else:
            t   = min((ov - 1) / max(mx - 1, 1), 1.0)
            col = np.array([0.15, 0.82 * (1 - t), 0.24 + 0.63 * t])
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * 0.4 + col * 0.6

    fig = Figure(figsize=(W / 100, H / 100))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))
    ax.set_title(f"Overlap max={mx}  (gray=0, green=1, blue≥2)", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 4 -- Multi-Response", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase4_patch_ids(img_bgr, patch_gid_map, grid_shape, patch_size=14, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    img_rgb = _img_to_rgb(img_bgr)
    gid_np  = patch_gid_map.cpu().numpy()
    unique  = np.unique(gid_np[gid_np >= 0])
    palette = make_distinct_colors(max(len(unique), 1))
    g2c     = {int(g): _norm_color(palette[i % len(palette)]) for i, g in enumerate(unique)}

    canvas = img_rgb.copy()
    for n in range(H_p * W_p):
        r, c   = n // W_p, n % W_p
        y0, x0 = r * patch_size, c * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        g = int(gid_np[n]) if n < len(gid_np) else -1
        if g >= 0:
            col = np.array(g2c.get(g, (0.5, 0.5, 0.5)))
            canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * 0.4 + col * 0.6

    fig = Figure(figsize=(W / 100, H / 100))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))
    ax.set_title(f"Patch GID ({len(unique)} groups)", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 4 -- Patch GIDs", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase4_anchor_xfeat(img_bgr, mat_xfeat, kp, grid_shape, patch_size=14,
                         max_anchors=8, save_path=None):
    K      = min(mat_xfeat.shape[0], max_anchors)
    mat_np = mat_xfeat.cpu().numpy()
    kp_np  = kp.cpu().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _img_to_rgb(img_bgr)

    fig = Figure(figsize=(img_bgr.shape[1] / 100, img_bgr.shape[0] / 100))
    ax  = fig.add_subplot(111)
    ax.imshow(img_rgb)
    for k in range(K):
        col = _norm_color(palette[k])
        jj  = np.where(mat_np[k] > 0.01)[0]
        for j in jj:
            if j < len(kp_np):
                ax.plot(kp_np[j, 0], kp_np[j, 1], ".", ms=4, color=col)
    ax.set_title(f"Anchor XFeat binding  K={K}", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 4 -- Anchor-XFeat", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)

# ── Phase 5 ──────────────────────────────────────────────────────

def phase5_bank_state(obj_bank, highlight_gids=None, save_path=None):
    try:
        entries = list(obj_bank.bank.values())
    except Exception:
        entries = []
    if not entries:
        fig = Figure(figsize=(5, 2))
        ax  = fig.add_subplot(111)
        ax.text(0.5, 0.5, "bank empty", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")
        fig.suptitle("PHASE 5 -- Bank State", fontsize=8, color="gold")
        fig.patch.set_facecolor("#111")
        _save_fig(fig, save_path)
        return

    gids  = [getattr(e, "gid", i) for i, e in enumerate(entries)]
    stabs = np.array([getattr(e, "stability", 0.0) for e in entries], dtype=float)
    pemas = np.array([getattr(e, "pure_area_ema", 0.0) for e in entries], dtype=float)
    sstds = np.array([getattr(e, "spatial_std_ema", 0.0) for e in entries], dtype=float)
    palette = make_distinct_colors(max(len(entries), 1))
    hl      = set(highlight_gids or [])
    bar_cols = [
        "cyan" if g in hl else _norm_color(palette[i])
        for i, g in enumerate(gids)
    ]

    fig = Figure(figsize=(9, 3))
    axes = fig.subplots(1, 3)
    for ax, vals, title in zip(axes,
                                [stabs, pemas, sstds],
                                ["stability", "pure_area_ema", "spatial_std_ema"]):
        ax.bar(range(len(vals)), vals, color=bar_cols)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=5)
    fig.suptitle(f"PHASE 5 -- Bank State  size={len(entries)}", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase5_bank_register(img_bgr, gids, vom, grid_shape, patch_size=14,
                          is_new=None, save_path=None):
    K       = vom.shape[0]
    vom_np  = vom.cpu().bool().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _img_to_rgb(img_bgr)
    H, W    = img_bgr.shape[:2]
    H_p, W_p = grid_shape

    canvas = img_rgb.copy()
    for k in range(K):
        col  = _norm_color(palette[k])
        mask = np.zeros_like(canvas)
        for n in np.where(vom_np[k])[0]:
            r, c   = n // W_p, n % W_p
            y0, x0 = r * patch_size, c * patch_size
            y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
            mask[y0:y1, x0:x1] = col
        canvas[:] = canvas * 0.6 + mask * 0.4

    fig = Figure(figsize=(W / 100, H / 100))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))
    for k in range(K):
        cx, cy   = _from_vom(vom_np[k], grid_shape, patch_size)
        new_flag = is_new[k] if is_new is not None else False
        col      = "lime" if new_flag else "deepskyblue"
        ax.plot(cx, cy, "*" if new_flag else "o", ms=8, color=col,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(cx + 5, cy - 5, f"g{gids[k]}{'*' if new_flag else ''}",
                fontsize=5, color="white")
    ax.set_title(f"K={K}  (*=new entry)", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 5 -- Bank Register", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)

# ── Phase 6 ──────────────────────────────────────────────────────

def phase6_cross_match(img1_bgr, img2_bgr, match_mask, match12, kp1, kp2,
                        anchor_labels1=None, max_lines=200, save_path=None):
    H1, W1 = img1_bgr.shape[:2]
    H2, W2 = img2_bgr.shape[:2]
    th = max(H1, H2)

    def _resize_h(im, h):
        import cv2
        oh, ow = im.shape[:2]
        return cv2.resize(im, (max(1, int(ow * h / oh)), h), interpolation=cv2.INTER_LINEAR)

    r1 = _resize_h(img1_bgr, th)
    r2 = _resize_h(img2_bgr, th)
    W1r, W2r = r1.shape[1], r2.shape[1]
    combined = np.hstack([r1[:, :, ::-1], r2[:, :, ::-1]]).astype(np.float32) / 255.0

    mask_np = match_mask.cpu().bool().numpy()
    m12_np  = match12.cpu().numpy()
    kp1_np  = kp1.cpu().numpy()
    kp2_np  = kp2.cpu().numpy()
    al_np   = anchor_labels1.cpu().numpy() if anchor_labels1 is not None else None
    K_anc   = int(al_np.max()) + 1 if al_np is not None else 1
    palette = make_distinct_colors(max(K_anc, 1))

    fig = Figure(figsize=((W1r + W2r) / 100, th / 100))
    ax  = fig.add_subplot(111)
    ax.imshow(combined)

    vidx = np.where(mask_np)[0]
    if len(vidx) > max_lines:
        vidx = vidx[np.linspace(0, len(vidx) - 1, max_lines, dtype=int)]

    s1x, s1y = W1r / W1, th / H1
    s2x, s2y = W2r / W2, th / H2
    for i in vidx:
        if i >= len(kp1_np):
            continue
        j = int(m12_np[i])
        if j >= len(kp2_np):
            continue
        x1b = kp1_np[i, 0] * s1x
        y1b = kp1_np[i, 1] * s1y
        x2b = kp2_np[j, 0] * s2x + W1r
        y2b = kp2_np[j, 1] * s2y
        k_idx = int(al_np[i]) % K_anc if al_np is not None and i < len(al_np) else 0
        col = _norm_color(palette[k_idx])
        ax.plot([x1b, x2b], [y1b, y2b], "-", color=col, lw=0.7, alpha=0.75)
        ax.plot(x1b, y1b, ".", ms=4, color=col)
        ax.plot(x2b, y2b, ".", ms=4, color=col)

    ax.axvline(W1r, color="white", lw=0.8, alpha=0.5)
    ax.set_title(f"matches={int(mask_np.sum())}", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 6 -- Cross-View Match", fontsize=8, color="gold")
    fig.patch.set_facecolor("#111")
    fig.tight_layout()
    _save_fig(fig, save_path)

# ── 컨트롤러 ─────────────────────────────────────────────────────

class PipelineDebugVisualizer:
    """파이프라인 디버그 시각화 컨트롤러.

    Parameters
    ----------
    output_dir : str
        이미지 저장 경로
    enable_phases : set[int] | None
        None이면 전체 활성화. {1,3,5} 처럼 집합으로 지정하면 해당 페이즈만 저장.
    patch_size : int
        DINOv2 패치 크기
    frame_id : str | None
        초기 프레임 ID
    """

    def __init__(self, output_dir="./debug", enable_phases=None,
                 patch_size=14, frame_id=None):
        self.output_dir    = output_dir
        self.enable_phases = enable_phases
        self.patch_size    = patch_size
        self.frame_id      = frame_id
        self._counter      = 0
        os.makedirs(output_dir, exist_ok=True)

    def set_frame_id(self, frame_id):
        self.frame_id  = frame_id
        self._counter += 1

    def _enabled(self, phase: int) -> bool:
        return self.enable_phases is None or phase in self.enable_phases

    def _path(self, name: str) -> str:
        fid = self.frame_id or "noframe"
        return os.path.join(self.output_dir, f"{self._counter:06d}_{fid}_{name}.png")

    # ── Phase 1
    def phase1_attention(self, img, attn, grid_shape):
        if self._enabled(1):
            phase1_attention(img, attn, grid_shape, self.patch_size, self._path("p1_attention"))

    def phase1_xfeat_patches(self, img, kp, bind_mat, grid_shape):
        if self._enabled(1):
            phase1_xfeat_patches(img, kp, bind_mat, grid_shape, self.patch_size, self._path("p1_xfeat"))

    # ── Phase 2
    def phase2_memory_pool(self, pool_vecs, source_labels=None, selected_mask=None):
        if self._enabled(2):
            phase2_memory_pool(pool_vecs, source_labels, selected_mask, self._path("p2_pool"))

    def phase2_neighbor_frames(self, imgs_bgr, sims, selected_idx=None):
        if self._enabled(2):
            phase2_neighbor_frames(imgs_bgr, sims, selected_idx, self._path("p2_neighbors"))

    # ── Phase 3
    def phase3_anchors(self, img, sample, vom, pure, grid_shape, suffix=""):
        if self._enabled(3):
            name = f"p3_anchors_{suffix}" if suffix else "p3_anchors"
            phase3_anchors(img, sample, vom, pure, grid_shape,
                           self.patch_size, suffix, self._path(name))

    def phase3_cross_frame(self, img, vom, pure, valid_mask, grid_shape):
        if self._enabled(3):
            phase3_cross_frame(img, vom, pure, valid_mask, grid_shape,
                               self.patch_size, self._path("p3_cross"))

    # ── Phase 4
    def phase4_quality_filter(self, img, vom, pure, keep, grid_shape, quality_metrics=None):
        if self._enabled(4):
            phase4_quality_filter(img, vom, pure, keep, grid_shape,
                                  self.patch_size, quality_metrics, self._path("p4_quality"))

    def phase4_multiresponse(self, img, I_n, grid_shape, clean_patch_mask=None):
        if self._enabled(4):
            phase4_multiresponse(img, I_n, grid_shape, self.patch_size,
                                 clean_patch_mask, self._path("p4_multiresponse"))

    def phase4_patch_ids(self, img, patch_gid_map, grid_shape):
        if self._enabled(4):
            phase4_patch_ids(img, patch_gid_map, grid_shape,
                             self.patch_size, self._path("p4_patchids"))

    def phase4_anchor_xfeat(self, img, mat_xfeat, kp, grid_shape, max_anchors=8):
        if self._enabled(4):
            phase4_anchor_xfeat(img, mat_xfeat, kp, grid_shape,
                                self.patch_size, max_anchors, self._path("p4_xfeat"))

    # ── Phase 5
    def phase5_bank_state(self, obj_bank, highlight_gids=None):
        if self._enabled(5):
            phase5_bank_state(obj_bank, highlight_gids, self._path("p5_bank"))

    def phase5_bank_register(self, img, gids, vom, grid_shape, is_new=None):
        if self._enabled(5):
            phase5_bank_register(img, gids, vom, grid_shape,
                                 self.patch_size, is_new, self._path("p5_register"))

    # ── Phase 6
    def phase6_cross_match(self, img1, img2, match_mask, match12, kp1, kp2,
                            anchor_labels1=None, max_lines=200):
        if self._enabled(6):
            phase6_cross_match(img1, img2, match_mask, match12, kp1, kp2,
                               anchor_labels1, max_lines, self._path("p6_match"))

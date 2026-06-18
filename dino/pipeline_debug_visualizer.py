"""
pipeline_debug_visualizer.py  —  Matplotlib OO API (thread-safe, no pyplot)

수정 사항 (기존 대비):
  - Figure / FigureCanvasAgg 를 함수 내에서 매번 새로 생성 → 이미지가 고정되는 문제 해결
  - _save_fig: output_dir 가 루트일 때 dirname 이 빈 문자열 → makedirs 예외 수정
  - _patch_overlay_ax 제거: 오버레이를 numpy 배열로 계산 후 ax.imshow 에 직접 전달
  - phase3_anchors: vom / pure 두 패널을 numpy로 완성 후 각각 imshow
  - phase3_cross_frame: valid / invalid 앵커를 색상으로 구분
  - phase4_quality_filter: keep(초록) / drop(빨강) 앵커 명시
  - phase4_multiresponse: I_n float → int 변환 후 overlap 카운트 열지도
  - phase5_bank_state / bank_register: is_new 없을 때 NoneType 오류 수정
  - 모든 함수: fig.clf() 제거 (Figure 를 저장 직후 버리므로 불필요)
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    from anchor_debug_visualizer import make_distinct_colors, draw_patch_mask
except ImportError:
    from dino.anchor_debug_visualizer import make_distinct_colors, draw_patch_mask


# ─────────────────────────────────────────────────────────────────
#  내부 유틸
# ─────────────────────────────────────────────────────────────────

def _c01(c):
    """(R,G,B) 0-255 → 0-1 tuple"""
    return tuple(v / 255.0 for v in c)


def _save_fig(fig: Figure, save_path: str, dpi: int = 120) -> None:
    """Figure → PNG 저장. Figure 객체는 호출 측에서 더 이상 사용하지 않는다."""
    if not save_path:
        return
    parent = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(parent, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"[viz] saved: {save_path}")


def _bgr_to_rgb01(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → RGB float32 [0,1]"""
    return img_bgr[:, :, ::-1].astype(np.float32) / 255.0


def _overlay_patches(img_rgb01: np.ndarray,
                     patch_indices: np.ndarray,
                     grid_shape: tuple,
                     patch_size: int,
                     color_rgb01: tuple,
                     alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb01 위에 patch_indices 위치를 color_rgb01 으로 alpha 블렌딩.
    원본 배열을 수정하지 않고 새 배열 반환.
    """
    out = img_rgb01.copy()
    H, W = out.shape[:2]
    H_p, W_p = grid_shape
    if len(patch_indices) == 0:
        return out
    col = np.array(color_rgb01, dtype=np.float32)

    # 패치 그리드 마스크 → 픽셀 마스크 (벡터화, 파이썬 루프 제거)
    pm = np.zeros(H_p * W_p, dtype=bool)
    idx = np.asarray(patch_indices, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < H_p * W_p)]
    pm[idx] = True
    pix = np.repeat(np.repeat(pm.reshape(H_p, W_p), patch_size, axis=0),
                    patch_size, axis=1)
    ph, pw = min(H_p * patch_size, H), min(W_p * patch_size, W)
    pix = pix[:ph, :pw]
    region = out[:ph, :pw]
    region[pix] = region[pix] * (1 - alpha) + col * alpha
    return out


def _vom_centroid_px(vom_row: np.ndarray, grid_shape: tuple, patch_size: int):
    """vom bool 행 → (cx, cy) 픽셀 중심"""
    H_p, W_p = grid_shape
    idx = np.where(vom_row)[0]
    if len(idx) == 0:
        return W_p * patch_size // 2, H_p * patch_size // 2
    cy = int((idx // W_p).mean() * patch_size + patch_size / 2)
    cx = int((idx  % W_p).mean() * patch_size + patch_size / 2)
    return cx, cy


# ─────────────────────────────────────────────────────────────────
#  Phase 1
# ─────────────────────────────────────────────────────────────────

def phase1_attention(img_bgr, attn, grid_shape, patch_size=14, save_path=None):
    """CLS attention 히트맵 + 원본 이미지 나란히"""
    H, W = img_bgr.shape[:2]
    H_p, W_p = grid_shape
    img_rgb = _bgr_to_rgb01(img_bgr)

    an = attn.cpu().float().numpy().reshape(H_p, W_p)
    an = (an - an.min()) / (an.max() - an.min() + 1e-8)

    heat_small = (an * 255).astype(np.uint8)
    heat_big   = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_rgb   = cv2.applyColorMap(heat_big, cv2.COLORMAP_INFERNO)[:, :, ::-1].astype(np.float32) / 255.0
    blended    = img_rgb * 0.45 + heat_rgb * 0.55

    fig = Figure(figsize=(W * 2 / 100 + 0.5, H / 100 + 0.8))
    axes = fig.subplots(1, 2)
    axes[0].imshow(img_rgb);   axes[0].set_title("Original",       fontsize=8); axes[0].axis("off")
    axes[1].imshow(blended);   axes[1].set_title("CLS Attention",  fontsize=8); axes[1].axis("off")
    fig.suptitle(f"PHASE 1 — CLS Attention  grid=[{H_p}×{W_p}]", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase1_xfeat_patches(img_bgr, kp, bind_mat, grid_shape, patch_size=14, save_path=None):
    """XFeat 키포인트 + 패치 귀속 시각화"""
    H, W = img_bgr.shape[:2]
    img_rgb = _bgr_to_rgb01(img_bgr)
    kp_np   = kp.cpu().numpy()
    bind_np = bind_mat.cpu().numpy()            # [N_patch, M_kp]

    # 키포인트가 귀속된 패치 수
    occupied = (bind_np.sum(axis=1) > 0)        # [N_patch] bool
    occ_idx  = np.where(occupied)[0]

    canvas = _overlay_patches(img_rgb, occ_idx, grid_shape, patch_size,
                               (0.2, 0.8, 0.2), alpha=0.30)

    fig = Figure(figsize=(W / 100 + 0.3, H / 100 + 0.8))
    ax  = fig.add_subplot(111)
    ax.imshow(canvas)
    if len(kp_np) > 0:
        ax.scatter(kp_np[:, 0], kp_np[:, 1], s=4, c="cyan", linewidths=0, alpha=0.7)
    ax.set_title(f"XFeat kp={len(kp_np)}  occupied_patches={int(occupied.sum())}", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 1 — XFeat Patch Binding", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ─────────────────────────────────────────────────────────────────
#  Phase 2
# ─────────────────────────────────────────────────────────────────

def phase2_neighbor_frames(imgs_bgr, sims, selected_idx=None, labels=None, save_path=None):
    """현재 + 인접 프레임 썸네일. labels: 각 패널 텍스트(예: 'CUR 123' / '120')"""
    N = len(imgs_bgr)
    if N == 0:
        return
    sel = set(selected_idx or [])

    TH = 120
    fig = Figure(figsize=(N * (TH * 1.78 / 100) + 0.3, TH / 100 + 1.0))
    axes = fig.subplots(1, N) if N > 1 else [fig.add_subplot(111)]

    for i, (im, sim) in enumerate(zip(imgs_bgr, sims)):
        rgb = _bgr_to_rgb01(im)
        axes[i].imshow(rgb)
        is_sel = i in sel
        col    = "green" if is_sel else "#333"
        lab    = (labels[i] + "  ") if (labels and i < len(labels)) else ""
        axes[i].set_title(f"{'[SEL] ' if is_sel else ''}{lab}sim={sim:.3f}",
                          fontsize=6, color=col)
        for sp in axes[i].spines.values():
            sp.set_edgecolor("lime" if is_sel else "#555")
            sp.set_linewidth(2.5 if is_sel else 0.8)
        axes[i].axis("off")

    fig.suptitle("PHASE 2 — Current + Neighbor Frames", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase2_memory_pool(pool_vecs, source_labels=None, selected_mask=None, save_path=None):
    """메모리 풀 벡터 PCA 2D 산점도 (pool_vecs: [M, D] tensor)"""
    if pool_vecs is None or pool_vecs.shape[0] == 0:
        return
    try:
        from sklearn.decomposition import PCA
        vecs = pool_vecs.cpu().float().numpy()
        pca  = PCA(n_components=2)
        xy   = pca.fit_transform(vecs)
    except Exception:
        return

    M  = vecs.shape[0]
    sel = np.array(selected_mask.cpu().bool().numpy() if selected_mask is not None
                   else [False] * M)

    fig = Figure(figsize=(5, 4))
    ax  = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.scatter(xy[~sel, 0], xy[~sel, 1], c="steelblue", s=15, alpha=0.6, label="pool")
    if sel.any():
        ax.scatter(xy[sel, 0], xy[sel, 1], c="lime", s=25, zorder=5, label="selected")
    ax.legend(fontsize=6)
    ax.set_title(f"Memory pool  M={M}  (PCA 2D)", fontsize=7)
    fig.suptitle("PHASE 2 — Memory Pool", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ─────────────────────────────────────────────────────────────────
#  Phase 3
# ─────────────────────────────────────────────────────────────────

def phase3_anchors(img_bgr, sample, vom, pure, grid_shape,
                   patch_size=14, title_suffix="", save_path=None):
    """VOM / Pure 두 패널로 앵커 영역 시각화"""
    K = vom.shape[0]
    if K == 0:
        return
    vom_np  = vom.cpu().bool().numpy()
    pure_np = pure.cpu().bool().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _bgr_to_rgb01(img_bgr)
    H, W    = img_rgb.shape[:2]

    canvas_vom  = img_rgb.copy()
    canvas_pure = img_rgb.copy()
    for k in range(K):
        col = _c01(palette[k])
        canvas_vom  = _overlay_patches(canvas_vom,  np.where(vom_np[k])[0],
                                        grid_shape, patch_size, col, 0.40)
        canvas_pure = _overlay_patches(canvas_pure, np.where(pure_np[k])[0],
                                        grid_shape, patch_size, col, 0.50)

    fig = Figure(figsize=(W * 2 / 100 + 0.5, H / 100 + 1.0))
    ax0, ax1 = fig.subplots(1, 2)

    ax0.imshow(np.clip(canvas_vom,  0, 1))
    ax1.imshow(np.clip(canvas_pure, 0, 1))

    for k in range(K):
        cx, cy = _vom_centroid_px(vom_np[k], grid_shape, patch_size)
        col    = _c01(palette[k])
        for ax in (ax0, ax1):
            ax.plot(cx, cy, "o", ms=5, color=col, markeredgecolor="white", markeredgewidth=0.5)
            ax.text(cx + 4, cy - 4, str(k), fontsize=5, color="white")

    ax0.set_title(f"VOM  K={K}", fontsize=7); ax0.axis("off")
    ax1.set_title("Pure (oc==1)",  fontsize=7); ax1.axis("off")
    suf = f"  [{title_suffix}]" if title_suffix else ""
    fig.suptitle(f"PHASE 3 — Anchors{suf}", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase3_cross_frame(img_bgr, vom, pure, valid_mask, grid_shape,
                       patch_size=14, save_path=None):
    """크로스 프레임 투영 결과 — 유효(lime) / 배제(red) 앵커"""
    K = vom.shape[0]
    if K == 0:
        return
    vom_np   = vom.cpu().bool().numpy()
    valid_np = valid_mask.cpu().bool().numpy()
    palette  = make_distinct_colors(max(K, 1))
    img_rgb  = _bgr_to_rgb01(img_bgr)
    H, W     = img_rgb.shape[:2]

    canvas = img_rgb.copy()
    for k in range(K):
        col    = _c01(palette[k])
        alpha  = 0.40 if valid_np[k] else 0.15
        canvas = _overlay_patches(canvas, np.where(vom_np[k])[0],
                                   grid_shape, patch_size, col, alpha)

    fig = Figure(figsize=(W / 100 + 0.3, H / 100 + 1.0))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))
    for k in range(K):
        cx, cy = _vom_centroid_px(vom_np[k], grid_shape, patch_size)
        dot_col = "lime" if valid_np[k] else "red"
        ax.plot(cx, cy, "o", ms=6, color=dot_col,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(cx + 4, cy - 4,
                f"{k} O" if valid_np[k] else f"{k} X",
                fontsize=5, color=dot_col)
    ax.set_title(f"K={K}  valid={int(valid_np.sum())}  invalid={int((~valid_np).sum())}",
                 fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 3 — Cross-Frame Projection", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase3_cross_context(img_bgr, ctx_mem, grid_shape, valid_mask=None,
                         patch_size=14, max_anchors=6, save_path=None):
    """크로스 프레임 메모리 컨텍스트(ctx_mem) 진단.

    패널 구성:
      [0] heatmap_sim [M×M] — 메모리 앵커 간 현재프레임 응답 패턴 유사도 (병합/중복 후보)
      [1] oc 오버랩 맵       — 패치별로 몇 개 메모리 앵커가 점유하는지
      [2..] 상위 앵커별 sim 연속 응답 — 각 메모리 앵커가 현재 프레임 어디에 반응하는지
    """
    sim_mat = ctx_mem.get("sim_matrix")
    if sim_mat is None or sim_mat.shape[0] == 0:
        return
    sim_np  = sim_mat.detach().cpu().float().numpy()              # [M, N]
    hsim    = ctx_mem.get("heatmap_sim")
    hsim_np = hsim.detach().cpu().float().numpy() if hsim is not None else None
    oc      = ctx_mem.get("oc")
    oc_np   = oc.detach().cpu().float().numpy() if oc is not None else None
    pure    = ctx_mem.get("pure")
    pure_np = pure.detach().cpu().bool().numpy() if pure is not None else None

    M, N     = sim_np.shape
    H_p, W_p = grid_shape
    img_rgb  = _bgr_to_rgb01(img_bgr)
    H, W     = img_rgb.shape[:2]

    valid_np = (valid_mask.detach().cpu().bool().numpy()
                if valid_mask is not None else np.ones(M, dtype=bool))
    pure_area = (pure_np.sum(axis=1) if pure_np is not None
                 else (sim_np > 0).sum(axis=1))

    # 보여줄 앵커: valid 우선, pure_area 큰 순
    order = list(np.argsort(-pure_area))
    show  = [m for m in order if valid_np[m]][:max_anchors]
    if not show:
        show = order[:max_anchors]
    n_show = len(show)

    n_panels = 2 + n_show
    fig  = Figure(figsize=(2.6 * n_panels + 0.5, 3.0))
    axes = fig.subplots(1, n_panels)
    if n_panels == 1:
        axes = [axes]

    # [0] heatmap_sim 행렬
    ax = axes[0]
    if hsim_np is not None and hsim_np.size:
        ax.imshow(hsim_np, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"heatmap_sim [{M}x{M}]\n앵커간 응답 유사도(병합후보)", fontsize=6)
        ax.set_xlabel("anchor id", fontsize=5)
        ax.set_ylabel("anchor id", fontsize=5)
        ax.tick_params(labelsize=4)
    else:
        ax.axis("off")

    # [1] oc 오버랩 맵
    ax = axes[1]
    if oc_np is not None:
        grid = np.zeros(H_p * W_p)
        grid[:min(len(oc_np), H_p * W_p)] = oc_np[:H_p * W_p]
        gmax = max(float(grid.max()), 1.0)
        big  = cv2.resize((np.clip(grid / gmax, 0, 1).reshape(H_p, W_p) * 255).astype(np.uint8),
                          (W, H), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(big, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32) / 255.0
        ax.imshow(np.clip(img_rgb * 0.4 + heat * 0.6, 0, 1))
        ax.set_title(f"oc 오버랩\n패치별 점유 앵커수(max={int(grid.max())})", fontsize=6)
    ax.axis("off")

    # [2..] 앵커별 sim 연속 응답
    for j, m in enumerate(show):
        ax = axes[2 + j]
        resp = np.clip(sim_np[m, :H_p * W_p], 0, None).reshape(H_p, W_p)
        resp = resp / (resp.max() + 1e-8)
        big  = cv2.resize((resp * 255).astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)[:, :, ::-1].astype(np.float32) / 255.0
        ax.imshow(np.clip(img_rgb * 0.45 + heat * 0.55, 0, 1))
        ax.set_title(f"anchor {m}  pure={int(pure_area[m])}", fontsize=6,
                     color=("lime" if valid_np[m] else "red"))
        for sp in ax.spines.values():
            sp.set_edgecolor("lime" if valid_np[m] else "red")
            sp.set_linewidth(2.0)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"PHASE 3 — Cross-Frame Memory Context  (M={M}, valid={int(valid_np.sum())})",
                 fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ─────────────────────────────────────────────────────────────────
#  Phase 4
# ─────────────────────────────────────────────────────────────────

def phase4_quality_filter(img_bgr, vom, pure, keep, grid_shape,
                           patch_size=14, quality_metrics=None, save_path=None):
    """품질 필터 결과 — keep(lime) / drop(red) 앵커"""
    K = vom.shape[0]
    if K == 0:
        return
    vom_np  = vom.cpu().bool().numpy()
    keep_np = keep.cpu().bool().numpy() if hasattr(keep, "cpu") else np.array(keep, dtype=bool)
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _bgr_to_rgb01(img_bgr)
    H, W    = img_rgb.shape[:2]

    canvas_keep = img_rgb.copy()
    canvas_drop = img_rgb.copy()
    for k in range(K):
        col   = _c01(palette[k])
        alpha = 0.45
        if keep_np[k]:
            canvas_keep = _overlay_patches(canvas_keep, np.where(vom_np[k])[0],
                                            grid_shape, patch_size, col, alpha)
        else:
            canvas_drop = _overlay_patches(canvas_drop, np.where(vom_np[k])[0],
                                            grid_shape, patch_size, col, alpha)

    n_keep = int(keep_np.sum())
    n_drop = K - n_keep

    fig = Figure(figsize=(W * 2 / 100 + 0.5, H / 100 + 1.0))
    ax0, ax1 = fig.subplots(1, 2)
    ax0.imshow(np.clip(canvas_keep, 0, 1))
    ax1.imshow(np.clip(canvas_drop, 0, 1))

    for k in range(K):
        cx, cy = _vom_centroid_px(vom_np[k], grid_shape, patch_size)
        col = _c01(palette[k])
        ax = ax0 if keep_np[k] else ax1
        ax.plot(cx, cy, "o", ms=5, color=col,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(cx + 3, cy - 3, str(k), fontsize=5, color="white")
        if quality_metrics is not None and k < len(quality_metrics.get("pure_area", [])):
            pa = quality_metrics["pure_area"][k]
            ax.text(cx + 3, cy + 8, f"pa={pa}", fontsize=4, color="yellow")

    ax0.set_title(f"KEEP  ({n_keep})", fontsize=7, color="lime"); ax0.axis("off")
    ax1.set_title(f"DROP  ({n_drop})", fontsize=7, color="red");  ax1.axis("off")
    fig.suptitle("PHASE 4 — Quality Filter", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase4_multiresponse(img_bgr, I_n, grid_shape, patch_size=14,
                          clean_patch_mask=None, save_path=None):
    """Multi-response overlap 카운트 열지도"""
    H, W    = img_bgr.shape[:2]
    H_p, W_p = grid_shape
    img_rgb = _bgr_to_rgb01(img_bgr)

    # I_n: [N] float or long
    counts = I_n.cpu().float().numpy() if hasattr(I_n, "cpu") else np.array(I_n, dtype=float)
    mx     = int(counts.max()) if len(counts) > 0 else 0

    # 패치별 색상 [Np,3] 벡터화 계산 후 픽셀로 업스케일 (파이썬 루프 제거)
    Np = H_p * W_p
    c  = np.zeros(Np, dtype=float)
    n_use = min(len(counts), Np)
    c[:n_use] = counts[:n_use]
    cols = np.tile(np.array([0.15, 0.15, 0.15], dtype=np.float32), (Np, 1))
    one  = (c == 1)
    cols[one] = np.array([0.15, 0.75, 0.25], dtype=np.float32)
    multi = (c >= 2)
    t = np.clip((c - 1) / max(mx - 1, 1), 0.0, 1.0)
    cols[multi, 0] = 0.3 + 0.7 * t[multi]
    cols[multi, 1] = 0.2
    cols[multi, 2] = 0.9 - 0.7 * t[multi]
    pix = np.repeat(np.repeat(cols.reshape(H_p, W_p, 3), patch_size, axis=0),
                    patch_size, axis=1)
    ph, pw = min(H_p * patch_size, H), min(W_p * patch_size, W)
    canvas = img_rgb.copy()
    canvas[:ph, :pw] = canvas[:ph, :pw] * 0.35 + pix[:ph, :pw] * 0.65

    fig = Figure(figsize=(W / 100 + 0.3, H / 100 + 1.0))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))
    ax.set_title(f"max_overlap={mx}  (dark=0, green=1, blue/red≥2)", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 4 — Multi-Response Overlap", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase4_patch_ids(img_bgr, patch_gid_map, grid_shape, patch_size=14, save_path=None):
    """패치별 global object ID 시각화"""
    H, W     = img_bgr.shape[:2]
    H_p, W_p = grid_shape
    img_rgb  = _bgr_to_rgb01(img_bgr)

    gid_np = patch_gid_map.cpu().numpy() if hasattr(patch_gid_map, "cpu") else np.array(patch_gid_map)
    unique = np.unique(gid_np[gid_np >= 0])
    palette = make_distinct_colors(max(len(unique), 1))
    g2c = {int(g): _c01(palette[i % len(palette)]) for i, g in enumerate(unique)}

    canvas = img_rgb.copy()
    for n in range(H_p * W_p):
        g = int(gid_np[n]) if n < len(gid_np) else -1
        if g < 0:
            continue
        r, c   = n // W_p, n % W_p
        y0, x0 = r * patch_size, c * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        col = np.array(g2c.get(g, (0.5, 0.5, 0.5)))
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * 0.40 + col * 0.60

    fig = Figure(figsize=(W / 100 + 0.3, H / 100 + 1.0))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))
    ax.set_title(f"Patch GID  n_objects={len(unique)}", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 4 — Patch GIDs", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase4_anchor_xfeat(img_bgr, mat_xfeat, kp, grid_shape,
                         patch_size=14, max_anchors=8, save_path=None):
    """앵커별 귀속 XFeat 키포인트 시각화"""
    H, W   = img_bgr.shape[:2]
    K      = min(mat_xfeat.shape[0], max_anchors)
    mat_np = mat_xfeat.cpu().numpy()
    kp_np  = kp.cpu().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _bgr_to_rgb01(img_bgr)

    fig = Figure(figsize=(W / 100 + 0.3, H / 100 + 1.0))
    ax  = fig.add_subplot(111)
    ax.imshow(img_rgb)
    for k in range(K):
        col = _c01(palette[k])
        jj  = np.where(mat_np[k] > 0.01)[0]
        for j in jj:
            if j < len(kp_np):
                ax.plot(kp_np[j, 0], kp_np[j, 1], ".",
                        ms=4, color=col, alpha=0.8)
    ax.set_title(f"Anchor-XFeat binding  K={K}", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 4 — Anchor XFeat", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ─────────────────────────────────────────────────────────────────
#  Phase 5
# ─────────────────────────────────────────────────────────────────

def phase5_bank_state(obj_bank, highlight_gids=None, save_path=None):
    """GlobalObjectBank 상태 바 차트 (stability / pure_area_ema / spatial_std_ema)"""
    try:
        # GlobalObjectBank 는 entry 를 self._entries 에 저장한다 (self.bank 아님).
        store = getattr(obj_bank, "_entries", getattr(obj_bank, "bank", {}))
        entries = list(store.values())
    except Exception:
        entries = []

    if not entries:
        fig = Figure(figsize=(4, 2))
        ax  = fig.add_subplot(111)
        ax.text(0.5, 0.5, "bank empty", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.axis("off")
        fig.suptitle("PHASE 5 — Bank State", fontsize=9, color="#222")
        fig.patch.set_facecolor("white")
        _save_fig(fig, save_path)
        return

    gids  = [getattr(e, "gid",           i)   for i, e in enumerate(entries)]
    stabs = np.array([getattr(e, "stability",       0.0) for e in entries], dtype=float)
    pemas = np.array([getattr(e, "pure_area_ema",   0.0) for e in entries], dtype=float)
    sstds = np.array([getattr(e, "spatial_std_ema", 0.0) for e in entries], dtype=float)

    palette  = make_distinct_colors(max(len(entries), 1))
    hl       = set(highlight_gids or [])
    bar_cols = ["cyan" if g in hl else _c01(palette[i]) for i, g in enumerate(gids)]

    fig  = Figure(figsize=(11, 3))
    axes = fig.subplots(1, 4)
    for ax, vals, title in zip(axes[:3],
                                [stabs, pemas, sstds],
                                ["stability\n관측 안정성 (1=안정)",
                                 "pure_area_ema\n독점 패치 수 (클수록 또렷)",
                                 "spatial_std_ema\n공간 분산 (클수록 발산)"]):
        bars = ax.bar(range(len(vals)), vals, color=bar_cols)
        for b, v in zip(bars, vals):                       # 막대 위 수치 표기
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=4, color="#222")
        ax.set_xticks(range(len(gids)))
        ax.set_xticklabels([f"g{g}" for g in gids], fontsize=5, rotation=45)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_facecolor("white")

    # 4번째 패널 — 각 entry 수치 텍스트 요약
    axes[3].axis("off")
    axes[3].set_title("gid: stab / pure / sstd", fontsize=6, color="#222")
    _lines = [f"g{g:>3}: {st:.2f} / {pa:5.1f} / {ss:4.1f}"
              for g, st, pa, ss in zip(gids, stabs, pemas, sstds)]
    axes[3].text(0.0, 1.0, "\n".join(_lines[:25]) if _lines else "(empty)",
                 va="top", ha="left", fontsize=5, color="#222",
                 family="monospace", transform=axes[3].transAxes)

    fig.suptitle(f"PHASE 5 — Bank State  size={len(entries)}", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


def phase5_bank_register(img_bgr, gids, vom, grid_shape,
                          patch_size=14, is_new=None, save_path=None):
    """bank_register 결과 — 앵커 vom 영역 + gid 레이블"""
    K = vom.shape[0]
    if K == 0:
        return
    vom_np  = vom.cpu().bool().numpy()
    palette = make_distinct_colors(max(K, 1))
    img_rgb = _bgr_to_rgb01(img_bgr)
    H, W    = img_rgb.shape[:2]

    canvas = img_rgb.copy()
    for k in range(K):
        col    = _c01(palette[k])
        canvas = _overlay_patches(canvas, np.where(vom_np[k])[0],
                                   grid_shape, patch_size, col, 0.40)

    fig = Figure(figsize=(W / 100 + 0.3, H / 100 + 1.0))
    ax  = fig.add_subplot(111)
    ax.imshow(np.clip(canvas, 0, 1))

    for k in range(K):
        cx, cy    = _vom_centroid_px(vom_np[k], grid_shape, patch_size)
        new_flag  = bool(is_new[k]) if (is_new is not None and k < len(is_new)) else False
        dot_col   = "lime" if new_flag else "deepskyblue"
        marker    = "*" if new_flag else "o"
        ax.plot(cx, cy, marker, ms=8, color=dot_col,
                markeredgecolor="white", markeredgewidth=0.5)
        ax.text(cx + 5, cy - 5,
                f"g{gids[k]}{'*' if new_flag else ''}",
                fontsize=5, color="white")

    n_new = int(sum(bool(is_new[k]) for k in range(K))) if is_new is not None else 0
    ax.set_title(f"K={K}  new={n_new}  (*=new entry)", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 5 — Bank Register", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ─────────────────────────────────────────────────────────────────
#  Phase 6
# ─────────────────────────────────────────────────────────────────

def phase6_cross_match(img1_bgr, img2_bgr, match_mask, match12, kp1, kp2,
                        anchor_labels1=None, max_lines=200, save_path=None):
    """크로스 뷰 매칭 결과 — 좌우 이미지에 매칭 선 표시

    match_mask : [K1, K2, F1] bool  또는  [F1] bool  (1D 하위 호환)
    match12    : [K1, K2, F1] long  또는  [F1] long
    """
    H1, W1 = img1_bgr.shape[:2]
    H2, W2 = img2_bgr.shape[:2]
    th     = max(H1, H2)

    def _rh(im, h):
        oh, ow = im.shape[:2]
        return cv2.resize(im, (max(1, int(ow * h / oh)), h),
                          interpolation=cv2.INTER_LINEAR)

    r1 = _bgr_to_rgb01(_rh(img1_bgr, th))
    r2 = _bgr_to_rgb01(_rh(img2_bgr, th))
    W1r, W2r = r1.shape[1], r2.shape[1]
    combined  = np.concatenate([r1, r2], axis=1)   # [th, W1r+W2r, 3]

    kp1_np = kp1.cpu().numpy()
    kp2_np = kp2.cpu().numpy()
    mask_t = match_mask.cpu().bool()
    m12_t  = match12.cpu()

    # ── 3D [K1, K2, F1] → (f1_idx, f2_idx, pair_color_idx) 목록으로 flatten ──
    pairs = []   # list of (f1_idx, f2_idx, pair_idx)
    if mask_t.dim() == 3:
        K1, K2, F1 = mask_t.shape
        palette = make_distinct_colors(min(max(K1 * K2, 1), 256))  # 팔레트 폭발 방지
        pair_idx = 0
        for k1 in range(K1):
            for k2 in range(K2):
                f1_hits = torch.where(mask_t[k1, k2])[0].numpy()
                for f1 in f1_hits:
                    f2 = int(m12_t[k1, k2, f1].item())
                    pairs.append((int(f1), f2, pair_idx))
                pair_idx += 1
    else:
        # 1D fallback (하위 호환)
        palette  = make_distinct_colors(1)
        f1_hits  = torch.where(mask_t)[0].numpy()
        m12_flat = m12_t.numpy().flatten()
        for f1 in f1_hits:
            f2 = int(m12_flat[f1]) if f1 < len(m12_flat) else -1
            pairs.append((int(f1), f2, 0))

    total_matches = len(pairs)
    if len(pairs) > max_lines:
        step  = max(1, len(pairs) // max_lines)
        pairs = pairs[::step][:max_lines]

    s1x, s1y = W1r / W1, th / H1
    s2x, s2y = W2r / W2, th / H2

    fig = Figure(figsize=((W1r + W2r) / 100 + 0.3, th / 100 + 1.0))
    ax  = fig.add_subplot(111)
    ax.imshow(combined)

    for f1, f2, pidx in pairs:
        if f1 >= len(kp1_np) or f2 < 0 or f2 >= len(kp2_np):
            continue
        col  = _c01(palette[pidx % len(palette)])
        x1b  = kp1_np[f1, 0] * s1x
        y1b  = kp1_np[f1, 1] * s1y
        x2b  = kp2_np[f2, 0] * s2x + W1r
        y2b  = kp2_np[f2, 1] * s2y
        ax.plot([x1b, x2b], [y1b, y2b], "-", color=col, lw=0.7, alpha=0.70)
        ax.plot(x1b, y1b, ".", ms=3, color=col)
        ax.plot(x2b, y2b, ".", ms=3, color=col)

    ax.axvline(W1r, color="white", lw=0.8, alpha=0.5)
    ax.set_title(f"total_matches={total_matches}  (showing {len(pairs)})", fontsize=7)
    ax.axis("off")
    fig.suptitle("PHASE 6 — Cross-View Match", fontsize=9, color="#222")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    _save_fig(fig, save_path)


# ─────────────────────────────────────────────────────────────────
#  컨트롤러
# ─────────────────────────────────────────────────────────────────

class PipelineDebugVisualizer:
    """
    파이프라인 디버그 시각화 컨트롤러.

    Parameters
    ----------
    output_dir : str
        PNG 저장 경로 (자동 생성됨)
    enable_phases : set[int] | None
        None 이면 전체 Phase 활성화.
        예: {2, 3, 4} 이면 해당 Phase 만 저장.
    patch_size : int
        DINOv2 패치 크기 (기본 14)
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

    def set_frame_id(self, frame_id: str) -> None:
        self.frame_id  = frame_id
        self._counter += 1

    def _enabled(self, phase: int) -> bool:
        return self.enable_phases is None or phase in self.enable_phases

    def _path(self, name: str) -> str:
        fid = self.frame_id or "noframe"
        return os.path.join(self.output_dir, f"{self._counter:06d}_{fid}_{name}.png")

    # ── Phase 1 ──────────────────────────────────────────────────

    def phase1_attention(self, img, attn, grid_shape):
        if self._enabled(1):
            phase1_attention(img, attn, grid_shape,
                             self.patch_size, self._path("p1_attention"))

    def phase1_xfeat_patches(self, img, kp, bind_mat, grid_shape):
        if self._enabled(1):
            phase1_xfeat_patches(img, kp, bind_mat, grid_shape,
                                 self.patch_size, self._path("p1_xfeat"))

    # ── Phase 2 ──────────────────────────────────────────────────

    def phase2_memory_pool(self, pool_vecs, source_labels=None, selected_mask=None):
        if self._enabled(2):
            phase2_memory_pool(pool_vecs, source_labels, selected_mask,
                               self._path("p2_pool"))

    def phase2_neighbor_frames(self, imgs_bgr, sims, selected_idx=None, labels=None):
        if self._enabled(2):
            phase2_neighbor_frames(imgs_bgr, sims, selected_idx, labels,
                                   self._path("p2_neighbors"))

    # ── Phase 3 ──────────────────────────────────────────────────

    def phase3_anchors(self, img, sample, vom, pure, grid_shape, suffix=""):
        if self._enabled(3):
            name = f"p3_anchors_{suffix}" if suffix else "p3_anchors"
            phase3_anchors(img, sample, vom, pure, grid_shape,
                           self.patch_size, suffix, self._path(name))

    def phase3_cross_frame(self, img, vom, pure, valid_mask, grid_shape):
        if self._enabled(3):
            phase3_cross_frame(img, vom, pure, valid_mask, grid_shape,
                               self.patch_size, self._path("p3_cross"))

    def phase3_cross_context(self, img, ctx_mem, grid_shape, valid_mask=None,
                             max_anchors=6):
        if self._enabled(3):
            phase3_cross_context(img, ctx_mem, grid_shape, valid_mask,
                                 self.patch_size, max_anchors,
                                 self._path("p3_cross_ctx"))

    # ── Phase 4 ──────────────────────────────────────────────────

    def phase4_quality_filter(self, img, vom, pure, keep, grid_shape,
                               quality_metrics=None):
        if self._enabled(4):
            phase4_quality_filter(img, vom, pure, keep, grid_shape,
                                  self.patch_size, quality_metrics,
                                  self._path("p4_quality"))

    def phase4_multiresponse(self, img, I_n, grid_shape, clean_patch_mask=None):
        if self._enabled(4):
            phase4_multiresponse(img, I_n, grid_shape,
                                 self.patch_size, clean_patch_mask,
                                 self._path("p4_multiresponse"))

    def phase4_patch_ids(self, img, patch_gid_map, grid_shape):
        if self._enabled(4):
            phase4_patch_ids(img, patch_gid_map, grid_shape,
                             self.patch_size, self._path("p4_patchids"))

    def phase4_anchor_xfeat(self, img, mat_xfeat, kp, grid_shape, max_anchors=8):
        if self._enabled(4):
            phase4_anchor_xfeat(img, mat_xfeat, kp, grid_shape,
                                self.patch_size, max_anchors,
                                self._path("p4_xfeat"))

    # ── Phase 5 ──────────────────────────────────────────────────

    def phase5_bank_state(self, obj_bank, highlight_gids=None):
        if self._enabled(5):
            phase5_bank_state(obj_bank, highlight_gids,
                              self._path("p5_bank"))

    def phase5_bank_register(self, img, gids, vom, grid_shape, is_new=None):
        if self._enabled(5):
            phase5_bank_register(img, gids, vom, grid_shape,
                                 self.patch_size, is_new,
                                 self._path("p5_register"))

    # ── Phase 6 ──────────────────────────────────────────────────

    def phase6_cross_match(self, img1, img2, match_mask, match12, kp1, kp2,
                            anchor_labels1=None, max_lines=200):
        if self._enabled(6):
            phase6_cross_match(img1, img2, match_mask, match12, kp1, kp2,
                               anchor_labels1, max_lines,
                               self._path("p6_match"))
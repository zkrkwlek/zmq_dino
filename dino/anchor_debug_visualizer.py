"""
anchor_debug_visualizer.py
──────────────────────────────────────────────────────────────────
patchcluster_new_methods 의 각 단계를 시각화하는 디버그 툴.

사용법:
    from anchor_debug_visualizer import AnchorDebugVisualizer
    viz = AnchorDebugVisualizer(grid_shape=(34, 45), patch_size=14)

    # 1. vom/pure 오버레이 (자기 시드 마스킹 버그 픽스 확인)
    viz.show_vom_pure(img_bgr, ctx, anchor_indices=[0, 5, 12])

    # 2. 히트맵 유사도 행렬 + 병합 결과
    viz.show_heatmap_sim(ctx, labels, n_comp)

    # 3. 앵커 병합 before/after
    viz.show_merge_result(img_bgr, ctx['centroids'], new_centroids, grid_shape)

    # 4. P3 단편화 필터 결과
    viz.show_p3_filter(img_bgr, group_vom, keep, grid_shape)

    # 5. cross-frame 필터 결과 (P1/P2)
    viz.show_cross_frame_filter(img1_bgr, img2_bgr,
                                 avg_vec_mem, feat_cur,
                                 valid_mask, vom_cur, pure_area_cur,
                                 grid_shape)

검증 체크리스트:
    □ show_vom_pure   : 시드 패치가 자신의 vom (파랑) 안에 반드시 포함되는가?
    □ show_heatmap_sim: 같은 객체 앵커끼리 클러스터링 되는가?
    □ show_merge_result: K개 앵커 → M(<K)개로 줄었는가?
    □ show_p3_filter  : 두 곳에 흩어진 앵커가 제거되는가?
    □ show_cross_frame_filter: 없는 객체/바닥 앵커가 배제되는가?
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import colorsys
import math
import os
from typing import List, Optional, Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

font_path = "C:/Windows/Fonts/malgun.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────
# 유틸: 최대 분산 색상 팔레트
# ─────────────────────────────────────────────────────────────────

def make_distinct_colors(K: int):
    """
    K개 색상을 가능한 한 고르게 분산.

    Golden ratio hue spacing + 3-tier saturation/value 교번.
    tab20처럼 20개 이상에서 색이 겹치는 문제 해결.

    Returns:
        colors : list of (R, G, B) int tuples  len=K
    """
    colors = []
    golden = 0.618033988749895
    h = 0.05  # 시작 오프셋
    # (saturation, value) 3단계 — 같은 hue라도 밝기로 구분
    tiers = [(0.90, 0.95), (0.65, 0.88), (0.45, 1.00)]
    for i in range(K):
        h = (h + golden) % 1.0
        s, v = tiers[i % len(tiers)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


# ─────────────────────────────────────────────────────────────────
# 유틸: 패치 인덱스 → 이미지 좌표
# ─────────────────────────────────────────────────────────────────

def patch_to_pixel(patch_idx: int, grid_shape: Tuple[int, int],
                   patch_size: int = 14) -> Tuple[int, int]:
    """패치 인덱스 → (x, y) 픽셀 좌표 (패치 중심)."""
    H_p, W_p = grid_shape
    r = patch_idx // W_p
    c = patch_idx % W_p
    x = int(c * patch_size + patch_size // 2)
    y = int(r * patch_size + patch_size // 2)
    return x, y


def draw_patch_mask(canvas: np.ndarray, patch_indices: np.ndarray,
                    grid_shape: Tuple[int, int], patch_size: int,
                    color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """패치 인덱스 목록을 반투명 컬러 사각형으로 오버레이."""
    H_p, W_p = grid_shape
    overlay = canvas.copy()
    for idx in patch_indices:
        r = idx // W_p
        c = idx % W_p
        x0 = c * patch_size
        y0 = r * patch_size
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    return cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)


# ─────────────────────────────────────────────────────────────────
# 메인 시각화 클래스
# ─────────────────────────────────────────────────────────────────

class AnchorDebugVisualizer:
    """
    Args:
        grid_shape : (H_p, W_p) — DINOv2 패치 그리드 (예: (34, 45))
        patch_size : 픽셀 단위 패치 크기 (ViT-S/14 → 14)
        save_dir   : None이면 plt.show(), 경로 지정 시 파일로 저장
    """

    def __init__(self, grid_shape: Tuple[int, int] = (34, 45),
                 patch_size: int = 14,
                 save_dir: Optional[str] = None,
                 frame_id: Optional[str] = None):
        self.grid_shape = grid_shape
        self.patch_size = patch_size
        self.save_dir = save_dir
        self.frame_id = frame_id
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def set_frame_id(self, frame_id: str):
        self.frame_id = frame_id

    def _show_or_save(self, fig: plt.Figure, filename: str):
        if self.frame_id:
            filename = f"{self.frame_id}_{filename}"
        if self.save_dir:
            path = os.path.join(self.save_dir, filename)
            fig.savefig(path, dpi=120, bbox_inches='tight')
            print(f"[viz] 저장됨: {path}")
        else:
            plt.tight_layout()
            plt.show()
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # 1. vom / pure 오버레이
    # ─────────────────────────────────────────────────────────────

    def show_vom_pure(self, img_bgr: np.ndarray, ctx: dict,
                      anchor_indices: Optional[List[int]] = None,
                      filename: str = "debug_vom_pure.png"):
        """
        각 앵커에 대해 vom/pure 오버레이 + vom 겹침 앵커 표시.

        색상:
          파랑  : vom 전용 (pure 아닌 vom) — 다른 앵커와 겹치는 영역
          초록  : pure (단독 점유)
          노랑테두리 점 : vom 패치를 공유하는 다른 앵커 시드
          빨간 원 : 현재 앵커 시드
        """
        vom_np    = ctx['vom'].cpu().numpy()    # [K, N] bool
        pure_np   = ctx['pure'].cpu().numpy()   # [K, N] bool
        centroids = ctx['centroids'].cpu().numpy()  # [K]
        K = ctx['K']

        indices = anchor_indices if anchor_indices is not None else list(range(min(K, 12)))
        n = len(indices)
        cols = 4
        rows = math.ceil(n / cols)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = np.array(axes).flatten()

        for ax_i, k in enumerate(indices):
            canvas = img_rgb.copy()
            vom_patches  = np.where(vom_np[k])[0]
            pure_patches = np.where(pure_np[k])[0]

            # vom 전용(파랑): pure가 아닌 vom 패치 — 겹치는 영역
            vom_only = np.setdiff1d(vom_patches, pure_patches)
            canvas = draw_patch_mask(canvas, vom_only,
                                     self.grid_shape, self.patch_size,
                                     color=(30, 100, 255), alpha=0.50)
            # pure(초록): 단독 점유
            canvas = draw_patch_mask(canvas, pure_patches,
                                     self.grid_shape, self.patch_size,
                                     color=(30, 220, 60), alpha=0.55)

            # vom 패치를 공유하는 다른 앵커 시드 표시 (노란 점)
            if len(vom_patches) > 0:
                overlap_anchor_ids = []
                for other_k in range(K):
                    if other_k == k:
                        continue
                    if np.any(vom_np[other_k][vom_patches]):
                        overlap_anchor_ids.append(other_k)
                for other_k in overlap_anchor_ids:
                    ox, oy = patch_to_pixel(int(centroids[other_k]),
                                            self.grid_shape, self.patch_size)
                    cv2.circle(canvas, (ox, oy), self.patch_size // 2 + 1,
                               (255, 220, 0), 2)

            # 현재 앵커 시드: 빨간 원
            sx, sy = patch_to_pixel(int(centroids[k]), self.grid_shape, self.patch_size)
            cv2.circle(canvas, (sx, sy), self.patch_size // 2 + 2, (255, 40, 40), 2)

            axes[ax_i].imshow(canvas)
            ratio = pure_patches.shape[0] / max(vom_patches.shape[0], 1)
            n_overlap = len(overlap_anchor_ids) if len(vom_patches) > 0 else 0
            axes[ax_i].set_title(
                f"Anchor {k} | vom={vom_patches.shape[0]} pure={pure_patches.shape[0]}"
                f" ({ratio:.2f})\noverlap anchors={n_overlap}",
                fontsize=8, fontproperties=font_prop
            )
            axes[ax_i].axis('off')

        for ax in axes[n:]:
            ax.axis('off')

        fig.suptitle("vom(파랑) / pure(초록) 오버레이 — 시드(빨강)가 vom 안에 있어야 함",
                     fontsize=11, fontproperties=font_prop)
        self._show_or_save(fig, filename)

    # ─────────────────────────────────────────────────────────────
    # 2. 히트맵 유사도 행렬 + 병합 그룹
    # ─────────────────────────────────────────────────────────────

    def show_heatmap_sim(self, ctx: dict,
                         labels: Optional[np.ndarray] = None,
                         n_comp: int = 0,
                         th_heatmap: float = 0.85,
                         filename: str = "debug_heatmap_sim.png"):
        """
        K×K 히트맵 유사도 행렬 시각화.

        검증 포인트:
          - 같은 객체를 가리키는 앵커끼리 높은 유사도(밝음) 클러스터 형성 여부
          - labels(그룹 색) 경계와 높은 유사도 영역이 일치하는지
          - th_heatmap 선(빨강 수평/수직 점선)이 올바른 위치에 있는지
        """
        heatmap_sim = ctx['heatmap_sim'].cpu().numpy()  # [K, K]
        K = ctx['K']

        # 그룹 순서로 행/열 재정렬
        if labels is not None:
            order = np.argsort(labels)
            heatmap_sim = heatmap_sim[order][:, order]
            sorted_labels = labels[order]
        else:
            sorted_labels = None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 왼쪽: 히트맵
        im = axes[0].imshow(heatmap_sim, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f"heatmap_sim [K={K}×K={K}]  (th={th_heatmap})", fontproperties=font_prop)
        plt.colorbar(im, ax=axes[0])

        # th_heatmap 경계선
        axes[0].contour(heatmap_sim >= th_heatmap, levels=[0.5],
                        colors='red', linewidths=0.8, linestyles='--')

        # 그룹 경계 표시
        if sorted_labels is not None:
            boundaries = np.where(np.diff(sorted_labels))[0] + 1
            for b in boundaries:
                axes[0].axhline(b - 0.5, color='white', lw=1.2)
                axes[0].axvline(b - 0.5, color='white', lw=1.2)

        # 오른쪽: 그룹별 히트맵 유사도 분포
        if labels is not None and n_comp > 0:
            intra, inter = [], []
            for i in range(K):
                for j in range(i + 1, K):
                    val = ctx['heatmap_sim'][i, j].item()
                    if labels[i] == labels[j]:
                        intra.append(val)
                    else:
                        inter.append(val)
            bins = np.linspace(0, 1, 40)
            axes[1].hist(intra, bins=bins, alpha=0.7, label=f'intra-group ({len(intra)})',
                         color='steelblue')
            axes[1].hist(inter, bins=bins, alpha=0.7, label=f'inter-group ({len(inter)})',
                         color='salmon')
            axes[1].axvline(th_heatmap, color='red', lw=1.5, linestyle='--',
                            label=f'th={th_heatmap}')
            axes[1].set_xlabel('heatmap cosine similarity')
            axes[1].set_ylabel('count')
            axes[1].set_title(f'그룹 내/간 유사도 분포  (n_comp={n_comp})', fontproperties=font_prop)
            axes[1].legend()
        else:
            axes[1].axis('off')

        self._show_or_save(fig, filename)

    # ─────────────────────────────────────────────────────────────
    # 3. 앵커 병합 before / after
    # ─────────────────────────────────────────────────────────────

    def show_merge_result(self, img_bgr: np.ndarray,
                          centroids_before: torch.Tensor,
                          centroids_after: torch.Tensor,
                          labels: Optional[np.ndarray] = None,
                          filename: str = "debug_merge.png"):
        """
        병합 전(K) → 병합 후(M) 앵커 시드 위치 비교.

        검증 포인트:
          - 같은 객체 위에 있던 여러 시드가 하나로 합쳐졌는가?
          - 병합 후 앵커 수 M << K 인가? (아니면 th_heatmap이 너무 높음)
          - 동일 그룹(같은 색)의 시드들이 공간적으로 가까운가?
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        K = centroids_before.shape[0]
        M = centroids_after.shape[0]
        palette = make_distinct_colors(max(K, 1))

        # 왼쪽: before
        ax = axes[0]
        ax.imshow(img_rgb)
        for k in range(K):
            idx = (labels[k] if labels is not None else k) % len(palette)
            r, g, b = palette[idx]
            color = (r/255, g/255, b/255)
            x, y = patch_to_pixel(int(centroids_before[k]), self.grid_shape, self.patch_size)
            ax.scatter(x, y, s=80, color=color, edgecolors='white', linewidths=0.8, zorder=3)
            if labels is not None:
                ax.annotate(str(labels[k]), (x, y), fontsize=6,
                            color='white', ha='center', va='center')
        ax.set_title(f"병합 전  K={K}개 앵커", fontproperties=font_prop)
        ax.axis('off')

        # 오른쪽: after
        ax = axes[1]
        ax.imshow(img_rgb)
        palette2 = make_distinct_colors(max(M, 1))
        for m in range(M):
            r, g, b = palette2[m]
            color = (r/255, g/255, b/255)
            x, y = patch_to_pixel(int(centroids_after[m]), self.grid_shape, self.patch_size)
            ax.scatter(x, y, s=120, color=color, edgecolors='white', linewidths=1.2,
                       marker='*', zorder=3)
        ax.set_title(f"병합 후  M={M}개 앵커  (축소율 {K}→{M})", fontproperties=font_prop)
        ax.axis('off')

        fig.suptitle("앵커 병합 결과 (같은 색 = 같은 그룹)", fontsize=11, fontproperties=font_prop)
        self._show_or_save(fig, filename)

    # ─────────────────────────────────────────────────────────────
    # 4. P3 단편화 필터 결과
    # ─────────────────────────────────────────────────────────────

    def show_p3_filter(self, img_bgr: np.ndarray,
                       group_vom: torch.Tensor,
                       keep: torch.Tensor,
                       centroids: torch.Tensor,
                       filename: str = "debug_p3_filter.png"):
        """
        P3 필터 결과: 유지(초록)와 제거(빨강) 앵커를 각각 표시.

        검증 포인트:
          - 제거된 앵커(빨강)의 vom이 실제로 두 곳으로 분리돼 있는가?
          - 유지된 앵커(초록)의 vom이 하나의 연속 영역인가?
          - 실제 객체가 있는 앵커가 잘못 제거되지는 않는가?
        """
        vom_np   = group_vom.cpu().numpy()   # [M, N] bool
        keep_np  = keep.cpu().numpy()        # [M] bool
        cent_np  = centroids.cpu().numpy()   # [M]

        M = vom_np.shape[0]
        kept_idx    = [i for i in range(M) if keep_np[i]]
        removed_idx = [i for i in range(M) if not keep_np[i]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for ax, group, title, color in [
            (axes[0], kept_idx,    f"유지 ({len(kept_idx)}개)", (30, 220, 60)),
            (axes[1], removed_idx, f"제거(P3) ({len(removed_idx)}개)", (255, 50, 50)),
        ]:
            canvas = img_rgb.copy()
            for k in group:
                patches = np.where(vom_np[k])[0]
                canvas = draw_patch_mask(canvas, patches, self.grid_shape,
                                         self.patch_size, color=color, alpha=0.35)
                sx, sy = patch_to_pixel(int(cent_np[k]), self.grid_shape, self.patch_size)
                cv2.circle(canvas, (sx, sy), 5, (255, 255, 255), -1)
            ax.imshow(canvas)
            ax.set_title(title, fontproperties=font_prop)
            ax.axis('off')

        fig.suptitle("P3 단편화 필터 — 제거된 앵커(빨강)의 vom이 분리돼 있어야 함",
                     fontsize=11, fontproperties=font_prop)
        self._show_or_save(fig, filename)

    # ─────────────────────────────────────────────────────────────
    # 5. Cross-frame 필터 결과 (P1/P2)
    # ─────────────────────────────────────────────────────────────

    def show_cross_frame_filter(self,
                                img1_bgr: np.ndarray,
                                img2_bgr: np.ndarray,
                                centroids_mem: torch.Tensor,
                                valid_mask: torch.Tensor,
                                pure_area_cur: torch.Tensor,
                                vom_cur: torch.Tensor,
                                min_pure_response: int = 3,
                                img2_frame_id: Optional[str] = None,
                                filename: str = "debug_cross_frame_filter.png"):
        """
        메모리 앵커(img2) → 현재 프레임(img1) 대응 시각화.

        레이아웃:
          왼쪽  (img2): 메모리 앵커 위치. 유효=앵커색 원, 배제=회색 X
          가운데 (img1): 유효 앵커의 vom 응답 영역. 앵커색 테두리로 표시
          오른쪽: pure_area_cur 막대그래프
        """
        vom_np   = vom_cur.cpu().numpy()         # [K, N] bool
        valid_np = valid_mask.cpu().numpy()      # [K] bool
        pure_np  = pure_area_cur.cpu().numpy()   # [K]
        cents_np = centroids_mem.cpu().numpy()   # [K]
        K = vom_np.shape[0]

        img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

        palette = make_distinct_colors(max(K, 1))

        def anchor_color_i(k):
            return palette[k % len(palette)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # ── 왼쪽: img2 메모리 앵커 위치 ──────────────────────────────
        canvas2 = img2_rgb.copy()
        for k in range(K):
            sx, sy = patch_to_pixel(int(cents_np[k]), self.grid_shape, self.patch_size)
            if valid_np[k]:
                color = anchor_color_i(k)
                cv2.circle(canvas2, (sx, sy), self.patch_size // 2 + 3, color, -1)
                cv2.circle(canvas2, (sx, sy), self.patch_size // 2 + 3, (255, 255, 255), 1)
                cv2.putText(canvas2, str(k), (sx - 4, sy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                # 배제 앵커: 회색 X
                d = self.patch_size // 2
                cv2.line(canvas2, (sx - d, sy - d), (sx + d, sy + d), (160, 160, 160), 2)
                cv2.line(canvas2, (sx + d, sy - d), (sx - d, sy + d), (160, 160, 160), 2)

        axes[0].imshow(canvas2)
        img2_label = f"img2 [{img2_frame_id}]" if img2_frame_id else "img2"
        axes[0].set_title(
            f"{img2_label} 메모리 앵커  유효={valid_np.sum()}/{K}",
            fontproperties=font_prop)
        axes[0].axis('off')

        # ── 가운데: img1 현재 프레임 vom 응답 (유효 앵커만, 테두리) ──
        canvas1 = img1_rgb.copy()
        H_p, W_p = self.grid_shape
        for k in range(K):
            if not valid_np[k]:
                continue
            color = anchor_color_i(k)
            patches = np.where(vom_np[k])[0]
            # 반투명 fill
            canvas1 = draw_patch_mask(canvas1, patches, self.grid_shape,
                                      self.patch_size, color=color, alpha=0.25)
            # 패치 테두리
            for idx in patches:
                r, c = int(idx) // W_p, int(idx) % W_p
                x0, y0 = c * self.patch_size, r * self.patch_size
                x1, y1 = x0 + self.patch_size, y0 + self.patch_size
                cv2.rectangle(canvas1, (x0, y0), (x1, y1), color, 1)

        axes[1].imshow(canvas1)
        axes[1].set_title("img1 현재 프레임 — 유효 앵커 vom 응답",
                          fontproperties=font_prop)
        axes[1].axis('off')

        # ── 오른쪽: pure_area_cur 막대그래프 ─────────────────────────
        bar_colors = [
            tuple(c / 255 for c in anchor_color_i(k)) if valid_np[k] else (0.6, 0.6, 0.6)
            for k in range(K)
        ]
        axes[2].bar(range(K), pure_np, color=bar_colors)
        axes[2].axhline(min_pure_response, color='red', linestyle='--',
                        label=f'min_pure={min_pure_response}')
        axes[2].set_xlabel('anchor index')
        axes[2].set_ylabel('pure_area_cur')
        axes[2].set_title('메모리 앵커별 현재 프레임 pure 반응', fontproperties=font_prop)
        axes[2].legend()

        fig.suptitle("Cross-frame 필터 — 색상 = 앵커 ID 공유 (유효: 원/테두리, 배제: 회색 X)",
                     fontsize=11, fontproperties=font_prop)
        self._show_or_save(fig, filename)

    # ─────────────────────────────────────────────────────────────
    # 6. 전체 파이프라인 한눈에 보기
    # ─────────────────────────────────────────────────────────────

    def show_pipeline_summary(self, img_bgr: np.ndarray,
                              ctx_before: dict,
                              ctx_after: dict,
                              keep_p3: torch.Tensor,
                              filename: str = "debug_pipeline_summary.png"):
        """
        단일 프레임 파이프라인 4단계 요약:
          (1) 원본 앵커 시드 K개
          (2) 히트맵 병합 후 M개
          (3) P3 필터 후 남은 앵커
          (4) pure 기반 평균 벡터 커버리지

        검증 포인트:
          - 각 단계에서 앵커 수가 합리적으로 줄어드는가?
          - P3가 너무 많이 제거하지는 않는가?
        """
        from scipy.sparse.csgraph import connected_components as _cc
        from scipy.sparse import csr_matrix as _csr

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        # Step 1: 원본 앵커
        canvas = img_rgb.copy()
        for k in range(ctx_before['K']):
            x, y = patch_to_pixel(int(ctx_before['centroids'][k]),
                                  self.grid_shape, self.patch_size)
            cv2.circle(canvas, (x, y), 6, (255, 80, 80), -1)
        axes[0].imshow(canvas)
        axes[0].set_title(f"① 원본 앵커  K={ctx_before['K']}", fontproperties=font_prop)
        axes[0].axis('off')

        # Step 2: 병합 후 (ctx_after 기준)
        canvas = img_rgb.copy()
        for k in range(ctx_after['K']):
            x, y = patch_to_pixel(int(ctx_after['centroids'][k]),
                                  self.grid_shape, self.patch_size)
            cv2.circle(canvas, (x, y), 7, (80, 150, 255), -1)
        axes[1].imshow(canvas)
        axes[1].set_title(f"② 병합 후  M={ctx_after['K']}", fontproperties=font_prop)
        axes[1].axis('off')

        # Step 3: P3 필터 후
        canvas = img_rgb.copy()
        keep_np = keep_p3.cpu().numpy()
        cents_after = ctx_after['centroids'].cpu().numpy()
        for k in range(ctx_after['K']):
            x, y = patch_to_pixel(int(cents_after[k]), self.grid_shape, self.patch_size)
            color = (60, 200, 60) if keep_np[k] else (200, 60, 60)
            cv2.circle(canvas, (x, y), 7, color, -1)
        axes[2].imshow(canvas)
        axes[2].set_title(f"③ P3 필터 후  {keep_np.sum()}/{ctx_after['K']}개", fontproperties=font_prop)
        axes[2].axis('off')

        # Step 4: pure 커버리지
        canvas = img_rgb.copy()
        vom_np  = ctx_after['vom'].cpu().numpy()
        pure_np = ctx_after['pure'].cpu().numpy()
        kept_indices = [k for k in range(ctx_after['K']) if keep_np[k]]
        palette = make_distinct_colors(max(len(kept_indices), 1))
        for i, k in enumerate(kept_indices):
            color_i = palette[i]
            patches = np.where(pure_np[k])[0]
            canvas = draw_patch_mask(canvas, patches, self.grid_shape,
                                     self.patch_size, color=color_i, alpha=0.50)
        axes[3].imshow(canvas)
        axes[3].set_title(f"④ pure 커버리지 (avg_vec 계산 영역)", fontproperties=font_prop)
        axes[3].axis('off')

        fig.suptitle("단일 프레임 파이프라인 요약", fontsize=12, fontproperties=font_prop)
        self._show_or_save(fig, filename)


# ─────────────────────────────────────────────────────────────────
# 빠른 검증 헬퍼 (서버 코드에서 바로 호출 가능)
# ─────────────────────────────────────────────────────────────────

def quick_check_vom(ctx: dict) -> dict:
    """
    vom/pure 계산 결과의 기초 통계를 출력.
    서버 루프 안에서 한 줄로 호출해 이상 여부 즉시 확인.

    반환:
        stats dict:
            seed_in_vom_rate  : 자기 시드가 vom에 포함된 앵커 비율 (1.0이어야 정상)
            mean_vom_size     : 앵커당 평균 vom 패치 수
            mean_pure_size    : 앵커당 평균 pure 패치 수
            zero_pure_count   : pure가 0인 앵커 수 (avg_vec 계산 불가 위험)
            background_risk   : vom > N*0.3인 앵커 수 (배경 의심)
    """
    vom  = ctx['vom']   # [K, N]
    pure = ctx['pure']  # [K, N]
    K, N = vom.shape
    cents = ctx['centroids']

    seed_in_vom = vom[torch.arange(K, device=vom.device), cents.long()]
    vom_sizes   = vom.sum(dim=1)
    pure_sizes  = pure.sum(dim=1)

    stats = {
        'K': K,
        'N': N,
        'seed_in_vom_rate' : seed_in_vom.float().mean().item(),
        'mean_vom_size'    : vom_sizes.float().mean().item(),
        'mean_pure_size'   : pure_sizes.float().mean().item(),
        'zero_pure_count'  : (pure_sizes == 0).sum().item(),
        'background_risk'  : (vom_sizes > N * 0.3).sum().item(),
    }

    print("── quick_check_vom ────────────────────────")
    print(f"  앵커 수        K = {stats['K']}")
    print(f"  seed_in_vom    = {stats['seed_in_vom_rate']:.3f}  (1.0이어야 정상)")
    print(f"  avg vom size   = {stats['mean_vom_size']:.1f} 패치")
    print(f"  avg pure size  = {stats['mean_pure_size']:.1f} 패치")
    print(f"  zero pure      = {stats['zero_pure_count']}개  (avg_vec 위험)")
    print(f"  background?    = {stats['background_risk']}개  (vom > 30% N)")
    print("───────────────────────────────────────────")

    return stats


def quick_check_cross_frame(valid_mask: torch.Tensor,
                             pure_area_cur: torch.Tensor,
                             min_pure: int = 3) -> None:
    """
    cross-frame 필터 결과를 한 줄로 요약 출력.
    """
    K = valid_mask.shape[0]
    n_valid = valid_mask.sum().item()
    print(f"── cross_frame_filter ─ K={K} → valid={n_valid} "
          f"(배제={K - n_valid}, min_pure={min_pure})")
    print(f"   pure_area 분포: min={pure_area_cur.min().item()} "
          f"max={pure_area_cur.max().item()} "
          f"mean={pure_area_cur.float().mean().item():.1f}")

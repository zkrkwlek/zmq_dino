"""
dino/patchcluster_v2.py
────────────────────────────────────────────────────────────────
DinoSemanticObjectExtractorV2
  - DinoSemanticObjectExtractor 를 상속
  - 신규 메서드 추가:
      compute_anchor_patch_context  — vom/pure/heatmap_sim 통합 계산
      merge_anchors_heatmap         — Concept A: 히트맵 유사도 기반 병합
      merge_anchors_shared_patch    — Concept B: 공유 패치 비율 기반 병합
      _filter_p3_fragmentation      — P3: 공간 단편화 앵커 배제
      _apply_group_and_recompute    — 병합 후처리 (group_pure + avg_vec)
      filter_memory_anchors_cross_frame — P1/P2: 크로스 프레임 앵커 필터

파이프라인 통일 API (alias):
    generate_seeds        ← sample_patch (부모)
    group_anchors         ← merge_anchors_heatmap
    compute_anchor_response ← _apply_group_and_recompute
    project_memory_to_frame ← filter_memory_anchors_cross_frame
    recompute_anchor_vecs ← extract_sample_neighborhood_average_pool
    filter_anchors        ← filter_by_quality
    detect_multiresponse  ← detect_mixed_boundary_patches_by_counting

타이밍:
    objpatcher = DinoSemanticObjectExtractorV2(timing=True)
    objpatcher.timing = False  # 런타임 토글

사용법:
    from dino.patchcluster_v2 import DinoSemanticObjectExtractorV2
    objpatcher = DinoSemanticObjectExtractorV2()
────────────────────────────────────────────────────────────────
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dino.patchcluster import DinoSemanticObjectExtractor


class DinoSemanticObjectExtractorV2(DinoSemanticObjectExtractor):
    """
    기존 DinoSemanticObjectExtractor 에 신규 앵커 정제 메서드를 추가한 버전.

    단일 프레임 파이프라인 (통일 API):
        generate_seeds          (= sample_patch)
        → compute_anchor_patch_context
        → group_anchors         (= merge_anchors_heatmap)
        → compute_anchor_response (= _apply_group_and_recompute)  ← P3 포함

    다중 프레임 파이프라인 (통일 API):
        project_memory_to_frame (= filter_memory_anchors_cross_frame)
        → recompute_anchor_vecs (= extract_sample_neighborhood_average_pool)

    공통 후처리:
        filter_anchors          (= filter_by_quality)
        detect_multiresponse    (= detect_mixed_boundary_patches_by_counting)
    """

    def __init__(self, *args, timing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing = timing  # True 이면 각 서브스텝 처리 시간 출력

    # ──────────────────────────────────────────────────────────────
    # 타이밍 헬퍼
    # ──────────────────────────────────────────────────────────────

    def _tlog(self, label: str, t0: float) -> float:
        """타이밍 로그. self.timing=True 일 때만 출력. 현재 시간 반환."""
        t1 = time.time()
        if self.timing:
            print(f"  [{label:<40s}] {(t1 - t0) * 1000:6.1f}ms")
        return t1

    # ──────────────────────────────────────────────────────────────
    # 1. compute_anchor_patch_context
    # ──────────────────────────────────────────────────────────────

    def compute_anchor_patch_context(self, centroids, feat,
                                     th_sim=0.60, th_heat=0.65, th_margin=0.12,
                                     use_heatmap_vom=False):
        """
        앵커별 패치 컨텍스트를 한 번에 계산.

        valid_overlap_mask (vom) [K, N] bool
            — 1등 앵커와 마진 < th_margin 인 패치. XFeat 귀속에 사용.
        pure_affinity (pure) [K, N] bool
            — 단독 점유 패치 (oc == 1). avg_vec 계산에만 사용.

        핵심 버그 픽스 — 자기 시드 마스킹:
            시드는 자기 자신에 sim≈1.0 → max_vals 항상 1.0
            → 다른 앵커의 margin이 항상 th_margin 초과 → vom 전체 False
            → 수정: max_vals 계산 시 자기 시드 위치를 0으로 마스킹

        Returns:
            ctx dict:
                centroids   [K]
                sim_matrix  [K, N] float32
                vom         [K, N] bool
                pure        [K, N] bool
                oc          [N]    long      — overlap counts
                H_matrix    [K, N] float32   — 노이즈 압착 히트맵
                heatmap_sim [K, K] float32   — 앵커 간 히트맵 유사도
                K, N        int
        """
        t0 = time.time()
        K = centroids.shape[0]
        N = feat.shape[0]
        device = feat.device

        _ekn = torch.zeros((0, N), dtype=torch.bool, device=device)
        if K == 0:
            return dict(
                centroids=centroids,
                sim_matrix=torch.zeros((0, N), device=device),
                vom=_ekn, pure=_ekn,
                oc=torch.zeros(N, device=device),
                H_matrix=torch.zeros((0, N), device=device),
                heatmap_sim=torch.zeros((0, 0), device=device),
                K=0, N=N,
            )

        seed_feats = feat[centroids.long()]              # [K, D]
        sim_matrix = torch.mm(seed_feats, feat.t())      # [K, N]
        t0 = self._tlog("ctx: sim_matrix mm", t0)

        sim_thr    = torch.where(sim_matrix >= th_sim,
                                 sim_matrix,
                                 torch.zeros_like(sim_matrix))

        # 자기 시드 마스킹
        self_mask = torch.zeros(K, N, dtype=torch.bool, device=device)
        self_mask[torch.arange(K, device=device), centroids.long()] = True
        sim_thr_for_max = sim_thr.masked_fill(self_mask, 0.0)
        max_vals = sim_thr_for_max.max(dim=0, keepdim=True).values  # [1, N]

        if use_heatmap_vom:
            vom = sim_matrix >= th_heat
        else:
            margins = max_vals - sim_thr
            vom  = (sim_thr > 0) & (margins < th_margin)
        oc   = vom.sum(dim=0)
        pure = vom & (oc.unsqueeze(0) == 1)
        t0 = self._tlog("ctx: vom/pure compute", t0)

        H_matrix   = torch.where(sim_matrix >= th_heat,
                                  sim_matrix,
                                  torch.zeros_like(sim_matrix))
        H_norm      = F.normalize(H_matrix, p=2, dim=1)
        heatmap_sim = torch.mm(H_norm, H_norm.t())        # [K, K]
        self._tlog("ctx: heatmap_sim", t0)

        return dict(
            centroids=centroids,
            sim_matrix=sim_matrix,
            vom=vom, pure=pure, oc=oc,
            H_matrix=H_matrix,
            heatmap_sim=heatmap_sim,
            K=K, N=N,
        )

    # ──────────────────────────────────────────────────────────────
    # 2. merge_anchors_heatmap  (Concept A)
    # ──────────────────────────────────────────────────────────────

    def merge_anchors_heatmap(self, ctx, th_heatmap=0.85):
        """
        Concept A — 히트맵 패턴이 유사한 앵커 병합.
        같은 객체에 반응하는 앵커 K개 → 그룹 M개 (M ≤ K).

        Returns:
            labels : [K] numpy int32  — 그룹 ID
            n_comp : int              — 총 그룹 수
        """
        t0 = time.time()
        K = ctx['K']
        if K == 0:
            return np.array([], dtype=np.int32), 0

        adj = (ctx['heatmap_sim'].cpu().numpy() >= th_heatmap).astype(np.int32)
        n_comp, labels = connected_components(
            csr_matrix(adj), directed=False, connection='weak', return_labels=True)
        self._tlog("group_anchors: connected_components", t0)
        return labels, n_comp

    # ──────────────────────────────────────────────────────────────
    # 3. merge_anchors_shared_patch  (Concept B)
    # ──────────────────────────────────────────────────────────────

    def merge_anchors_shared_patch(self, ctx, th_shared_ratio=0.5):
        """
        Concept B — 공유 패치 비율 + 시드 포함 관계로 앵커 병합.

        병합 조건 (OR):
          - shared_ratio[i,j] >= th_shared_ratio
          - seed_j ∈ vom_i
          - seed_i ∈ vom_j

        Returns:
            labels : [K] numpy int32
            n_comp : int
        """
        t0 = time.time()
        K = ctx['K']
        if K == 0:
            return np.array([], dtype=np.int32), 0

        vom       = ctx['vom'].float()
        centroids = ctx['centroids']

        shared      = torch.mm(vom, vom.t())
        own_counts  = vom.sum(dim=1).clamp(min=1.0)
        min_counts  = torch.min(own_counts.unsqueeze(1),
                                own_counts.unsqueeze(0))
        shared_ratio = shared / min_counts

        seed_in_vom  = vom[:, centroids.long()]
        adj_mask = (shared_ratio >= th_shared_ratio) \
                   | seed_in_vom.bool() \
                   | seed_in_vom.t().bool()
        adj_np = adj_mask.cpu().numpy().astype(np.int32)
        np.fill_diagonal(adj_np, 1)

        n_comp, labels = connected_components(
            csr_matrix(adj_np), directed=False, connection='weak', return_labels=True)
        self._tlog("group_anchors(B): connected_components", t0)
        return labels, n_comp

    # ──────────────────────────────────────────────────────────────
    # 4. filter_by_purity
    # ──────────────────────────────────────────────────────────────

    def filter_by_purity(self, ctx, min_pure: int = 1, min_ratio: float = 0.0):
        """
        pure/vom 비율 기반 앵커 필터.

        제거 조건 (OR):
          - pure == 0              : 독점 영역 없음
          - pure/vom < min_ratio   : vom 대비 pure 비율이 너무 낮음

        Returns:
            keep : [K] bool tensor
        """
        vom_sizes  = ctx['vom'].sum(dim=1).float()
        pure_sizes = ctx['pure'].sum(dim=1).float()

        keep = pure_sizes >= min_pure
        if min_ratio > 0.0:
            ratio = pure_sizes / vom_sizes.clamp(min=1.0)
            keep = keep & (ratio >= min_ratio)
        return keep

    # ──────────────────────────────────────────────────────────────
    # 5. _compute_anchor_quality  /  filter_by_quality
    # ──────────────────────────────────────────────────────────────

    def _compute_anchor_quality(self, vom, pure, grid_shape=None):
        """
        앵커별 품질 메트릭 계산.

        Returns:
            vom_ratio      : [K] float  vom 점유율 (0~1)
            pure_vom_ratio : [K] float  pure/vom 비율 (0~1)
            spatial_std    : [K] float  vom 공간 표준편차 (패치 단위, grid_shape 없으면 0)
        """
        K, N = vom.shape
        vom_sizes  = vom.sum(dim=1).float()
        pure_sizes = pure.sum(dim=1).float()

        vom_ratio      = vom_sizes / N
        pure_vom_ratio = pure_sizes / vom_sizes.clamp(min=1.0)

        spatial_std = torch.zeros(K, device=vom.device)
        if grid_shape is not None:
            H_p, W_p = grid_shape
            vom_np = vom.cpu().numpy()
            for k in range(K):
                idx = np.where(vom_np[k])[0]
                if len(idx) < 2:
                    continue
                rows = (idx // W_p).astype(float)
                cols = (idx % W_p).astype(float)
                spatial_std[k] = float(np.std(rows) + np.std(cols))

        return vom_ratio, pure_vom_ratio, spatial_std

    def filter_by_quality(self, vom, pure,
                          min_pure_response: int = 1,
                          min_pure_vom_ratio: float = 0.10,
                          max_spatial_std: float = 8.0,
                          grid_shape=None):
        """
        메모리 뱅크 / 단일 프레임 앵커 품질 필터 (3가지 기준).

        제거 조건 (OR):
          ① pure_area < min_pure_response       : 독점 패치 없음
          ② pure_vom_ratio < min_pure_vom_ratio : 혼합 신호 앵커
          ③ spatial_std >= max_spatial_std      : vom 공간 발산 (배경/바닥 앵커)

        Note:
          vom_ratio 상한은 사용하지 않음.
          배경 앵커는 크로스 프레임 발산(spatial_std)으로 감지.

        Returns:
            keep : [K] bool tensor
        """
        t0 = time.time()
        _, pure_vom_ratio, spatial_std = self._compute_anchor_quality(
            vom, pure, grid_shape)

        pure_sizes = pure.sum(dim=1).float()

        keep = pure_sizes >= min_pure_response                   # ①
        keep = keep & (pure_vom_ratio >= min_pure_vom_ratio)     # ②
        if grid_shape is not None and max_spatial_std > 0:
            keep = keep & (spatial_std < max_spatial_std)        # ③

        self._tlog("filter_anchors: quality check", t0)
        return keep

    # ──────────────────────────────────────────────────────────────
    # 6. _filter_p3_fragmentation
    # ──────────────────────────────────────────────────────────────

    def _filter_p3_fragmentation(self, vom, grid_shape, max_components=1):
        """
        P3 필터 — 공간 단편화 앵커 배제.
        vom 패치가 2개 이상 disconnected region을 형성하면 제거.

        Returns:
            keep : [K] bool tensor
        """
        t0 = time.time()
        H_p, W_p = grid_shape
        K = vom.shape[0]
        keep   = torch.ones(K, dtype=torch.bool, device=vom.device)
        vom_np = vom.cpu().numpy()

        for k in range(K):
            patch_idx = np.where(vom_np[k])[0]
            if len(patch_idx) < 2:
                continue

            patch_set = set(int(p) for p in patch_idx)
            visited   = set()
            n_comp_k  = 0

            for start in patch_idx:
                start = int(start)
                if start in visited:
                    continue
                n_comp_k += 1
                if n_comp_k > max_components:
                    keep[k] = False
                    break
                queue = [start]
                while queue:
                    node = queue.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    r, c = node // W_p, node % W_p
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H_p and 0 <= nc < W_p:
                            nb = nr * W_p + nc
                            if nb in patch_set and nb not in visited:
                                queue.append(nb)

        self._tlog("P3: fragmentation filter", t0)
        return keep

    # ──────────────────────────────────────────────────────────────
    # 7. merge_to_ctx  /  _apply_group_and_recompute
    # ──────────────────────────────────────────────────────────────

    def merge_to_ctx(self, centroids, group_labels, n_comp, feat):
        """
        히트맵 병합 후 새 시드 확정 + ctx 재계산까지만.
        P3 필터 / avg_vec 계산은 포함하지 않음.

        Returns:
            new_centroids : [M] long
            new_ctx       : dict
        """
        t0 = time.time()
        device = feat.device
        N = feat.shape[0]

        new_centroids = self.extract_new_group_centroids_vectorized(
            centroids, group_labels, n_comp)

        if new_centroids.shape[0] == 0:
            empty = torch.zeros((0, N), dtype=torch.bool, device=device)
            return new_centroids, dict(
                centroids=new_centroids, vom=empty, pure=empty,
                oc=torch.zeros(N, device=device),
                sim_matrix=torch.zeros((0, N), device=device),
                H_matrix=torch.zeros((0, N), device=device),
                heatmap_sim=torch.zeros((0, 0), device=device),
                K=0, N=N)

        new_ctx = self.compute_anchor_patch_context(new_centroids, feat)
        self._tlog("merge_to_ctx: seed + ctx", t0)
        return new_centroids, new_ctx

    def _apply_group_and_recompute(self, centroids, group_labels, n_comp,
                                   ctx, feat, x_cat,
                                   attn=None, grid_shape=None):
        """
        merge_anchors_* 후 공통 후처리:
          1. 그룹별 대표 시드 선출 + ctx 재계산 (merge_to_ctx)
          2. P3 필터 (grid_shape 제공 시)
          3. pure 패치 기반 avg_vec 계산

        Returns:
            new_centroids : [M] long
            group_vom     : [M, N] bool
            group_pure    : [M, N] bool
            avg_vec       : [M, D] float32
            new_ctx       : dict
        """
        t0_total = time.time()
        device = feat.device
        N = feat.shape[0]
        _, D, _, _ = x_cat.shape

        new_centroids, new_ctx = self.merge_to_ctx(centroids, group_labels, n_comp, feat)

        if new_centroids.shape[0] == 0:
            empty = torch.zeros((0, N), dtype=torch.bool, device=device)
            return (new_centroids, empty, empty,
                    torch.zeros((0, D), device=device), new_ctx)

        group_vom  = new_ctx['vom']
        group_pure = new_ctx['pure']

        # P3 필터
        if grid_shape is not None and new_centroids.shape[0] > 0:
            keep = self._filter_p3_fragmentation(group_vom, grid_shape)
            if keep.any() and not keep.all():
                new_centroids = new_centroids[keep]
                new_ctx    = self.compute_anchor_patch_context(new_centroids, feat)
                group_vom  = new_ctx['vom']
                group_pure = new_ctx['pure']

        t0 = time.time()
        avg_vec, _ = self.extract_sample_neighborhood_average_pool(
            x_cat, group_pure.float(), attn)
        self._tlog("compute_anchor_response: avg_vec pool", t0)
        self._tlog("compute_anchor_response: total", t0_total)

        return new_centroids, group_vom, group_pure, avg_vec, new_ctx

    def _apply_keep_mask(self, new_ctx, new_centroids, keep, x_cat, feat, attn):
        """
        이미 계산된 keep 마스크를 적용하고 avg_vec 재계산.
        (filter_by_quality 결과를 받아 ctx 정리할 때 사용)
        """
        if not keep.any():
            device = feat.device
            N = feat.shape[0]
            D = x_cat.shape[1]
            empty = torch.zeros((0, N), dtype=torch.bool, device=device)
            return (new_centroids[:0], empty, empty,
                    torch.zeros((0, D), device=device), new_ctx)

        if not keep.all():
            new_centroids = new_centroids[keep]
            new_ctx = self.compute_anchor_patch_context(new_centroids, feat)

        group_vom  = new_ctx['vom']
        group_pure = new_ctx['pure']

        avg_vec, _ = self.extract_sample_neighborhood_average_pool(
            x_cat, group_pure.float(), attn)

        return new_centroids, group_vom, group_pure, avg_vec, new_ctx

    # ──────────────────────────────────────────────────────────────
    # 8. filter_memory_anchors_cross_frame
    # ──────────────────────────────────────────────────────────────

    def filter_memory_anchors_cross_frame(self, avg_vec_mem, feat_cur,
                                          centroids_mem=None,
                                          min_pure_response=3,
                                          min_pure_vom_ratio=0.10,
                                          max_spatial_std=8.0,
                                          grid_shape=None,
                                          th_sim=0.60, th_margin=0.12,
                                          use_raw_feat=False,
                                          use_heatmap_vom=False,
                                          th_heat=0.65):
        """
        다중 프레임 앵커 정제 — 품질 기준 적용.

        Args:
            avg_vec_mem        : [K, D]  메모리 앵커 avg_vec (L2 정규화)
            feat_cur           : [N, D]  현재 프레임 패치 피처
            min_pure_response  : int     ① 최소 pure 패치 수 (기본 3)
            min_pure_vom_ratio : float   ② pure/vom 비율 하한 (기본 0.10)
            max_spatial_std    : float   ③ vom 공간 std 상한 (기본 8.0)
            grid_shape         : (H_p, W_p)

        Returns:
            valid_mask    : [K] bool
            pure_area_cur : [K] long
            vom           : [K, N] bool
            pure          : [K, N] bool
        """
        t0_total = time.time()
        K = avg_vec_mem.shape[0]
        N = feat_cur.shape[0]
        device = feat_cur.device

        if K == 0:
            empty = torch.zeros((0, N), dtype=torch.bool, device=device)
            return (torch.ones(0, dtype=torch.bool, device=device),
                    torch.zeros(0, device=device),
                    empty, empty)

        t0 = time.time()
        sim_matrix = torch.mm(avg_vec_mem, feat_cur.t())   # [K, N]
        self._tlog("project_memory: sim_matrix mm", t0)

        t0 = time.time()
        if use_heatmap_vom:
            vom = sim_matrix >= th_heat
        else:
            sim_thr  = torch.where(sim_matrix >= th_sim,
                                   sim_matrix,
                                   torch.zeros_like(sim_matrix))
            max_vals = sim_thr.max(dim=0, keepdim=True).values
            margins  = max_vals - sim_thr
            vom      = (sim_thr > 0) & (margins < th_margin)

        oc   = vom.sum(dim=0)
        pure = vom & (oc.unsqueeze(0) == 1)
        self._tlog("project_memory: vom/pure compute", t0)

        valid_mask = self.filter_by_quality(
            vom, pure,
            min_pure_response=min_pure_response,
            min_pure_vom_ratio=min_pure_vom_ratio,
            max_spatial_std=max_spatial_std,
            grid_shape=grid_shape,
        )

        self._tlog("project_memory: total", t0_total)
        pure_area_cur = pure.sum(dim=1)
        return valid_mask, pure_area_cur, vom, pure

    # ──────────────────────────────────────────────────────────────
    # 파이프라인 통일 API — alias
    # 서버 코드가 아래 이름으로 호출하면 내부 구현 교체 시 alias만 수정
    # ──────────────────────────────────────────────────────────────

    def generate_seeds(self, *args, **kwargs):
        """STAGE: generate_seeds — sample_patch alias"""
        return self.sample_patch(*args, **kwargs)

    def group_anchors(self, *args, **kwargs):
        """STAGE: group_anchors — merge_anchors_heatmap alias"""
        return self.merge_anchors_heatmap(*args, **kwargs)

    def compute_anchor_response(self, *args, **kwargs):
        """STAGE: compute_anchor_response — _apply_group_and_recompute alias"""
        return self._apply_group_and_recompute(*args, **kwargs)

    def project_memory_to_frame(self, *args, **kwargs):
        """STAGE: project_memory_to_frame — filter_memory_anchors_cross_frame alias"""
        return self.filter_memory_anchors_cross_frame(*args, **kwargs)

    def recompute_anchor_vecs(self, *args, **kwargs):
        """STAGE: recompute_anchor_vecs — extract_sample_neighborhood_average_pool alias"""
        return self.extract_sample_neighborhood_average_pool(*args, **kwargs)

    def filter_anchors(self, *args, **kwargs):
        """STAGE: filter_anchors — filter_by_quality alias"""
        return self.filter_by_quality(*args, **kwargs)

    def detect_multiresponse(self, *args, **kwargs):
        """STAGE: detect_multiresponse — detect_mixed_boundary_patches_by_counting alias"""
        return self.detect_mixed_boundary_patches_by_counting(*args, **kwargs)

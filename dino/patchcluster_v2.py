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

    def __init__(self, *args, timing: bool = False, stage_impl: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing = timing  # True 이면 각 서브스텝 처리 시간 출력

        # 스테이지 → 실제 구현 메서드명 매핑. 런타임/호출 시점에 교체 가능하여
        # 각 단계의 알고리즘을 alias 수정 없이 갈아끼울 수 있다(ablation/실험용).
        self.stage_impl = {
            "generate_seeds":          "sample_patch",
            "group_anchors":           "merge_anchors_heatmap",
            "compute_anchor_response": "_apply_group_and_recompute",
            "project_memory_to_frame": "filter_memory_anchors_cross_frame",
            "recompute_anchor_vecs":   "extract_sample_neighborhood_average_pool",
            "filter_anchors":          "filter_by_quality",
            "detect_multiresponse":    "detect_mixed_boundary_patches_by_counting",
        }
        if stage_impl:
            self.stage_impl.update(stage_impl)

    def set_stage_impl(self, stage: str, method):
        """스테이지 구현 교체. method 는 메서드명(str) 또는 callable."""
        self.stage_impl[stage] = method

    def _run_stage(self, stage, *args, method=None, **kwargs):
        """stage 에 매핑된 구현을 실행. method 로 호출 시점 1회 오버라이드 가능.
        method/매핑 값은 메서드명(str, getattr) 또는 callable 둘 다 허용."""
        impl = method or self.stage_impl.get(stage, stage)
        fn = impl if callable(impl) else getattr(self, impl)
        return fn(*args, **kwargs)

    @staticmethod
    def _reduce_ctx(ctx, keep):
        """keep(bool [K]) 마스크로 ctx 의 앵커 차원을 줄인다 (텐서는 참조 슬라이싱).
        K 차원 배열(vom/pure/sim_matrix/H_matrix/avg_vec/sample/centroids/heatmap_sim)만 축소."""
        keep = keep.bool()
        out = dict(ctx)                                   # 얕은 복사(참조만, 데이터 복사 아님)
        K = keep.shape[0]
        for k in ("vom", "pure", "sim_matrix", "H_matrix", "avg_vec", "sample", "centroids"):
            v = out.get(k)
            if isinstance(v, torch.Tensor) and v.shape and v.shape[0] == K:
                out[k] = v[keep]
        hs = out.get("heatmap_sim")
        if isinstance(hs, torch.Tensor) and hs.shape[:1] == (K,):
            out["heatmap_sim"] = hs[keep][:, keep]
        if isinstance(out.get("vom"), torch.Tensor) and out["vom"].numel():
            out["oc"] = out["vom"].sum(dim=0)
        out["K"] = int(keep.sum())
        return out

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

    def compute_anchor_patch_context(self, centroids, feat, self_idx=None,
                                     th_sim=0.60, th_heat=0.65, th_margin=0.12,
                                     use_heatmap_vom=False):
        """앵커별 패치 컨텍스트를 한 번에 계산 (단일/크로스 프레임 공용).

        centroids:
            [K]   long  → 단일 프레임. feat 내 시드 인덱스 (query=feat[idx], 자기 시드 마스킹 자동)
            [K,D] float → 크로스 프레임. 쿼리 벡터(메모리 뱅크 등). 마스킹 없음.
        feat      : [N, D] L2 정규화 대상 프레임 패치 피처
        self_idx  : [K] or None — 명시 시 자기 시드 마스킹 위치 (단일은 자동 설정)

        valid_overlap_mask (vom) [K, N] bool — 1등 앵커와 마진 < th_margin (XFeat 귀속)
        pure_affinity (pure) [K, N] bool      — 단독 점유 패치 (oc == 1)

        자기 시드 마스킹: 쿼리가 feat 안의 패치일 때(단일) 시드가 자기 자신에 sim≈1.0 →
        max_vals 오염 → vom 전체 False. 크로스(쿼리=메모리 벡터)는 feat에 그 패치가
        없으므로 마스킹하지 않는다.

        Returns ctx dict: centroids, sim_matrix[K,N], vom[K,N], pure[K,N],
                          oc[N], H_matrix[K,N], heatmap_sim[K,K], K, N
        (heatmap_sim 은 merge 용 — 단일/메모리 모두 계산)
        """
        t0 = time.time()
        N = feat.shape[0]
        device = feat.device
        K = centroids.shape[0]

        if K == 0:
            _ekn = torch.zeros((0, N), dtype=torch.bool, device=device)
            return dict(
                centroids=centroids,
                sim_matrix=torch.zeros((0, N), device=device),
                vom=_ekn, pure=_ekn,
                oc=torch.zeros(N, device=device),
                H_matrix=torch.zeros((0, N), device=device),
                heatmap_sim=torch.zeros((0, 0), device=device),
                K=0, N=N,
            )

        # 쿼리 결정: 1D=시드 인덱스(단일), 2D=쿼리 벡터(크로스)
        if centroids.dim() == 1:
            seed_idx = centroids.long()
            query = feat[seed_idx]                       # [K, D]
            if self_idx is None:
                self_idx = seed_idx                      # 단일 프레임 → 자기 시드 마스킹
            ret_centroids = seed_idx
        else:
            query = centroids                            # [K, D] 쿼리 벡터(메모리 등)
            ret_centroids = None

        query      = F.normalize(query, p=2, dim=1)
        sim_matrix = torch.mm(query, feat.t())           # [K, N]
        t0 = self._tlog("ctx: sim_matrix mm", t0)

        sim_thr    = torch.where(sim_matrix >= th_sim,
                                 sim_matrix,
                                 torch.zeros_like(sim_matrix))

        # 자기 시드 마스킹 (단일 프레임에서만)
        sim_thr_for_max = sim_thr
        if self_idx is not None:
            self_mask = torch.zeros(K, N, dtype=torch.bool, device=device)
            self_mask[torch.arange(K, device=device), self_idx.long()] = True
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
        heatmap_sim = torch.mm(H_norm, H_norm.t())        # [K, K]  (merge 용, 단일/메모리 공용)
        self._tlog("ctx: heatmap_sim", t0)

        return dict(
            centroids=ret_centroids,
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
    # 파이프라인 통일 API — 스테이지 디스패치
    #   각 스테이지는 self.stage_impl[stage] 에 매핑된 구현으로 라우팅된다.
    #   교체 방법:
    #     - 전역:   objpatcher.set_stage_impl("filter_anchors", "my_filter")
    #               objpatcher.stage_impl["filter_anchors"] = some_callable
    #     - 호출시: objpatcher.filter_anchors(..., method="my_filter")
    #               objpatcher.filter_anchors(..., method=some_callable)
    # ──────────────────────────────────────────────────────────────

    def generate_seeds(self, *args, method=None, **kwargs):
        """STAGE: generate_seeds (기본 sample_patch)"""
        return self._run_stage("generate_seeds", *args, method=method, **kwargs)

    def group_anchors(self, *args, method=None, **kwargs):
        """STAGE: group_anchors (기본 merge_anchors_heatmap)"""
        return self._run_stage("group_anchors", *args, method=method, **kwargs)

    def compute_anchor_response(self, *args, method=None, **kwargs):
        """STAGE: compute_anchor_response (기본 _apply_group_and_recompute)

        ctx 모드: 첫 인자가 dict 이면 ctx 에서 필요한 필드를 꺼내 실행 후 결과를
        ctx 에 써넣어 반환. (ctx 필요 필드: centroids, group_labels, n_comp,
        feat, x_cat, [attn], [grid_shape])
        위치인자 모드(기존): compute_anchor_response(centroids, group_labels, n_comp,
        ctx, feat, x_cat, attn=, grid_shape=) 그대로 동작.
        """
        if args and isinstance(args[0], dict):
            ctx = args[0]
            nc, gvom, gpure, avg, new_ctx = self._run_stage(
                "compute_anchor_response",
                ctx["centroids"], ctx["group_labels"], ctx["n_comp"],
                ctx, ctx["feat"], ctx["x_cat"],
                attn=ctx.get("attn"), grid_shape=ctx.get("grid_shape"),
                method=method, **kwargs)
            new_ctx = dict(new_ctx)
            new_ctx.update({
                "centroids": nc, "sample": nc, "vom": gvom, "pure": gpure,
                "avg_vec": avg,
                "feat": ctx["feat"], "x_cat": ctx["x_cat"],
                "attn": ctx.get("attn"), "grid_shape": ctx.get("grid_shape"),
            })
            return new_ctx
        return self._run_stage("compute_anchor_response", *args, method=method, **kwargs)

    def project_memory_to_frame(self, *args, method=None, **kwargs):
        """STAGE: project_memory_to_frame (기본 filter_memory_anchors_cross_frame)"""
        return self._run_stage("project_memory_to_frame", *args, method=method, **kwargs)

    def recompute_anchor_vecs(self, *args, method=None, **kwargs):
        """STAGE: recompute_anchor_vecs (기본 extract_sample_neighborhood_average_pool)"""
        return self._run_stage("recompute_anchor_vecs", *args, method=method, **kwargs)

    def filter_anchors(self, *args, method=None, reduce=True, **kwargs):
        """STAGE: filter_anchors (기본 filter_by_quality)

        ctx 모드: 첫 인자가 dict 이면 ctx["vom"]/ctx["pure"]/ctx["grid_shape"]를
        꺼내 keep(valid_mask) 계산 → ctx["keep"] 저장 → reduce=True 면 keep 으로
        ctx 앵커 차원 축소하여 반환. (데이터 복사 없이 참조 슬라이싱)
        위치인자 모드(기존): filter_anchors(vom, pure, ...) → keep 반환.
        """
        if args and isinstance(args[0], dict):
            ctx = args[0]
            kwargs.setdefault("grid_shape", ctx.get("grid_shape"))
            keep = self._run_stage("filter_anchors", ctx["vom"], ctx["pure"],
                                   method=method, **kwargs)
            ctx["keep"] = keep
            return self._reduce_ctx(ctx, keep) if reduce else ctx
        return self._run_stage("filter_anchors", *args, method=method, **kwargs)

    def detect_multiresponse(self, *args, method=None, **kwargs):
        """STAGE: detect_multiresponse (기본 detect_mixed_boundary_patches_by_counting)"""
        return self._run_stage("detect_multiresponse", *args, method=method, **kwargs)

    # ──────────────────────────────────────────────────────────────
    # PHASE 2 — 객체 벡터 표현 + 메모리 풀 (서버 인라인에서 이관)
    # ──────────────────────────────────────────────────────────────

    def compute_object_vectors(self, feat, sample, vom, attn=None,
                               repr="avg", weight="uniform", overlap="exclude"):
        """앵커별 객체 벡터 계산 (표현/가중/중복처리 토글).

        feat    : [N, D] L2 정규화 패치 피처
        sample  : [K]    앵커 대표(시드) 패치 인덱스
        vom     : [K, N] valid_overlap_mask (bool)
        attn    : [N] 또는 [1, N] CLS attention (weight='attn' 시 필요)
        repr    : 'avg' | 'patch'
        weight  : 'uniform' | 'attn' | 'sim'
        overlap : 'keep'(vom 전체) | 'exclude'(pure, 기본) | 'argmax'(중복은 최고 sim 앵커)
        반환    : [K, D] L2 정규화 객체 벡터
        """
        D = feat.shape[1]
        K = int(vom.shape[0])
        if K == 0:
            return torch.zeros((0, D), device=feat.device)

        # 대표 패치 표현
        if repr == "patch":
            return F.normalize(feat[sample], p=2, dim=1)

        # ---- 평균(avg) 표현 ----
        vom_b = vom.bool()
        oc = vom_b.sum(dim=0)                                  # [N] overlap count

        if overlap == "exclude":
            membership = vom_b & (oc == 1).unsqueeze(0)        # pure
        elif overlap == "argmax":
            sim_seed = feat[sample] @ feat.t()                 # [K, N]
            winner = sim_seed.masked_fill(~vom_b, float("-inf")).argmax(dim=0)  # [N]
            membership = torch.zeros_like(vom_b)
            n_idx = torch.nonzero(oc > 0, as_tuple=True)[0]
            membership[winner[n_idx], n_idx] = True
        else:  # keep
            membership = vom_b

        # 가중치 [K, N]
        if weight == "attn" and attn is not None:
            w = attn.reshape(1, -1).to(feat.device).expand(K, -1).clone()
        elif weight == "sim":
            w = (feat[sample] @ feat.t()).clamp(min=0.0)
        else:  # uniform
            w = torch.ones((K, feat.shape[0]), device=feat.device)

        w = w * membership.float()
        denom = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        vecs = (w @ feat) / denom                              # [K, D]
        return F.normalize(vecs, p=2, dim=1)

    def build_object_vector_pool(self, dino_mgr, neigh_keys,
                                 repr="avg", max_frames=5):
        """인접 프레임 저장 데이터에서 객체 벡터 풀 구성.

        dino_mgr.get(src, fid) → (x_cat, sample, vom, avg_vec, bind_xfeat)
          repr='avg'   : 저장된 avg_vec(slot 3) 사용
          repr='patch' : x_cat(slot 0)+sample(slot 1)로 대표 패치 피처 재구성
        반환: pool_vecs [M, D] (cpu), pool_meta list[(src, fid, k)]
        """
        vecs, meta, cnt = [], [], 0
        for (nsrc, nfid) in neigh_keys:
            if cnt >= max_frames:
                break
            try:
                data = dino_mgr.get(nsrc, nfid)
            except Exception:
                continue
            if repr == "patch":
                x_cat, sample = data[0], data[1]
                if sample is None or sample.shape[0] == 0:
                    continue
                feat, _, _ = self._prepare_features(x_cat)
                vec = F.normalize(feat[sample], p=2, dim=1)
            else:
                vec = data[3]
            if vec is None or vec.shape[0] == 0:
                continue
            vecs.append(vec.cpu())
            for k in range(vec.shape[0]):
                meta.append((nsrc, nfid, k))
            cnt += 1
        if not vecs:
            return None, []
        return torch.cat(vecs, dim=0), meta

    @staticmethod
    def select_object_vectors(pool_vecs, sim_thresh=0.90, max_k=64):
        """풀에서 중복 객체 표현을 greedy 제거 후 대표 벡터 선택.
        반환: sel_vecs [K, D], keep_idx [K]
        """
        if pool_vecs is None or pool_vecs.shape[0] == 0:
            return None, None
        v = F.normalize(pool_vecs.float(), dim=1)
        M = v.shape[0]
        sim = v @ v.t()
        taken = torch.zeros(M, dtype=torch.bool)
        keep = []
        for i in range(M):
            if taken[i]:
                continue
            keep.append(i)
            taken |= (sim[i] >= sim_thresh)
            if len(keep) >= max_k:
                break
        keep_idx = torch.tensor(keep, dtype=torch.long)
        return pool_vecs[keep_idx], keep_idx

    ##직접 구현
    def test_group(self, ctx, grid_shape):
        print("group test etsestestset testset testset")
    def test_filter(self, ctx, grid_shape):
        print("filter test etsestestset testset testset")
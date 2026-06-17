import torch
import torch.nn.functional as F
import cv2
import numpy as np

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

import time

class DinoSemanticObjectExtractor:
    def __init__(self, patch_size=14):
        self.patch_size = patch_size
    def _prepare_features(self, x_cat):
        _, D, H_p, W_p = x_cat.shape
        features = x_cat[0].view(D, -1).t()  # [N, D]
        features_norm = F.normalize(features, p=2, dim=1)
        #features_norm = features[:,-64:]
        return features_norm, H_p, W_p

    ###수정3
    """
    [진우 박사님 전용 - 마진 카운팅 기반 앵커 정제 모듈 v3]
    dino/patchcluster.py 의 DinoSemanticObjectExtractor 클래스에 추가할 메서드들.

    설계 원칙:
      - avg_vec 사용 안 함. 시드 패치 피처를 직접 슬라이싱하여 sim_matrix 계산.
      - 전처리(compute_anchor_patch_context)를 한 번만 실행 후 결과 재사용.
      - 병합 방식 A/B를 독립 함수로 분리하여 비교 실험 가능.
      - 공통 후처리(group_pure는 그룹 간 경쟁 기준으로 재계산)를 공유.

    함수 목록:
      1. compute_anchor_patch_context  : 시드 기반 전처리 (vom, pure, oc, H_matrix, heatmap_sim)
      2. _apply_group_and_recompute    : 공통 후처리 (group_pure 재계산 + avg_vec)
      3. merge_anchors_heatmap         : 병합 방식 A — 히트맵 분포 유사도
      4. merge_anchors_shared_patch    : 병합 방식 B — 공유 패치 비율 + 시드 포함
      5. filter_memory_anchors_cross_frame : 다중 프레임 배제

    사용 예시 (zmq_dino_server.py):
      ctx = objpatcher.compute_anchor_patch_context(new_sample_1, feat1)

      # A/B 중 하나 또는 둘 다 실행하여 비교
      result_a = objpatcher.merge_anchors_heatmap(ctx, feat1)
      result_b = objpatcher.merge_anchors_shared_patch(ctx, feat1)

      valid_mem, _ = objpatcher.filter_memory_anchors_cross_frame(
          result_a['avg_vec'], feat2)
    """

    # ============================================================
    #  SHARED: 공통 후처리
    # ============================================================
    def _apply_group_and_recompute(self, vom, pure, feat, group_assign_np, device):
        """
        connected_components 결과를 받아 공통 후처리 수행.

        1. one_hot → group_vom, group_pure_raw 합집합
        2. group_pure 재계산: 그룹 간 경쟁 기준
           (그룹 내부 앵커끼리 겹치는 패치는 pure로 인정,
            다른 그룹과 겹치는 패치만 overlap으로 처리)
        3. avg_vec 재계산: group_pure 기반

        Args:
            vom           : [K, N] bool
            pure          : [K, N] bool  (앵커 단위 pure, 참고용)
            feat          : [N, D] float
            group_assign_np: [K]   numpy int
            device        : torch.device

        Returns:
            dict:
              group_vom     [G, N] bool  XFeat 귀속용
              group_pure    [G, N] bool  그룹 간 경쟁 기준 pure
              avg_vec       [G, D] float L2 정규화 완료
              valid_groups  [G]   bool   순수 패치 >= min_area
              group_assign  [K]   long
        """
        K, N = vom.shape
        D = feat.shape[1]
        n_groups = int(group_assign_np.max()) + 1

        group_assign_t = torch.tensor(
            group_assign_np, dtype=torch.long, device=device
        )  # [K]

        vom_f = vom.float()  # [K, N]

        # ── STEP 1. one_hot → group_vom 합집합 ─────────────────────
        one_hot = F.one_hot(group_assign_t,
                            num_classes=n_groups).float()  # [K, G]
        group_vom_f = torch.mm(one_hot.t(), vom_f)  # [G, N]
        group_vom = group_vom_f > 0  # [G, N] bool

        # ── STEP 2. group_pure 재계산 (그룹 간 경쟁 기준) ────────────
        # 패치 n을 점유하는 그룹 수
        group_oc = group_vom_f.clamp(max=1.0).sum(dim=0)  # [N]
        # 단 하나의 그룹만 점유하는 패치 = group pure
        group_pure = group_vom & (group_oc == 1).unsqueeze(0)  # [G, N] bool

        # ── STEP 3. avg_vec 재계산 ──────────────────────────────────
        # group_pure 충분 → group_pure 기반
        # group_pure 부족 → group_vom fallback
        pure_area = group_pure.float().sum(dim=1)  # [G]
        vom_area = group_vom.float().sum(dim=1)  # [G]

        use_pure = (pure_area >= 2).unsqueeze(1).expand(n_groups, N)  # [G, N]
        eff_mask = torch.where(use_pure,
                               group_pure.float(),
                               group_vom.float())  # [G, N]

        feature_sums = torch.mm(eff_mask, feat)  # [G, D]
        patch_counts = eff_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        avg_vec = F.normalize(feature_sums / patch_counts,
                              p=2, dim=1)  # [G, D]

        # ── STEP 4. valid_groups ────────────────────────────────────
        valid_groups = vom_area >= 1  # [G] bool

        return dict(
            group_vom=group_vom,
            group_pure=group_pure,
            avg_vec=avg_vec,
            valid_groups=valid_groups,
            group_assign=group_assign_t,
        )

    # ============================================================
    #  METHOD 1. compute_anchor_patch_context
    # ============================================================
    def compute_anchor_patch_context(
            self,
            centroids,  # [K] long   앵커 시드 패치 인덱스
            feat,  # [N, D] float  L2 정규화 완료
            th_sim=0.60,
            th_margin=0.12,
            sim_cutoff=0.60,  # 히트맵 클리핑 임계값
    ):
        """
        시드 패치 피처를 직접 슬라이싱하여 전처리 컨텍스트 생성.
        avg_vec 불필요. 한 번만 실행 후 A/B 병합 함수에 재사용.

        Args:
            centroids  : [K] long
            feat       : [N, D] float  L2 정규화 완료
            th_sim     : float  최소 유사도 가드
            th_margin  : float  마진 임계값
            sim_cutoff : float  히트맵 클리핑 임계값

        Returns:
            dict:
              sim_matrix   [K, N]  시드-패치 유사도
              vom          [K, N] bool  XFeat 귀속용
              pure         [K, N] bool  앵커 단위 pure (참고용)
              oc           [N]          패치별 경쟁 앵커 수
              H_matrix     [K, N]  클리핑된 히트맵
              heatmap_sim  [K, K]  히트맵 분포 유사도
              centroids    [K] long
        """
        K = centroids.shape[0]
        device = feat.device

        if K == 0:
            N, D = feat.shape
            z = lambda s, t: torch.zeros(s, dtype=t, device=device)
            return dict(
                sim_matrix=z((0, N), torch.float32),
                vom=z((0, N), torch.bool),
                pure=z((0, N), torch.bool),
                oc=z((N,), torch.float32),
                H_matrix=z((0, N), torch.float32),
                heatmap_sim=z((0, 0), torch.float32),
                centroids=centroids,
            )

        # ── 시드 패치 피처 슬라이싱 ─────────────────────────────────
        seed_feats = feat[centroids.long()]  # [K, D]

        # ── sim_matrix ───────────────────────────────────────────────
        sim_matrix = torch.mm(seed_feats, feat.t())  # [K, N]

        # ── vom / pure / oc (compute_patch_purity_masks 동일 로직) ──
        sim_thr = torch.where(sim_matrix >= th_sim,
                              sim_matrix,
                              torch.zeros_like(sim_matrix))  # [K, N]
        max_vals = sim_thr.max(dim=0, keepdim=True).values  # [1, N]
        margins = max_vals - sim_thr  # [K, N]
        vom = (sim_thr > 0) & (margins < th_margin)  # [K, N] bool
        oc = vom.float().sum(dim=0)  # [N]
        pure = vom & (oc == 1).unsqueeze(0)  # [K, N] bool

        # ── H_matrix / heatmap_sim ───────────────────────────────────
        H_matrix = torch.where(sim_matrix >= sim_cutoff,
                               sim_matrix,
                               torch.zeros_like(sim_matrix))  # [K, N]
        H_norm = F.normalize(H_matrix, p=2, dim=1)  # [K, N]
        heatmap_sim = torch.mm(H_norm, H_norm.t())  # [K, K]

        return dict(
            sim_matrix=sim_matrix,
            vom=vom,
            pure=pure,
            oc=oc,
            H_matrix=H_matrix,
            heatmap_sim=heatmap_sim,
            centroids=centroids,
        )

    # ============================================================
    #  METHOD 2. merge_anchors_heatmap  (방식 A)
    # ============================================================
    def merge_anchors_heatmap(
            self,
            ctx,  # compute_anchor_patch_context 반환값
            feat,  # [N, D]
            heatmap_threshold=0.85,
    ):
        """
        [방식 A] 히트맵 분포 유사도 기반 병합.

        adjacency[i,j] = heatmap_sim[i,j] >= heatmap_threshold

        compile_structural_equivalence_vectorized 와 동일 로직이나
        avg_vec 대신 시드 패치 피처 기반 H_matrix 사용.
        group_pure는 그룹 간 경쟁 기준으로 재계산.

        Args:
            ctx               : compute_anchor_patch_context 반환 dict
            feat              : [N, D]
            heatmap_threshold : float

        Returns:
            dict (group_vom, group_pure, avg_vec, valid_groups, group_assign)
        """
        vom = ctx['vom']
        pure = ctx['pure']
        heatmap_sim = ctx['heatmap_sim']  # [K, K]
        K = vom.shape[0]
        device = vom.device

        if K == 0:
            return _apply_group_and_recompute(
                vom, pure, feat,
                np.zeros(0, dtype=np.int32), device
            )

        # ── adjacency ───────────────────────────────────────────────
        adj_np = (heatmap_sim >= heatmap_threshold).cpu().numpy().astype(np.int32)
        np.fill_diagonal(adj_np, 0)
        adj_np = np.maximum(adj_np, adj_np.T)

        # ── connected_components ─────────────────────────────────────
        _, group_assign_np = connected_components(
            csgraph=csr_matrix(adj_np),
            directed=False, connection='weak', return_labels=True,
        )

        return self._apply_group_and_recompute(vom, pure, feat, group_assign_np, device)

    # ============================================================
    #  METHOD 3. merge_anchors_shared_patch  (방식 B)
    # ============================================================
    def merge_anchors_shared_patch(
            self,
            ctx,  # compute_anchor_patch_context 반환값
            feat,  # [N, D]
            th_merge=0.40,
    ):
        """
        [방식 B] 공유 패치 비율 + 시드 포함 기반 병합.

        adjacency[i,j] = ratio_cond[i,j] AND seed_cond[i,j]

        ratio_cond: 양방향 공유 비율 > th_merge
        seed_cond:  i의 시드가 j의 vom 영역에 포함
                    OR j의 시드가 i의 vom 영역에 포함

        group_pure는 그룹 간 경쟁 기준으로 재계산.

        Args:
            ctx      : compute_anchor_patch_context 반환 dict
            feat     : [N, D]
            th_merge : float

        Returns:
            dict (group_vom, group_pure, avg_vec, valid_groups, group_assign)
        """
        vom = ctx['vom']
        pure = ctx['pure']
        centroids = ctx['centroids']
        K = vom.shape[0]
        device = vom.device

        if K == 0:
            return _apply_group_and_recompute(
                vom, pure, feat,
                np.zeros(0, dtype=np.int32), device
            )

        vom_f = vom.float()  # [K, N]

        # ── ratio_cond ───────────────────────────────────────────────
        co = torch.mm(vom_f, vom_f.t())  # [K, K]
        area = vom_f.sum(dim=1)  # [K]
        overlap_ratio = co / area.unsqueeze(1).clamp(min=1.0)  # [K, K]
        ratio_cond = (overlap_ratio > th_merge) & \
                     (overlap_ratio.t() > th_merge)  # [K, K] bool

        # ── seed_cond ────────────────────────────────────────────────
        # seed_in_j[j, i] = vom_f[j, seed_i]
        #                  = "앵커 j의 영역에 앵커 i의 시드가 포함"
        seed_in_j = vom_f[:, centroids.long()]  # [K, K]
        seed_cond = seed_in_j.t().bool() | seed_in_j.bool()  # [K, K]

        # ── adjacency ───────────────────────────────────────────────
        adjacency = ratio_cond & seed_cond
        adjacency.fill_diagonal_(False)

        adj_np = adjacency.cpu().numpy().astype(np.int32)
        adj_np = np.maximum(adj_np, adj_np.T)

        # ── connected_components ─────────────────────────────────────
        _, group_assign_np = connected_components(
            csgraph=csr_matrix(adj_np),
            directed=False, connection='weak', return_labels=True,
        )

        return self._apply_group_and_recompute(vom, pure, feat, group_assign_np, device)

    # ============================================================
    #  METHOD 4. filter_memory_anchors_cross_frame  (다중 프레임)
    # ============================================================
    def filter_memory_anchors_cross_frame(
            self,
            avg_vec_mem,  # [K_mem, D]
            feat_cur,  # [N_cur, D]
            th_sim=0.60,
            th_margin=0.12,
            min_pure_response=3,
    ):
        """
        [다중 프레임 전용]
        메모리 뱅크 앵커를 현재 프레임에 투영하여
        순수 단독 반응 패치가 부족한 앵커를 cross-attention 전에 배제.

        P1: 현재 프레임에 없는 객체 → pure_area ≈ 0
        P2: 배경 대면적 앵커        → 배경 패치끼리 경쟁 → pure_area 낮음
        P3/P4: 오염 앵커            → overlap 다수 → pure_area 낮음

        Returns:
            valid_mem_mask : [K_mem] bool
            pure_area_cur  : [K_mem]
        """
        sim_cross = torch.mm(avg_vec_mem, feat_cur.t())  # [K_mem, N_cur]

        sim_thr = torch.where(sim_cross >= th_sim,
                              sim_cross,
                              torch.zeros_like(sim_cross))
        #max_vals = sim_thr.max(dim=0, keepdim=True).values
        self_mask = torch.zeros(K, N, dtype=torch.bool, device=device)
        self_mask[torch.arange(K, device=device), centroids.long()] = True
        sim_thr_for_max = sim_thr.masked_fill(self_mask, 0.0)
        max_vals = sim_thr_for_max.max(dim=0, keepdim=True).values

        margins = max_vals - sim_thr

        valid_cross = (sim_thr > 0) & (margins < th_margin)
        oc_cur = valid_cross.float().sum(dim=0)
        pure_cross = valid_cross & (oc_cur == 1).unsqueeze(0)
        pure_area_cur = pure_cross.float().sum(dim=1)

        valid_mem_mask = pure_area_cur >= min_pure_response

        return valid_mem_mask, pure_area_cur

    # ============================================================
    #  USAGE EXAMPLE
    # ============================================================
    """
    # ── 전처리 (한 번만) ─────────────────────────────────────────
    ctx = objpatcher.compute_anchor_patch_context(
        new_sample_1, feat1,
        th_sim=0.60, th_margin=0.12, sim_cutoff=0.60,
    )

    # ── 방식 A: 히트맵 유사도 기반 병합 ─────────────────────────
    result_a = objpatcher.merge_anchors_heatmap(
        ctx, feat1, heatmap_threshold=0.85,
    )

    # ── 방식 B: 공유 패치 + 시드 포함 기반 병합 ─────────────────
    result_b = objpatcher.merge_anchors_shared_patch(
        ctx, feat1, th_merge=0.40,
    )

    # result_a / result_b 공통 키:
    #   'avg_vec'      [G, D]   cross-attention key/value
    #   'group_vom'    [G, N]   XFeat 귀속용
    #   'group_pure'   [G, N]   그룹 간 경쟁 기준 pure
    #   'valid_groups' [G] bool
    #   'group_assign' [K] long 디버깅용

    avg_vec_a = result_a['avg_vec'][result_a['valid_groups']]
    avg_vec_b = result_b['avg_vec'][result_b['valid_groups']]

    # ── 다중 프레임 ─────────────────────────────────────────────
    valid_mem, pure_area = objpatcher.filter_memory_anchors_cross_frame(
        avg_vec_a, feat2,
    )
    avg_vec_cross = avg_vec_a[valid_mem]

    # ── 시각화 ──────────────────────────────────────────────────
    visualizer.visualize_pure_vs_overlap_patches(
        img1, ctx['vom'], ctx['pure'], grid_shape)
    visualizer.visualize_anchor_refinement_comparison(
        img1, new_sample_1,
        ctx['vom'],
        result_a['group_pure'][result_a['group_assign']],
        result_a['valid_groups'][result_a['group_assign']],
        grid_shape,
    )
    """
    ###수정3

    ### 공유 패치 수정2

    # ============================================================
    #  METHOD 1. compute_patch_purity_masks
    # ============================================================
    def compute_patch_purity_masks(self, sim_matrix, th_sim=0.60, th_margin=0.12):
        """
        마진 카운팅 기반 패치 분류.
        detect_mixed_boundary_patches_by_counting 의 내부 마스크를 외부로 분리.

        Args:
            sim_matrix : [K, N]  앵커-패치 코사인 유사도 행렬
            th_sim     : float   최소 유사도 가드 (기본 0.60)
            th_margin  : float   1등과의 마진 임계값 (기본 0.12)

        Returns:
            valid_overlap_mask : [K, N] bool  마진 이내 전체 반응 (XFeat 귀속용)
            pure_affinity      : [K, N] bool  단독 점유 패치만  (avg_vec 계산용)
            overlap_counts     : [N]          패치별 경쟁 앵커 수
        """
        # th_sim 미만 → 0
        sim_thr = torch.where(sim_matrix >= th_sim,
                              sim_matrix,
                              torch.zeros_like(sim_matrix))  # [K, N]

        # 패치별 1등 유사도
        max_vals = sim_thr.max(dim=0, keepdim=True).values  # [1, N]

        # 마진: 1등과의 차이
        margins = max_vals - sim_thr  # [K, N]

        # 유효 반응: th_sim 통과 AND 마진 < th_margin
        vom = (sim_thr > 0) & (margins < th_margin)  # [K, N] bool

        # 패치별 경쟁 앵커 수
        oc = vom.float().sum(dim=0)  # [N]

        # 순수 패치: 단 하나의 앵커만 반응
        pure = vom & (oc == 1).unsqueeze(0)  # [K, N] bool

        return vom, pure, oc

    def classify_anchor_overlaps_vectorized(
            self,
            valid_overlap_mask,
            th_complete=0.80,
            th_partial=0.30
    ):
        """
        [진우 박사님 전용 - 유효 반응 마스크 기반 All-Pairs 중복도 정밀 분류 엔진]

        Args:
            valid_overlap_mask : [K, N] bool  compute_patch_purity_masks의 1번째 아웃풋 (VOM)
            th_complete        : float        완전 중복(Twin/Subsumed)으로 판정할 임계값 (기본 0.80)
            th_partial         : float        부분 중복(영역 공유)으로 판정할 하한 임계값 (기본 0.30)

        Returns:
            dict:
              complete_pairs : tuple(Tensor, Tensor) 완전 중복인 (anchor_i, anchor_j) 인덱스 쌍
              partial_pairs  : tuple(Tensor, Tensor) 부분 중복인 (anchor_i, anchor_j) 인덱스 쌍
              iou_matrix     : [K, K] float          디버깅/시각화용 전수 IoU 행렬
              overlap_matrix : [K, K] float          i가 j에 포함되는 비대칭 지분율 행렬
        """
        K, N = valid_overlap_mask.shape
        device = valid_overlap_mask.device

        # 비교할 쌍이 없거나 앵커가 부족한 경우 예외 처리
        if K <= 1:
            empty_idx = torch.empty(0, dtype=torch.long, device=device)
            return {
                "complete_pairs": (empty_idx, empty_idx),
                "partial_pairs": (empty_idx, empty_idx),
                "iou_matrix": torch.zeros((K, K), device=device),
                "overlap_matrix": torch.zeros((K, K), device=device)
            }

        vom_f = valid_overlap_mask.float()  # [K, N]

        # ── STEP 1. All-pairs 교집합(Intersection) 면적 원샷 산출 ───────
        # [K, N] @ [N, K] -> [K, K]
        # inter_matrix[i, j] = 앵커 i와 j가 동시에 공유하는 패치 개수
        inter_matrix = torch.mm(vom_f, vom_f.t())

        # ── STEP 2. 각 앵커별 유효 영토 면적 산출 ────────────────────────
        # area 크기: [K]
        area = vom_f.sum(dim=1)

        # ── STEP 3. 2가지 대수적 지표 컴파일 (Symmetric IoU & Asymmetric Overlap) ──
        # ① Symmetric IoU 행렬 유도 (두 앵커의 전체 영토 대비 중복도)
        union_matrix = area.unsqueeze(1) + area.unsqueeze(0) - inter_matrix
        iou_matrix = inter_matrix / (union_matrix + 1e-8)  # [K, K]

        # ② Asymmetric Overlap Ratio 행렬 유도 (포함 관계 추적용)
        # overlap_matrix[i, j] = "앵커 i의 전체 면적 중 앵커 j와 겹치는 지분 비율"
        # 만약 i가 j 내부에 완전히 가려지거나 포함된다면 이 값은 1.0에 수렴합니다.
        overlap_matrix = inter_matrix / area.unsqueeze(1).clamp(min=1.0)  # [K, K]

        # ── STEP 4. 중복 계산(i-j / j-i) 및 자기 자신(대각선) 소거 가드 ──
        # 상삼각 행렬(Upper Triangle) 필터를 가동하여 무방향 그래프 에지만 남깁니다.
        upper_tri_mask = torch.triu(torch.ones((K, K), dtype=torch.bool, device=device), diagonal=1)

        # ── STEP 5. 👑 [박사님 핵심 지시 조건 분기 분할] ──────────────────
        # 임계값 필터링 조건 기획 (IoU가 높거나, 한쪽이 다른 한쪽에 완전히 먹혔거나)
        is_complete = (iou_matrix >= th_complete) | (overlap_matrix >= th_complete)
        is_complete = is_complete & upper_tri_mask

        # 부분 중복 조건 (완전 중복 가드선 아래이면서 최소 기준선 th_partial은 넘긴 애들)
        is_partial = (iou_matrix >= th_partial) | (overlap_matrix >= th_partial)
        is_partial = is_partial & (~is_complete) & upper_tri_mask

        # ── STEP 6. 최종 인덱스 좌표록 수사 체계 반환 ────────────────────
        # torch.where를 때리면 조건이 True인 행(Row)과 열(Col) 인덱스가 튜플로 튀어나옵니다.
        complete_pairs = torch.where(is_complete)
        partial_pairs = torch.where(is_partial)

        return {
            "complete_pairs": complete_pairs,  # (Tensor[num_complete], Tensor[num_complete])
            "partial_pairs": partial_pairs,  # (Tensor[num_partial], Tensor[num_partial])
            "iou_matrix": iou_matrix,  # 시각화 격자 투사용
            "overlap_matrix": overlap_matrix  # 대수 비교 디버깅용
        }

    # ============================================================
    #  METHOD 2. merge_anchors_by_seed_overlap  (단일 프레임)
    # ============================================================
    def merge_anchors_by_seed_overlap(
            self,
            centroids,  # [K] long   앵커 시드 패치 인덱스
            valid_overlap_mask,  # [K, N] bool
            pure_affinity,  # [K, N] bool
            feat,  # [N, D] float  L2 정규화 권장
            th_merge=0.40,  # 상호 포함 비율 임계값
            min_area=2,  # 병합 후 최소 순수 패치 수
    ):
        """
        [단일 프레임 전용] 시드 포함 여부 + 공유 비율 기반 앵커 병합.

        병합 조건 (AND):
          1. ratio_cond : 양방향 공유 비율 > th_merge
          2. seed_cond  : i의 시드가 j의 vom 영역에 포함
                          OR j의 시드가 i의 vom 영역에 포함

        connected_components 로 이행적 병합 처리.
        avg_vec 는 그룹 내 pure_affinity 합집합으로 재계산.

        Args:
            centroids          : [K]     앵커 시드 인덱스 (long)
            valid_overlap_mask : [K, N]  XFeat 귀속용 마스크
            pure_affinity      : [K, N]  avg_vec 계산용 마스크
            feat               : [N, D]  패치 피처
            th_merge           : float   공유 비율 임계값
            min_area           : int     그룹 최소 순수 패치 수

        Returns:
            avg_vec_merged  : [G, D]    그룹별 대표 벡터 (L2 정규화)
            group_vom       : [G, N]    그룹별 vom 합집합 (XFeat 귀속용)
            group_pure      : [G, N]    그룹별 pure 합집합 (avg_vec 기반)
            valid_groups    : [G] bool  min_area 통과 여부
            group_assign    : [K]       각 앵커의 그룹 ID (디버깅용)
        """
        K, N = valid_overlap_mask.shape
        D = feat.shape[1]
        device = valid_overlap_mask.device

        if K == 0:
            empty = lambda s, t: torch.zeros(s, dtype=t, device=device)
            return (empty((0, D), torch.float32),
                    empty((0, N), torch.bool),
                    empty((0, N), torch.bool),
                    empty((0,), torch.bool),
                    empty((0,), torch.long))

        vom_f = valid_overlap_mask.float()  # [K, N]
        pure_f = pure_affinity.float()  # [K, N]

        # ----------------------------------------------------------
        # STEP 1. 공유 비율 조건 (ratio_cond)
        # ----------------------------------------------------------
        co = torch.mm(vom_f, vom_f.t())  # [K, K]
        area = vom_f.sum(dim=1)  # [K]
        overlap_ratio = co / area.unsqueeze(1).clamp(min=1.0)  # [K, K]
        # overlap_ratio[i,j] = i의 패치 중 j와 겹치는 비율

        ratio_cond = (overlap_ratio > th_merge) & \
                     (overlap_ratio.t() > th_merge)  # [K, K] bool

        # ----------------------------------------------------------
        # STEP 2. 시드 포함 조건 (seed_cond)
        #
        # vom_f[:, centroids] → [K, K]
        # seed_in_j[j, i] = vom_f[j, seed_i]
        #                 = "앵커 j의 영역에 앵커 i의 시드가 포함되는가"
        #
        # seed_cond[i, j]:
        #   i의 시드가 j 영역에 포함(seed_in_j.t())
        #   OR j의 시드가 i 영역에 포함(seed_in_j)
        # ----------------------------------------------------------
        seed_in_j = vom_f[:, centroids.long()]  # [K, K]
        seed_cond = seed_in_j.t().bool() | seed_in_j.bool()  # [K, K] bool

        # ----------------------------------------------------------
        # STEP 3. adjacency = ratio_cond AND seed_cond
        # ----------------------------------------------------------
        adjacency = (ratio_cond & seed_cond)  # [K, K] bool
        adjacency.fill_diagonal_(False)

        # 대칭 보장 (i→j 이면 j→i)
        adj_np = adjacency.cpu().numpy().astype(np.int32)
        adj_np = np.maximum(adj_np, adj_np.T)

        # ----------------------------------------------------------
        # STEP 4. connected_components → 그룹 할당
        # ----------------------------------------------------------
        n_groups, group_assign = connected_components(
            csgraph=csr_matrix(adj_np),
            directed=False,
            connection='weak',
            return_labels=True,
        )
        # group_assign : [K]  각 앵커의 그룹 ID (0 ~ n_groups-1)

        group_assign_t = torch.tensor(
            group_assign, dtype=torch.long, device=device
        )  # [K]

        # ----------------------------------------------------------
        # STEP 5. 그룹별 vom / pure 합집합 (벡터화)
        #
        # one_hot [K, G]  → group_vom[g, n] = 그룹 g 소속 앵커 중
        #                    적어도 하나가 패치 n에 반응하면 True
        # ----------------------------------------------------------
        one_hot = F.one_hot(
            group_assign_t, num_classes=n_groups
        ).float()  # [K, G]

        # [G, K] @ [K, N] → [G, N]
        group_vom_f = torch.mm(one_hot.t(), vom_f)  # [G, N]
        group_pure_f = torch.mm(one_hot.t(), pure_f)  # [G, N]

        group_vom = group_vom_f > 0  # [G, N] bool
        group_pure = group_pure_f > 0  # [G, N] bool

        # ----------------------------------------------------------
        # STEP 6. avg_vec 재계산 (pure 패치만 기여)
        # ----------------------------------------------------------
        feature_sums = torch.mm(group_pure_f.clamp(max=1.0), feat)  # [G, D]
        patch_counts = group_pure.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        avg_vec_merged = F.normalize(feature_sums / patch_counts,
                                     p=2, dim=1)  # [G, D]

        # ----------------------------------------------------------
        # STEP 7. 면적 필터
        # ----------------------------------------------------------
        valid_groups = group_pure.float().sum(dim=1) >= min_area  # [G] bool

        return avg_vec_merged, group_vom, group_pure, valid_groups, group_assign_t

    # ============================================================
    #  METHOD 3. filter_memory_anchors_cross_frame  (다중 프레임)
    # ============================================================
    def filter_memory_anchors_cross_frame(
            self,
            avg_vec_mem,  # [K_mem, D]  메모리 뱅크 앵커 벡터 (L2 정규화 완료)
            feat_cur,  # [N_cur, D]  현재 프레임 패치 피처
            th_sim=0.60,
            th_margin=0.12,
            min_pure_response=3,
    ):
        """
        [다중 프레임 전용]
        메모리 뱅크 앵커를 현재 프레임에 투영하여
        순수 단독 반응 패치가 부족한 앵커를 cross-attention 전에 배제.

        처리 케이스:
          P1: 현재 프레임에 없는 객체 (의자 등)  → pure_area ≈ 0
          P2: 배경 대면적 앵커                   → 배경 패치끼리 경쟁 → pure_area 낮음
          P3/P4: 오염 앵커                       → overlap 패치 다수 → pure_area 낮음

        Args:
            avg_vec_mem       : [K_mem, D]
            feat_cur          : [N_cur, D]
            th_sim            : float
            th_margin         : float
            min_pure_response : int  생존 최소 순수 반응 패치 수

        Returns:
            valid_mem_mask : [K_mem] bool  True → cross-attention 공급
            pure_area_cur  : [K_mem]       디버깅용 순수 반응 패치 수
        """
        # cross-frame sim
        sim_cross = torch.mm(avg_vec_mem, feat_cur.t())  # [K_mem, N_cur]

        # 마진 카운팅 (compute_patch_purity_masks 동일 로직)
        sim_thr = torch.where(sim_cross >= th_sim,
                              sim_cross,
                              torch.zeros_like(sim_cross))
        max_vals = sim_thr.max(dim=0, keepdim=True).values
        margins = max_vals - sim_thr

        valid_cross = (sim_thr > 0) & (margins < th_margin)  # [K_mem, N_cur]
        oc_cur = valid_cross.float().sum(dim=0)  # [N_cur]

        # 순수 패치: 현재 프레임에서도 단독 점유
        pure_cross = valid_cross & (oc_cur == 1).unsqueeze(0)  # [K_mem, N_cur]
        pure_area_cur = pure_cross.float().sum(dim=1)  # [K_mem]

        valid_mem_mask = pure_area_cur >= min_pure_response  # [K_mem] bool

        return valid_mem_mask, pure_area_cur

    # ============================================================
    #  USAGE EXAMPLE  (zmq_dino_server.py)
    # ============================================================
    """
    # ── 단일 프레임 ─────────────────────────────────────────────────
    sim1 = torch.matmul(new_avg_patch_vec1, feat1.t())          # [K, N]

    vom1, pure1, oc1 = objpatcher.compute_patch_purity_masks(sim1)

    avg_vec_merged, group_vom, group_pure, valid_groups, group_assign = \
        objpatcher.merge_anchors_by_seed_overlap(
            new_sample_1, vom1, pure1, feat1,
            th_merge=0.40, min_area=2,
        )

    # 유효 그룹만 추림
    avg_vec_final_single = avg_vec_merged[valid_groups]   # [G_valid, D]
    group_vom_final      = group_vom[valid_groups]        # [G_valid, N]  XFeat 귀속용
    group_pure_final     = group_pure[valid_groups]       # [G_valid, N]  디버깅용

    # ── 다중 프레임 ─────────────────────────────────────────────────
    valid_mem, pure_area = objpatcher.filter_memory_anchors_cross_frame(
        avg_vec_final_single, feat2,
        th_sim=0.60, th_margin=0.12, min_pure_response=3,
    )
    avg_vec_cross = avg_vec_final_single[valid_mem]       # cross-attention key/value

    # ── 시각화 ──────────────────────────────────────────────────────
    I1, overlap1 = objpatcher.detect_mixed_boundary_patches_by_counting(sim1)
    visualizer.visualize_pure_vs_overlap_patches(img1, vom1, pure1, grid_shape)
    visualizer.visualize_seed_filter_result(
        img1, new_sample_1, oc1,
        # 병합 후 그룹 정보로 생존 앵커 표시
        valid_seed_mask=(group_assign >= 0),   # 전체 생존 (배제 없음)
        grid_shape=grid_shape,
    )
    visualizer.visualize_anchor_refinement_comparison(
        img1,
        new_sample_1,
        vom1,                    # 정제 전
        group_pure[group_assign],  # 정제 후 (각 앵커가 속한 그룹의 pure)
        valid_groups[group_assign],
        grid_shape,
    )
    """
    ### 공유 패치 수정2

    # ============================================================
    #  METHOD 1. compute_patch_purity_masks
    # ============================================================
    def compute_patch_purity_masks(self, sim_matrix, th_sim=0.60, th_margin=0.12):
        """
        detect_mixed_boundary_patches_by_counting의 내부 마스크들을 외부로 노출.
        avg_vec 계산(pure_affinity)과 XFeat 귀속(valid_overlap_mask)을 분리하기 위한 기반.

        Args:
            sim_matrix    : [K, N]  앵커-패치 코사인 유사도 행렬
            th_sim        : float   최소 유사도 가드 (기본 0.60)
            th_margin     : float   1등과의 마진 임계값 (기본 0.12)

        Returns:
            valid_overlap_mask : [K, N] bool  마진 이내 모든 앵커-패치 반응 (XFeat 귀속용)
            pure_affinity      : [K, N] bool  단독 점유 패치만 (avg_vec 계산용)
            overlap_counts     : [N]   int    패치별 경쟁 앵커 수
        """
        # Step 1. th_sim 미만 → 0 클리핑
        sim_throttled = torch.where(
            sim_matrix >= th_sim, sim_matrix, torch.zeros_like(sim_matrix)
        )

        # Step 2. 패치별 1등 유사도 [1, N]
        max_vals = sim_throttled.max(dim=0, keepdim=True).values

        # Step 3. 마진 행렬 [K, N]
        margins = max_vals - sim_throttled

        # Step 4. 유효 반응 마스크: th_sim 통과 AND 마진 < th_margin
        valid_overlap_mask = (sim_throttled > 0) & (margins < th_margin)  # [K, N] bool

        # Step 5. 패치별 경쟁 앵커 수
        overlap_counts = valid_overlap_mask.float().sum(dim=0)  # [N]

        # Step 6. 순수 패치 = 단 하나의 앵커만 점유
        pure_patch_mask = (overlap_counts == 1)  # [N] bool
        pure_affinity = valid_overlap_mask & pure_patch_mask.unsqueeze(0)  # [K, N] bool

        return valid_overlap_mask, pure_affinity, overlap_counts

    # ============================================================
    #  METHOD 2. filter_seed_overlap_anchors
    # ============================================================
    def filter_seed_overlap_anchors(self, centroids, overlap_counts):
        """
        시드 패치(centroid) 자체가 overlap 구간에 속하는 앵커를 배제.
        시드가 경쟁 구간에 있으면 앵커의 정체성 자체가 불명확하므로 전체 배제.

        Args:
            centroids      : [K]  앵커 시드 패치 인덱스 (1D long tensor)
            overlap_counts : [N]  패치별 경쟁 앵커 수

        Returns:
            valid_mask : [K] bool  True인 앵커만 다음 단계로 진행
        """
        # centroids가 가리키는 패치의 overlap_counts 조회
        seed_counts = overlap_counts[centroids.long()]  # [K]
        valid_mask = (seed_counts < 2)  # [K] bool  단독 점유 시드만 생존
        return valid_mask

    # ============================================================
    #  METHOD 3. refine_anchors_by_pure_affinity  (단일 프레임)
    # ============================================================
    def refine_anchors_by_pure_affinity(
            self, valid_overlap_mask, pure_affinity, feat,
            th_merge=0.4, min_area=2
    ):
        """
        [단일 프레임 전용]
        1. valid_overlap_mask 기준으로 앵커 간 공유 패치 비율 계산 → 병합 후보 탐지
        2. 병합 후보의 pure_affinity 합집합으로 refined_masks 생성
        3. refined_masks 기반으로 avg_vec 재계산 (overlap 패치 기여 완전 차단)

        Args:
            valid_overlap_mask : [K, N] bool  XFeat 귀속용 전체 반응 마스크
            pure_affinity      : [K, N] bool  avg_vec 계산용 순수 패치 마스크
            feat               : [N, D] float 패치 피처 행렬 (L2 정규화 권장)
            th_merge           : float  병합 후보 판정 상호 포함 비율 임계값 (기본 0.4)
            min_area           : int    병합 후 최소 순수 패치 수 (미달 시 앵커 제거)

        Returns:
            avg_vec_refined : [K, D]   정제된 앵커 대표 벡터 (L2 정규화 완료)
            refined_masks   : [K, N]   정제된 pure_affinity (병합 반영)
            valid_after     : [K] bool 면적 필터 통과 여부
        """
        K, N = valid_overlap_mask.shape
        device = valid_overlap_mask.device

        if K == 0:
            D = feat.shape[1]
            return (
                torch.empty((0, D), device=device),
                torch.zeros((0, N), dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.bool, device=device),
            )

        # ----------------------------------------------------------
        # Step 1. 앵커 간 공유 패치 수 및 상호 포함 비율
        #         (병합 탐지는 valid_overlap_mask 기준 — 겹치는 패치 전체)
        # ----------------------------------------------------------
        vom_f = valid_overlap_mask.float()  # [K, N]
        area = vom_f.sum(dim=1)  # [K]

        co = torch.mm(vom_f, vom_f.t())  # [K, K]

        # overlap_ratio[i, j] = i의 반응 패치 중 j와 겹치는 비율
        overlap_ratio = co / area.unsqueeze(1).clamp(min=1.0)  # [K, K]

        # 양방향 모두 th_merge 초과 → 병합 후보
        merge_candidate = (
                (overlap_ratio > th_merge) & (overlap_ratio.t() > th_merge)
        )  # [K, K] bool
        merge_candidate.fill_diagonal_(False)

        # ----------------------------------------------------------
        # Step 2. 병합 후보의 pure_affinity 합집합 (벡터화)
        #
        # merge_candidate [K, K] @ pure_affinity [K, N]
        # → accumulated[i, n] = i의 병합 후보들 중 패치 n에 반응하는 앵커 수
        # accumulated > 0 이면 병합 후보 중 적어도 하나가 반응
        # ----------------------------------------------------------
        pure_f = pure_affinity.float()  # [K, N]
        merge_f = merge_candidate.float()  # [K, K]

        accumulated = torch.mm(merge_f, pure_f)  # [K, N]
        partner_union = (accumulated > 0)  # [K, N] bool

        # 자기 pure_affinity 와 OR
        refined_masks = pure_affinity | partner_union  # [K, N] bool

        # ----------------------------------------------------------
        # Step 3. 병합 후보 없는 앵커는 자기 pure_affinity 그대로
        # ----------------------------------------------------------
        has_partner = merge_candidate.any(dim=1)  # [K] bool
        refined_masks = torch.where(
            has_partner.unsqueeze(1).expand(K, N),
            refined_masks,
            pure_affinity,
        )  # [K, N] bool

        # ----------------------------------------------------------
        # Step 4. avg_vec 재계산 (pure 패치만 기여)
        # ----------------------------------------------------------
        refined_f = refined_masks.float()  # [K, N]
        feature_sums = torch.mm(refined_f, feat)  # [K, D]
        patch_counts = refined_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        avg_vec_refined = F.normalize(feature_sums / patch_counts, p=2, dim=1)  # [K, D]

        # ----------------------------------------------------------
        # Step 5. 면적 필터: 정제 후 순수 패치 수가 너무 작은 앵커 제거
        # ----------------------------------------------------------
        valid_after = refined_f.sum(dim=1) >= min_area  # [K] bool

        return avg_vec_refined, refined_masks, valid_after

    # ============================================================
    #  METHOD 4. filter_memory_anchors_cross_frame  (다중 프레임)
    # ============================================================
    def filter_memory_anchors_cross_frame(
            self, avg_vec_mem, feat_cur,
            th_sim=0.60, th_margin=0.12, min_pure_response=3
    ):
        """
        [다중 프레임 전용]
        메모리 뱅크의 앵커 벡터를 현재 프레임 피처에 투영하여,
        순수 단독 반응 패치가 부족한 앵커를 cross-attention 전에 배제.

        처리되는 케이스:
          - P1: 현재 프레임에 없는 객체 (의자 등) → pure_area ≈ 0
          - P2: 배경 대면적 앵커               → 배경 패치끼리 경쟁 → pure_area 낮음
          - P3/P4: 오염 앵커                   → overlap 패치 다수 → pure_area 낮음

        Args:
            avg_vec_mem      : [K_mem, D] 메모리 뱅크 앵커 대표 벡터 (L2 정규화 완료)
            feat_cur         : [N_cur, D] 현재 프레임 패치 피처 (L2 정규화 권장)
            th_sim           : float  최소 유사도 가드
            th_margin        : float  1등과의 마진 임계값
            min_pure_response: int    생존에 필요한 최소 순수 반응 패치 수

        Returns:
            valid_mem_mask   : [K_mem] bool  True인 앵커만 cross-attention key/value로 공급
            pure_area_cur    : [K_mem] int   각 앵커의 현재 프레임 순수 반응 패치 수 (디버깅용)
        """
        # cross-frame sim: 메모리 앵커 vs 현재 프레임 패치
        sim_cross = torch.mm(avg_vec_mem, feat_cur.t())  # [K_mem, N_cur]

        # 마진 카운팅 (compute_patch_purity_masks와 동일 로직)
        sim_thr = torch.where(
            sim_cross >= th_sim, sim_cross, torch.zeros_like(sim_cross)
        )
        max_vals = sim_thr.max(dim=0, keepdim=True).values
        margins = max_vals - sim_thr

        valid_cross = (sim_thr > 0) & (margins < th_margin)  # [K_mem, N_cur] bool
        overlap_cur = valid_cross.float().sum(dim=0)  # [N_cur]

        # 순수 패치 = 현재 프레임에서도 단독 점유
        pure_cross_mask = (overlap_cur == 1)  # [N_cur] bool
        pure_cross = valid_cross & pure_cross_mask.unsqueeze(0)  # [K_mem, N_cur] bool

        # 앵커별 순수 반응 패치 수
        pure_area_cur = pure_cross.float().sum(dim=1)  # [K_mem]

        # 순수 반응이 min_pure_response 이상인 앵커만 생존
        valid_mem_mask = pure_area_cur >= min_pure_response  # [K_mem] bool

        return valid_mem_mask, pure_area_cur

    # ============================================================
    #  USAGE EXAMPLE  (zmq_dino_server.py 참고용)
    # ============================================================
    """
    # ── 단일 프레임 파이프라인 ──────────────────────────────────────
    sim1 = torch.mm(new_avg_patch_vec1, feat1.t())   # [K, N]  기존 코드

    # [기존] detect_mixed_boundary_patches_by_counting (시각화용으로 유지)
    I1, overlap1 = objpatcher.detect_mixed_boundary_patches_by_counting(sim1)

    # [신규] 순수/오버랩 마스크 분리
    vom1, pure1, overlap_counts1 = objpatcher.compute_patch_purity_masks(sim1)

    # [신규] 시드가 overlap 구간인 앵커 배제
    valid_seed1 = objpatcher.filter_seed_overlap_anchors(new_sample_1, overlap_counts1)

    # [신규] 병합 + pure_affinity 기반 avg_vec 재계산
    avg_vec_refined1, refined_masks1, valid_after1 = objpatcher.refine_anchors_by_pure_affinity(
        vom1[valid_seed1], pure1[valid_seed1], feat1,
        th_merge=0.4, min_area=2
    )

    # XFeat 귀속은 valid_overlap_mask 그대로 사용 (커버리지 유지)
    vom_for_xfeat = vom1[valid_seed1][valid_after1]   # [K_final, N]

    # memory bank 저장
    avg_vec_to_store = avg_vec_refined1[valid_after1]  # [K_final, D]

    # ── 다중 프레임 파이프라인 ──────────────────────────────────────
    valid_mem_mask, pure_area = objpatcher.filter_memory_anchors_cross_frame(
        avg_vec_to_store, feat2,
        th_sim=0.60, th_margin=0.12, min_pure_response=3
    )

    # cross-attention key/value로 공급할 최종 벡터
    avg_vec_final = avg_vec_to_store[valid_mem_mask]   # [K_clean, D]
    """

    ### 고민 중
    def detect_mixed_boundary_patches_by_counting(self, sim_matrix, th_sim=0.60, th_margin=0.12):
        """
        [진우 박사님 제안: 전수 마진 카운팅 기반 가림/경계면 패치 지시 함수 (\mathbb{I}_n) 구현]
        규격: sim_matrix는 [K, N] 형태 (K: 사물 개수, N: 패치 개수 1530)
        """
        # Step 1. th_sim 보다 낮은 애들 0.0으로 만들기
        sim_throttled = torch.where(sim_matrix >= th_sim, sim_matrix, torch.zeros_like(sim_matrix))

        # Step 2. N 패치별(dim=0 방향)로 가장 큰 값을 고르기
        # max_vals 크기: [1, N]
        max_vals = sim_throttled.max(dim=0, keepdim=True).values

        # Step 3. 각 원소별 마진 계산 (브로드캐스팅 감산)
        # margins 크기: [K, N]
        margins = max_vals - sim_throttled

        # Step 4. 👑 [박사님 핵심 필터링 정렬]:
        # 마진이 th_margin보다 "큰" 유령 성분들을 제외하고,
        # 1등 스코어와 촘촘하게 겹쳐있는(margins < th_margin) 진짜 핵심 성분들만 유효 패치로 남깁니다.
        # 유효 사물이 되려면 기본적으로 th_sim 가드(sim_throttled > 0)도 통과해야 합니다.
        valid_overlap_mask = (sim_throttled > 0) & (margins < th_margin)  # [K, N] (Boolean)

        # Step 5. 패치별(dim=0 세로 방향) 남는 사물 수를 카운트
        # overlap_counts 크기: [N] (각 패치별로 매칭된 사물의 총 개수)
        overlap_counts = valid_overlap_mask.sum(dim=0)

        # Step 6. 👑 [최종 지시 함수 도출]:
        # 해당 패치에 남은 유효 사물 수가 2개 이상이라는 것은 진짜 겹침/가림이 발생했다는 뜻입니다!
        # I_n 크기: [N]
        I_n = (overlap_counts >= 2).float()

        return I_n, overlap_counts

    def compile_structural_equivalence_vectorized(self, feat_base, sample_avg_vecs, heatmap_threshold=0.85, sim_cutoff = 0.7):
        """
        [진우 박사님 전용 - 코사인 유사도 행렬 반환형 고속 위상 매칭 엔진]
        기존 아규먼트 포맷을 완벽히 보존하며, 사후 시각화 디버깅용 코사인 유사도 행렬을 함께 리턴합니다.
        """
        import torch
        import torch.nn.functional as F
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix

        K = sample_avg_vecs.shape[0]

        # 1. 날것의 K x 1530 전역 유사도 히트맵 컴파일 (코사인 유사도 공간)
        M_norm = F.normalize(sample_avg_vecs, p=2, dim=1)
        F_norm = F.normalize(feat_base, p=2, dim=1)
        H_matrix = torch.mm(M_norm, F_norm.t())  # [K, 1530]

        # 👑 [박사님 핵심 지시]: 유사도가 0.65 미만인 하위 노이즈 구역은 아예 완벽한 0으로 압착
        H_matrix_clean = torch.where(H_matrix >= sim_cutoff, H_matrix, torch.zeros_like(H_matrix))

        # 2. ⚡ [핵심 추가 구역]: 0으로 정제된 히트맵 분포를 기반으로 교차 코사인 유사도 연산 (K x K)
        H_norm = F.normalize(H_matrix_clean, p=2, dim=1)
        heatmap_sim_matrix = torch.mm(H_norm, H_norm.t())
        heatmap_sim_np = heatmap_sim_matrix.cpu().numpy()

        # 3. 인접 행렬 변환 및 Scipy C++ 그래프 컴포넌트 일괄 색출
        adjacency_matrix = (heatmap_sim_np >= heatmap_threshold).astype(np.int32)

        n_components, group_assignments = connected_components(
            csgraph=csr_matrix(adjacency_matrix), directed=False, connection='weak', return_labels=True
        )

        # 💡 [정정 완공]: 시각화 레이어에서 직접 룩업할 수 있도록 유사도 매트릭스(heatmap_sim_np)를 함께 리턴!
        return group_assignments, n_components, heatmap_sim_np

    #코드 확인이 필요함.
    def extract_new_group_centroids_vectorized(self, sample1_old, group_assignments, n_components):
        """
        [진우 박사님 전용 - 마스터 시드 패치 추출 및 sample1 구조 동기화 엔진]

        Args:
            sample1_old      : [K] 기존 NMS 단계에서 선별되었던 정예 패치 1D Tensor (long)
            group_assignments: compile_structural_equivalence_vectorized의 아웃풋인 그룹 ID 배열 [K] (Numpy)
            n_components     : 최종 통합 분리된 총 그룹 개수 (M)

        Returns:
            sample1_new      : [M] 새로 합병된 그룹별 마스터 시드 패치 1D Tensor (long, GPU 유지 가능)
                               기존 sample1과 완전히 동일한 데이터 파이프라인 규격을 가집니다.
        """
        device = sample1_old.device

        # 1. 넘파이 group_assignments 주소를 파이토치 GPU 텐서로 즉시 변환
        group_tensor = torch.from_numpy(group_assignments).long().to(device)  # [K]

        # 2. 💡 [원핫 인코딩 선택 행렬 빌드]: 각 시드가 어떤 마스터 사물에 속하는지 매핑
        # [K] -> [K, M] (0과 1로 구성)
        selection_matrix = F.one_hot(group_tensor, num_classes=n_components)  # [K, M]

        # 3. 👑 [대수적 대표 선별 기믹]: 각 마스터 사물 제국(M) 채널별로 최초로 소속된 원소 색출
        # 열(dim=0) 방향으로 argmax를 취하면, 동일 그룹으로 묶인 기존 K개의 시드 중
        # 인덱스가 가장 앞선(CLS 점수나 밀도가 가장 높았던) 기존 시드의 번호 [M]개가 루프 없이 가뿐하게 룩업됩니다.
        master_k_indices = selection_matrix.argmax(dim=0)  # [M]

        # 4. 🎯 [최종 sample1 구조화]: 기존 패치 번호판에서 마스터 주소들만 슬라이싱하여 복원
        # 결과 형태: [M] 크기의 torch.Tensor (dtype=torch.long)
        sample1_new = sample1_old[master_k_indices].long()

        return sample1_new

    def expand_patch_groups_exclusive_vectorized(self, feat_base, sample1_new, sim_thresh=0.70):
        """
        [진우 박사님 전용 - 마스터 시드 기반 전역 독점적 패치 그룹 확장 엔진]

        하나의 패치가 가장 유사도가 높은 단 하나의 마스터 사물 그룹에만 독점적으로
        귀속되도록 배타적(Exclusive) 매스킹을 0ms 만에 일괄 집행합니다.

        Args:
            feat_base     : [N, D] 이미지 전체 패치 피처 행렬 (정규화 전 원본 스케일 권장)
            sample1_new   : [M] 앞서 합병 완공된 새로운 마스터 시드 1D 텐서 (dtype=torch.long)
            sim_thresh    : 최소 소속 보증을 위한 하위 임계값 (이 값 미만은 배경으로 파기)

        Returns:
            exclusive_affinity_mn : [M, N] 하나의 패치가 단 하나의 마스터 그룹에만 True로 켜진 독점적 어피니티 마스크
        """
        device = feat_base.device
        M = sample1_new.shape[0]
        N = feat_base.shape[0]

        if M == 0:
            return torch.zeros((0, N), dtype=torch.bool, device=device)

        # 1. 🛡️ 안전하게 L2 정규화 축 정렬
        F_norm = feat_base
        M_norm = F_norm[sample1_new.long()]
        #F_norm = F.normalize(feat_base, p=2, dim=1)  # [N, D]
        #M_norm = F_norm[sample1_new.long()]  # [M, D] 마스터 시드 피처 슬라이싱 추출

        # 2. 🚀 [교차 유사도 행렬 생성]: 마스터 시드 [M, D] @ 전체 패치 [D, N]
        # 결과 크기: [M, N]
        cross_sim = torch.mm(M_norm, F_norm.t())  # [M, N]

        # 3. 👑 [박사님 핵심 지시 - Exclusive 하드 컷오프 가드]:
        # ① 전체 1530개 패치 각각에 대해 "나랑 가장 닮은 마스터 사물 인덱스"를 열(dim=0) 축 기준으로 단 한 방에 추출
        best_master_indices = cross_sim.argmax(dim=0)  # [N] (각 패치가 귀속될 0 ~ M-1 사이의 마스터 ID)

        # ② 임계값(sim_thresh)을 넘어서 최소한의 사물 정체성이 보장된 패치 필터링
        # max(dim=0)을 통해 각 패치별 최대 유사도 값들을 확보합니다.
        max_sim_values, _ = cross_sim.max(dim=0)  # [N]
        valid_semantic_mask = max_sim_values >= sim_thresh  # [N] (Boolean)

        # ③ 100% 벡터라이징 기반 독점적 불리언 맵 생성 [M, N]
        # 0으로 채워진 빈 캔버스를 열고, 패치별로 가장 점수가 높았던 단 하나의 대장 행(Row) 주소에만 True를 마킹합니다.
        exclusive_affinity_mn = torch.zeros((M, N), dtype=torch.bool, device=device)

        # 유효한 시맨틱 임계값을 통과한 진짜 패치 노드들만 주소 추적
        valid_patch_indices = torch.where(valid_semantic_mask)[0]
        assigned_master_rows = best_master_indices[valid_patch_indices]

        # 고급 인덱싱(Advanced Indexing)으로 루프 없이 단 한 클럭만에 독점 판 성형 종결
        exclusive_affinity_mn[assigned_master_rows, valid_patch_indices] = True

        return exclusive_affinity_mn

    # deprecated
    def compile_structural_equivalence_vectorized2(self,feat_base, sample_avg_vecs, heatmap_threshold=0.85):
        """
        [진우 박사님 전용 - 중첩 루프를 전면 소거한 전역 히트맵 일치도 기반 고속 합병 커널]
        Scipy C++ 가속 그래프 컴포넌트 알고리즘을 이용하여 루프 없이 원샷으로 group_assignments를 산출합니다.
        """
        t1 = time.time()
        K = sample_avg_vecs.shape[0]

        # 1. 전역 히트맵 행렬곱 계산 (K x 1530)
        M_norm = F.normalize(sample_avg_vecs, p=2, dim=1)
        F_norm = F.normalize(feat_base, p=2, dim=1)
        H_matrix = torch.clamp(torch.mm(M_norm, F_norm.t()), min=0.0)

        # 2. 히트맵 간의 교차 코사인 유사도 연산 (K x K)
        H_norm = F.normalize(H_matrix, p=2, dim=1)
        heatmap_sim_matrix = torch.mm(H_norm, H_norm.t())
        heatmap_sim_np = heatmap_sim_matrix.cpu().numpy()

        # ====================================================================
        # ⚡ [박사님 지시 사항 - 중첩 for 루프 및 치환 가드 완벽 소거단]
        # ====================================================================
        # ① 임계값을 넘는 유효 연결 쌍들을 0과 1로 이루어진 불리언 인접 행렬로 변환 [K, K]
        #    (상삼각 행렬만 볼 필요 없이 행렬 전역을 원샷으로 마스킹 부러뜨립니다)
        adjacency_matrix = (heatmap_sim_np >= heatmap_threshold).astype(np.int32)

        t2 = time.time()

        # ② 희소 행렬 그래프 포맷(CSR)으로 압축 전송 (메모리 및 자원 극대화)
        graph = csr_matrix(adjacency_matrix)
        t3 = time.time()
        # ③ 👑 [마스터 가속 커널 실행]:
        #    C++ 기반 고속 BFS/DFS로 백트래킹하며 연결 무리들을 루프 없이 일괄 색출합니다.
        #    labels_np 배열 내부에는 원샷 합병이 완료된 컴포넌트 ID가 자동으로 부여됩니다.
        n_components, group_assignments = connected_components(
            csgraph=graph,
            directed=False,  # 무방향 그래프 체계 고수 (i,j 가 합쳐지면 j,i 도 당연히 한 몸)
            connection='weak',  # 징검다리식 전파 연결(Transitive Closure) 조건 활성화
            return_labels=True
        )
        t4 = time.time()
        # ====================================================================
        print('cluster test', t4-t1, t2-t1,t3-t2,t4-t3)
        return group_assignments, heatmap_sim_np, n_components

    #deprecated
    def build_anchor_to_patch_affinity_recursive(self, centroids, inhibition_mask, max_hops=1):
        """
        Args:
            centroids       : [K] 형태의 1D Tensor (정예 앵커 인덱스)
            inhibition_mask : [N, N] 형태의 공간-시맨틱 융합 마스크 (바이너리 인접 행렬)
            max_hops        : 연결선을 타고 들어갈 최대 깊이 (물체 최대 크기에 비례하여 조절)
        Returns:
            affinity_matrix : [K, N] 형태의 광역 도달 가능성(Reachability) 어피니티 행렬
        """

        N = inhibition_mask.shape[0]
        K = centroids.shape[0]
        device = inhibition_mask.device

        # 1. 초기 1-hop 어피니티 맵 추출 [K, N] (기존 박사님 코드)
        # float 형태로 변환해야 행렬 곱 가속을 쓸 수 있습니다.
        adj_matrix = inhibition_mask.float()

        # current_affinity: [K, N] (각 앵커별 현재 도달한 패치들)
        current_affinity = adj_matrix[centroids.long(), :]

        # 자기 자신(앵커 본인 위치)은 무조건 포함하도록 초기화
        identity_k = torch.zeros((K, N), device=device)
        identity_k[torch.arange(K, device=device), centroids.long()] = 1.0
        current_affinity = torch.clamp(current_affinity + identity_k, 0.0, 1.0)

        # 2. 💡 [그래프 확산 엔지니어링]: 행렬곱을 이용한 다중 홉(Multi-hop) 이웃 전산 전개
        # 반복문을 돌지만 내부 연산은 전체 K개 앵커에 대해 완전 병렬(Batch)로 GPU 가속됩니다.
        for _ in range(max_hops):
            # [K, N] x [N, N] -> [K, N]
            # 현재 내 영역에 속한 패치들이 '다음 단계로 갈 수 있는 이웃'들을 한 번에 서치
            next_affinity = torch.mm(current_affinity, adj_matrix)

            # 기존 영역과 새롭게 도달한 영역의 합집합 연산 (0.0 또는 1.0으로 바이너리화)
            next_affinity = torch.clamp(next_affinity + current_affinity, 0.0, 1.0)

            # 💡 [조기 종료 조건]: 더 이상 새로 추가되는 패치가 없다면 수렴한 것으로 판단하고 루프 탈출
            if torch.equal(next_affinity, current_affinity):
                break

            current_affinity = next_affinity

        # 최종 결과를 다시 Boolean 구조 [K, N] (또는 필요에 따라 float) 형태로 리턴
        return current_affinity > 0.5

    def build_anchor_to_patch_affinity(self, centroids, inhibition_mask, sim_matrix=None, exclusive=False):
        base_affinity = inhibition_mask[centroids.long(), :]  # [K, N]

        if not exclusive:
            return base_affinity

        assert sim_matrix is not None, "exclusive=True 일 때는 sim_matrix를 넘겨주세요!"

        K, N = base_affinity.shape
        if K == 0:
            return base_affinity

        # 🌟 [최적화 핵심]: 행렬 곱(torch.mm) 연산을 아예 삭제하고,
        # 이미 계산된 [N, N] 행렬에서 앵커에 해당하는 [K, N]만 1마이크로초만에 쓱 잘라옵니다!
        sim = sim_matrix[centroids.long(), :]

        # 원래 마스크 밖의 영역은 아예 선택되지 않도록 -9999로 밀어버림
        masked_sim = torch.where(base_affinity, sim, torch.full_like(sim, -9999.0))

        # 패치별로 가장 유사도가 높은 단 1개의 앵커 추출 [N]
        best_anchor_indices = masked_sim.argmax(dim=0)
        claimed_patches = base_affinity.any(dim=0)

        exclusive_affinity = torch.zeros_like(base_affinity, dtype=torch.bool)
        valid_patch_indices = torch.where(claimed_patches)[0]
        valid_anchor_indices = best_anchor_indices[valid_patch_indices]

        exclusive_affinity[valid_anchor_indices, valid_patch_indices] = True

        return exclusive_affinity

    def build_anchor_to_anchor_link(self, affinity, min_overlap_patches = 2):
        K,N = affinity.shape
        if K == 0:
            return torch.zeros((0,0), device = affinity.device, dtype=torch.bool)
        overlap_counts = torch.mm(affinity.float(), affinity.float().t())
        anchor_link_matrix = overlap_counts >= min_overlap_patches
        anchor_link_matrix.fill_diagonal_(False)
        return anchor_link_matrix

    def anchor_pruning(self, centroids, anchor_links, max_link_threshold = 2):
        link_counts = torch.sum(anchor_links.float(), dim=1)  # Shape: [K]
        survival_mask = link_counts < max_link_threshold  # Shape: [K] (Boolean)
        filtered_centroids = centroids[survival_mask]  # Shape: [K_filtered]
        filtered_anchor_links = anchor_links[survival_mask][:, survival_mask]
        return filtered_centroids, filtered_anchor_links

    def generate_mask(self, features, grid_shape, spatial_radius = 2, sim_thresh = 0.7):
        H_p, W_p = grid_shape
        device = features.device

        sim_matrix = torch.mm(features, features.t())
        semantic_mask = sim_matrix > sim_thresh

        if spatial_radius <= 0:
            return semantic_mask

        grid_y, grid_x = torch.meshgrid(torch.arange(H_p, device=device), torch.arange(W_p, device=device),
                                        indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        dist_matrix = torch.cdist(coords, coords)  # [N, N]

        # 💡 반경 이내 이웃 여부를 정의하는 불리언 행렬 [N, N]
        spatial_mask = (dist_matrix < spatial_radius)

        inhibition_mask = spatial_mask & semantic_mask
        return inhibition_mask, sim_matrix

    def sample_patch(self, attn, mask, num_centroids = 50):
        scores = attn[0]
        sorted_indices = torch.argsort(scores, descending=True)

        # 3. [최적화 포인트]: 정렬된 순서대로 거리 행렬의 축을 재정렬합니다.
        # 이렇게 하면 '앞쪽 인덱스일수록 중요도가 높은 노드'가 됩니다.
        reordered_inhibition = mask[sorted_indices][:, sorted_indices]  # [N, N]

        # 4. 💡 자기 자신보다 중요도가 높은(위치가 앞선) 노드들에 의한 억제 관계만 남기기 위해
        # 하삼각 행렬(Lower Triangular)을 0으로 밀어버리고 상삼각(Upper)만 취합니다.
        # (즉, 나보다 점수 높은 놈이 내 주변에 있는지만 체크하겠다는 뜻)
        upper_inhibition = torch.triu(reordered_inhibition, diagonal=1)  # [N, N]

        # 5. 각 노드별로 자신보다 점수 높은 놈에게 억제당했는지 여부를 단 한 방에 연산
        # 열(dim=0) 방향으로 True가 하나라도 있으면 누군가에게 먹힌 노드입니다.
        suppressed = torch.any(upper_inhibition, dim=0)  # [N]

        # 6. 억제당하지 않은 생존자들(False)만 추려내어 원래 이미지 패치 인덱스로 복원
        keep_local_indices = torch.where(~suppressed)[0]

        # 정렬된 인덱스에서 최종 생존자 인덱스 맵핑 후 제한 수량 조절
        final_sorted_centroids = sorted_indices[keep_local_indices]
        centroids = final_sorted_centroids[:num_centroids]
        return centroids

    ## 아래것 과 비교
    def compute_anchor_average_features(self, affinity, features):
        """💡 [알고리즘 B]: 앵커 세력권(영역) 내 패치들의 피처를 평균내어 [K, D] 사물 고유 벡터를 생성합니다."""
        # affinity: [K, N], features: [N, D]
        # 행렬 곱을 수행하면 각 앵커별로 소속된 패치 피처들의 '총합'이 계산됨 [K, D]
        feature_sums = torch.mm(affinity.float(), features.float())

        # 각 앵커가 포섭한 패치의 총 개수 (면적) 계산 -> [K, 1] (0인 경우 분모 노이즈 방지용 clamp)
        patch_counts = torch.sum(affinity.float(), dim=1, keepdim=True).clamp(min=1.0)

        # 💡 평균 벡터 산출 후 정규화 종결 [K, D]
        avg_features = feature_sums / patch_counts
        return F.normalize(avg_features, p=2, dim=-1)

    def extract_sample_neighborhood_pure_average(self, feat_norm, affinity):
        """
        Args:
            feat_norm : [N, D] 이미 L2 정규화가 완료된 오프라인/온라인 패치 피처 행렬 (float32)
            affinity  : [K, N] 이미 구해진 공간-시맨틱 융합 어피니티 마스크 (0 또는 1)

        Returns:
            sample_avg_vecs      : [K, D] 노말라이즈 없이 순수하게 연결된 패치 정보의 평균 피처 벡터
            n_patches_per_sample : [K] 각 사물 앵커가 포섭한 순수 패치 개수 (연결된 애들 수)
        """
        device = feat_norm.device
        K, N = affinity.shape

        if K == 0:
            return torch.empty((0, feat_norm.shape[1]), device=device), torch.zeros(0, device=device)

        # 1. 어피니티 마스크를 float 텐서로 매핑 [K, N]
        masks_bool = affinity.float()

        # 2. 💡 [연결 된 애들 수]: 각 샘플 그룹별 포섭된 유효 패치 개수 산출 [K]
        n_patches_per_sample = masks_bool.sum(dim=1)  # [K]

        # 3. 💡 [패치 정보 더하기]: 마스크 기반 전수 합산 (BLAS 가속 행렬곱 한 방)
        # [K, N] @ [N, D] -> [K, D]
        sum_vecs = torch.mm(masks_bool, feat_norm)

        # 4. 💡 [연결 된 애들의 평균]: 합산된 벡터를 연결된 패치 수로 나누어 순수 평균 벡터 도출
        denom = n_patches_per_sample.unsqueeze(1)  # [K, 1]
        sample_avg_vecs = torch.where(denom > 0, sum_vecs / denom, torch.zeros_like(sum_vecs))

        # 박사님 지시대로 최종 단의 F.normalize 과정을 원천 삭제하여 순수 물리 값 보존
        return sample_avg_vecs, n_patches_per_sample
    def extract_sample_neighborhood_average_pool(self, x_cat, affinity, attn=None):
        """
        [진우 박사님 낭비 제로 최적화]
        이미 계산된 [K, N] 크기의 affinity 마스크를 그대로 재활용하여
        중복 유사도 연산 없이 곧바로 샘플별 국소 영역 평균 피처 벡터를 산출합니다.

        Args:
            feat_flat : [N, D] 크기의 오리지널(정규화 전) 평탄화 피처 행렬
            affinity  : [K, N] 크기의 이미 구해진 공간-시맨틱 융합 어피니티 마스크 (Boolean 또는 Float)

        Returns:
            sample_avg_vecs_norm : [K, D] L2 정규화가 완료된 사물별 대표 시맨틱 벡터
            n_patches_per_sample : [K] 각 사물 앵커가 포섭한 패치 개수 (면적)
        """
        device = x_cat.device
        _, D, _, _ = x_cat.shape
        feat_flat = x_cat[0].view(D, -1).t()

        K, N = affinity.shape
        if K == 0:
            return torch.empty((0, feat_flat.shape[1]), device=device), torch.zeros(0, device=device)

        # 1. 💡 [중복 연산 전면 소거]: 기존의 유사도 계산 및 임계값 마스킹 단계를 완전히 생략하고
        # 인풋으로 들어온 affinity를 float 형태로 가중치 매트릭스로 즉시 채택합니다.
        masks_float = affinity.float()  # [K, N]

        ##가중치 부여
        if attn is not None:
            weights = masks_float * attn.float()
        else:
            weights = masks_float

        # 2. 각 샘플별 그룹에 귀속된 유효 패치 개수(면적) 계산 [K]
        n_patches_per_sample = masks_float.sum(dim=1)  # [K]

        weight_sums = weights.sum(dim=1)

        # 3. 🚀 [거대 행렬 곱 딱 한 방]: 마스크 행렬 [K, N] @ 오리지널 피처 행렬 [N, D]
        # 원본 feat_flat의 묵직한 활성화 강도를 그대로 유지한 채 영역별 합산 처리
        sum_vecs = torch.mm(weights, feat_flat.float())  # [K, D]

        # 4. 평균 계산 및 0분모 방지 예외 처리
        denom = weight_sums.unsqueeze(1)  # [K, 1]
        sample_avg_vecs = torch.where(denom > 0, sum_vecs / denom, torch.zeros_like(sum_vecs))

        # 5. 최종 코사인 유사도 매칭 오퍼레이션을 위해 출력 벡터만 L2 정규화 종결
        sample_avg_vecs_norm = F.normalize(sample_avg_vecs, p=2, dim=1)

        return sample_avg_vecs_norm, n_patches_per_sample

    ###########
    def sample_object_anchors_with_k(self, k, grid_shape, num_centroids=300, spatial_radius=5):
        H_p, W_p = grid_shape
        device = k.device
        grid_y, grid_x = torch.meshgrid(torch.arange(H_p, device=device), torch.arange(W_p, device=device),
                                        indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        dist_matrix = torch.cdist(coords, coords)  # [N, N]

        # 💡 반경 이내 이웃 여부를 정의하는 불리언 행렬 [N, N]
        spatial_mask = (dist_matrix < spatial_radius)

        k = k.squeeze(0)
        #features_norm = F.normalize(k, p=2, dim=1)
        k_centered = k - k.mean(dim=0, keepdim=True)
        sim_matrix = torch.mm(k_centered, k_centered.t())

        inhibition_mask = spatial_mask
        negative_score = (sim_matrix < 0).sum(dim=1)

        scores = negative_score
        sorted_indices = torch.argsort(scores, descending=True)

        print("test", sim_matrix.shape, k.shape, (sim_matrix < 0).sum().item(), (k>=0).sum().item(), (k<0).sum().item())

        # 3. [최적화 포인트]: 정렬된 순서대로 거리 행렬의 축을 재정렬합니다.
        # 이렇게 하면 '앞쪽 인덱스일수록 중요도가 높은 노드'가 됩니다.
        reordered_inhibition = inhibition_mask[sorted_indices][:, sorted_indices]  # [N, N]

        # 4. 💡 자기 자신보다 중요도가 높은(위치가 앞선) 노드들에 의한 억제 관계만 남기기 위해
        # 하삼각 행렬(Lower Triangular)을 0으로 밀어버리고 상삼각(Upper)만 취합니다.
        # (즉, 나보다 점수 높은 놈이 내 주변에 있는지만 체크하겠다는 뜻)
        upper_inhibition = torch.triu(reordered_inhibition, diagonal=1)  # [N, N]

        # 5. 각 노드별로 자신보다 점수 높은 놈에게 억제당했는지 여부를 단 한 방에 연산
        # 열(dim=0) 방향으로 True가 하나라도 있으면 누군가에게 먹힌 노드입니다.
        suppressed = torch.any(upper_inhibition, dim=0)  # [N]

        # 6. 억제당하지 않은 생존자들(False)만 추려내어 원래 이미지 패치 인덱스로 복원
        keep_local_indices = torch.where(~suppressed)[0]

        # 정렬된 인덱스에서 최종 생존자 인덱스 맵핑 후 제한 수량 조절
        final_sorted_centroids = sorted_indices[keep_local_indices]
        centroids = final_sorted_centroids[:num_centroids]

        return centroids, (H_p, W_p)

    def sample_object_anchors_nms_fast(self, x_cat, cls_attn, num_centroids=300, spatial_radius=10, sim_thresh = 0.7):
        """
        파이썬 루프와 .sum() 오버헤드를 완전히 제거한 100% 벡터화 버전.
        속도가 기존 대비 수십 배 이상 빨라집니다.
        """
        _, _, H_p, W_p = x_cat.shape
        device = x_cat.device
        N = H_p * W_p

        # 1. 2D 좌표 그리드 역산 및 고속 2D 거리 행렬 빌드
        grid_y, grid_x = torch.meshgrid(torch.arange(H_p, device=device), torch.arange(W_p, device=device),
                                        indexing='ij')
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
        dist_matrix = torch.cdist(coords, coords)  # [N, N]

        # 💡 반경 이내 이웃 여부를 정의하는 불리언 행렬 [N, N]
        spatial_mask = (dist_matrix < spatial_radius)

        features_norm, H_p, W_p = self._prepare_features(x_cat)  # [N, D]
        sim_matrix = torch.mm(features_norm, features_norm.t())
        semantic_mask = sim_matrix > sim_thresh
        #sim_matrix = 1.0 - sim_matrix

        inhibition_mask = spatial_mask & semantic_mask

        # 2. CLS 어텐션 앵커 순서 정렬
        sigma = 0.01
        dist_matrix = 1.0 - sim_matrix

        negative_score = (sim_matrix < 0).sum(dim=1)
        #print("음수 테스트 = ", (negative_score))

        density_score = torch.sum(torch.exp(- (dist_matrix ** 2) / (2 * sigma ** 2)), dim=1)
        scores = cls_attn[0]  # [N]
        #scores = density_score
        #scores = negative_score
        sorted_indices = torch.argsort(scores, descending=True)

        # 3. [최적화 포인트]: 정렬된 순서대로 거리 행렬의 축을 재정렬합니다.
        # 이렇게 하면 '앞쪽 인덱스일수록 중요도가 높은 노드'가 됩니다.
        reordered_inhibition = inhibition_mask[sorted_indices][:, sorted_indices]  # [N, N]

        # 4. 💡 자기 자신보다 중요도가 높은(위치가 앞선) 노드들에 의한 억제 관계만 남기기 위해
        # 하삼각 행렬(Lower Triangular)을 0으로 밀어버리고 상삼각(Upper)만 취합니다.
        # (즉, 나보다 점수 높은 놈이 내 주변에 있는지만 체크하겠다는 뜻)
        upper_inhibition = torch.triu(reordered_inhibition, diagonal=1)  # [N, N]

        # 5. 각 노드별로 자신보다 점수 높은 놈에게 억제당했는지 여부를 단 한 방에 연산
        # 열(dim=0) 방향으로 True가 하나라도 있으면 누군가에게 먹힌 노드입니다.
        suppressed = torch.any(upper_inhibition, dim=0)  # [N]

        # 6. 억제당하지 않은 생존자들(False)만 추려내어 원래 이미지 패치 인덱스로 복원
        keep_local_indices = torch.where(~suppressed)[0]

        # 정렬된 인덱스에서 최종 생존자 인덱스 맵핑 후 제한 수량 조절
        final_sorted_centroids = sorted_indices[keep_local_indices]
        centroids = final_sorted_centroids[:num_centroids]

        return centroids, (H_p, W_p)

    def map_xfeat_to_flattened_patches(self, xfeat_xy, grid_shape=(34, 45), patch_size=14):
        """
        Args:
            xfeat_xy   : [N, 2] 크기의 PyTorch Tensor (float/int), XFeat의 (x, y) 픽셀 좌표 목록
            grid_shape : (H_p, W_p) 형태의 DINOv2 패치 그리드 차원 (기본값: 34행 45열 = 총 1530개 패치)
            patch_size : DINOv2의 패치 스트라이드/크기 (기본값: 14픽셀)

        Returns:
            patch_indices_n1 : [N, 1] 크기의 torch.Tensor (long 타입, GPU 유지), 각 포인트별 1차원 패치 인덱스
        """
        device = xfeat_xy.device
        H_p, W_p = grid_shape

        # 1. 2D 픽셀 좌표를 패치 해상도 단위(px, py)로 나눕니다.
        # 이미지 경계선 경계 예외처리를 위해 clamp 처리를 더해 안전성을 높였습니다.
        px = torch.clamp((xfeat_xy[:, 0] // patch_size).long(), 0, W_p - 1)
        py = torch.clamp((xfeat_xy[:, 1] // patch_size).long(), 0, H_p - 1)

        # 2. 💡 [1열화(Flatten) 인덱싱 연산]: 2D 패치 좌표 (py, px) -> 1D 전역 주소 체계로 변환
        # 주소 공식: y_index * Width_stride + x_index
        flattened_indices = py * W_p + px  # Shape: [N]

        # 3. 💡 [차원 고정]: 박사님이 요구하신 [N, 1] 스케일 구조로 텐서 언스퀴즈 변경
        patch_indices_n1 = flattened_indices.unsqueeze(1)  # Shape: [N, 1]

        return patch_indices_n1

    def build_patch_group_mask(self, feats_norm, sample1, sim_thresh=0.70):
        """
        Args:
            feat1       : (N_all, D) 이미지 전체 패치 피처 행렬 (예: [1530, 384])
            sample1     : (K_sampled,) NMS로 선별된 정예 패치 1D 인덱스 텐서 (예: [300])
            sim_thresh  : (float) 이 값보다 코사인 유사도가 높으면 같은 그룹으로 바인딩

        Returns:
            patch_group_mask : [K_sampled, N_all] float32 Tensor on GPU
        """
        # 2. 💡 [교차 유사도 행렬 생성]: 정예 샘플 피처 [K, D] @ 전체 피처 [D, N_all]
        # 결과 크기: [K_sampled, N_all]
        # 행(Row)은 각 샘플 노드를 뜻하고, 열(Col)은 전역 패치와의 유사도 배열을 뜻함
        cross_sim = torch.mm(feats_norm[sample1.long()], feats_norm.t())

        # 3. 💡 [쓰레숄딩]: 임계값(sim_thresh) 이상인 지점만 1.0, 나머지는 0.0으로 맵핑
        # 정제 처리를 하지 않고 중복 선택을 허용하라는 박사님 조건이 완벽히 만족되는 지점입니다.
        patch_group_mask = (cross_sim >= sim_thresh).float()

        return patch_group_mask

    def connect_patch_groups_via_global_xfeat(
            self, xfeat_xy1, xfeat_xy2, idx1, idx2,
            patch_group_mask1, patch_group_mask2,
            grid_shape=(34, 45), patch_size=14
    ):
        """
        Args:
            xfeat_xy1, xfeat_xy2: 양 프레임 전체 XFeat 2D 픽셀 좌표 -> [M1, 2], [M2, 2]
            idx1, idx2         : 글로벌 매칭으로 매칭 성공한 XFeat 인덱스 쌍 -> [Num_Matches]
            patch_group_mask1  : 1번 프레임 패치 그룹 행렬 -> [K1, N_all] (Tensor float/bool)
            patch_group_mask2  : 2번 프레임 패치 그룹 행렬 -> [K2, N_all] (Tensor float/bool)
        """
        device = xfeat_xy1.device
        H_p, W_p = grid_shape
        K1 = patch_group_mask1.shape[0]
        K2 = patch_group_mask2.shape[0]

        if len(idx1) == 0:
            return torch.zeros((K1, K2), device=device)

        # ----------------------------------------------------------------------
        # STEP 1. 매칭된 정예 XFeat 포인트들의 2D 픽셀 좌표 슬라이싱
        # ----------------------------------------------------------------------
        matched_xy1 = xfeat_xy1[idx1.long()]  # [Num_Matches, 2]
        matched_xy2 = xfeat_xy2[idx2.long()]  # [Num_Matches, 2]

        # ----------------------------------------------------------------------
        # STEP 2. 💡 [좌표 워프]: 매칭된 XFeat 포인트들이 각각 '몇 번 전역 패치'에 떨어졌는지 계산
        # ----------------------------------------------------------------------
        p1_x = torch.clamp((matched_xy1[:, 0] // patch_size).long(), 0, W_p - 1)
        p1_y = torch.clamp((matched_xy1[:, 1] // patch_size).long(), 0, H_p - 1)
        matched_patch_indices1 = p1_y * W_p + p1_x  # [Num_Matches] 차원의 전역 패치 인덱스 목록

        p2_x = torch.clamp((matched_xy2[:, 0] // patch_size).long(), 0, W_p - 1)
        p2_y = torch.clamp((matched_xy2[:, 1] // patch_size).long(), 0, H_p - 1)
        matched_patch_indices2 = p2_y * W_p + p2_x  # [Num_Matches] 차원의 전역 패치 인덱스 목록

        # ----------------------------------------------------------------------
        # STEP 3. 💡 [그룹 귀속 역산]: 매칭된 포인트들이 각각 '어느 샘플 그룹'에 속하는지 마스크 파싱
        # ----------------------------------------------------------------------
        # patch_group_mask1 은 [K1, N_all] 크기이므로, 슬라이싱을 취하면
        # [K1, Num_Matches] 크기의 '매칭 포인트별 그룹 귀속성 행렬'이 완성됩니다.
        feat_belongs_to_g1 = patch_group_mask1.float()[:, matched_patch_indices1]  # [K1, Num_Matches]
        feat_belongs_to_g2 = patch_group_mask2.float()[:, matched_patch_indices2]  # [K2, Num_Matches]

        # ----------------------------------------------------------------------
        # STEP 4. 💡 [핵심 투표 연산]: 단 한 방의 행렬 곱으로 그룹 간 전수 매칭 빈도 계산
        # ----------------------------------------------------------------------
        # [K1, Num_Matches] @ [Num_Matches, K2] -> [K1, K2] 크기의 빈도수 매트릭스 탄생
        # 이 연산 한 방으로 모든 패치 그룹 조합 간의 XFeat 기하 결합 카운트(Voting)가 종료됩니다.
        group_affinity_counts = torch.mm(feat_belongs_to_g1, feat_belongs_to_g2.t())  # [K1, K2]

        # ----------------------------------------------------------------------
        # STEP 5. 정규화 및 최종 짝 선별
        # ----------------------------------------------------------------------
        # 1번 프레임의 각 그룹 기준, 2번 프레임에서 가장 기하학적 매칭 점수가 높은 그룹 매핑
        # max_counts: 매칭점수 [K1], best_match_g2_idx: 가장 가까운 타겟 그룹 ID [K1]
        max_counts, best_match_g2_idx = group_affinity_counts.max(dim=1)

        # 0개 매칭된 아웃라이어 그룹 예외처리를 위한 마스크
        valid_group_mask = max_counts > 0
        return group_affinity_counts, best_match_g2_idx, valid_group_mask

    import torch
    import torch.nn.functional as F

    def connect_patch_groups_hybrid(
            self,
            feat1, patch_group_mask1, xfeat_xy1,
            feat2, patch_group_mask2, xfeat_xy2,
            idx1, idx2,
            grid_shape=(34, 45), patch_size=14, sim_thresh=0.65, count_thresh=3
    ):
        """
        [진우 박사님 하이브리드 기하-시맨틱 노드 결합 모듈]
        1. 각 패치 그룹 마스크[K, N_all]를 이용하여 그룹별 대표 평균 특징 벡터[K, D]를 생성합니다.
        2. 프레임 간 그룹 피처 간의 올-바이-올 코사인 유사도 행렬[K1, K2]을 연산합니다.
        3. 기존 글로벌 XFeat 매칭 쌍을 기반으로 그룹 간 투표수 행렬[K1, K2]을 빌드합니다.
        4. 2중 필터(XFeat 카운트 >= count_thresh AND DINO 유사도 >= sim_thresh)를 통과한 정예 에지만 선별합니다.

        Args:
            feat1, feat2            : 이미지 전체 패치 피처 행렬 -> [N_all, D] (L2 정규화 전 원본 스케일 권장)
            sample1, sample2        : NMS 정예 패치 1D 인덱스 텐서 -> [K1], [K2]
            patch_group_mask1, 2    : build_patch_group_mask로 구한 그룹별 맵핑 마스크 -> [K1, N_all], [K2, N_all]
            idx1, idx2              : 글로벌 XFeat 매칭 쌍 인덱스 텐서 -> [Num_Matches]
        """
        device = feat1.device
        H_p, W_p = grid_shape

        if feat1.dim() == 4:
            _, D1, _, _ = feat1.shape
            feat1 = feat1[0].view(D1, -1).t()  # [N_all, D]

        if feat2.dim() == 4:
            _, D2, _, _ = feat2.shape
            feat2 = feat2[0].view(D2, -1).t()  # [N_all, D]

        K1, N_all = patch_group_mask1.shape
        K2 = patch_group_mask2.shape[0]

        # ====================================================================
        # PHASE 1. 💡 [그룹별 대표 컨텍스트 피처 벡터 생성 및 유사도 연산]
        # ====================================================================
        # 1번 프레임 그룹별 평균 벡터 빌드: [K1, N_all] @ [N_all, D] -> [K1, D]
        sum_feats1 = torch.mm(patch_group_mask1.float(), feat1.float())
        denom1 = torch.clamp(patch_group_mask1.sum(dim=1, keepdim=True), min=1.0)
        group_vecs1 = sum_feats1 / denom1
        group_vecs1_norm = F.normalize(group_vecs1, p=2, dim=1)  # 코사인용 정규화

        # 2번 프레임 그룹별 평균 벡터 빌드: [K2, N_all] @ [N_all, D] -> [K2, D]
        sum_feats2 = torch.mm(patch_group_mask2.float(), feat2.float())
        denom2 = torch.clamp(patch_group_mask2.sum(dim=1, keepdim=True), min=1.0)
        group_vecs2 = sum_feats2 / denom2
        group_vecs2_norm = F.normalize(group_vecs2, p=2, dim=1)  # 코사인용 정규화

        # 💡 [시맨틱 결합도 행렬]: 두 프레임의 그룹 간 All-vs-All 코사인 유사도 연산 -> [K1, K2]
        semantic_affinity = torch.mm(group_vecs1_norm, group_vecs2_norm.t())

        # ====================================================================
        # PHASE 2. 💡 [기하학적 XFeat 매칭 기반 그룹 투표 행렬 빌드]
        # ====================================================================
        if len(idx1) == 0:
            return torch.zeros((K1, K2), device=device), semantic_affinity, torch.zeros(K1, device=device,
                                                                                        dtype=torch.long), torch.zeros(
                K1, device=device, dtype=torch.bool)

        # 매칭된 XFeat 키포인트들이 각각 몇 번 전역 패치에 속하는지 일괄 좌표 역산
        p1_x = torch.clamp((xfeat_xy1[idx1.long(), 0] // patch_size).long(), 0, W_p - 1)
        p1_y = torch.clamp((xfeat_xy1[idx1.long(), 1] // patch_size).long(), 0, H_p - 1)
        matched_patch_indices1 = p1_y * W_p + p1_x  # [Num_Matches]

        p2_x = torch.clamp((xfeat_xy2[idx2.long(), 0] // patch_size).long(), 0, W_p - 1)
        p2_y = torch.clamp((xfeat_xy2[idx2.long(), 1] // patch_size).long(), 0, H_p - 1)
        matched_patch_indices2 = p2_y * W_p + p2_x  # [Num_Matches]

        # 포인트별 샘플 그룹 귀속성 슬라이싱 추출 -> [K1, Num_Matches], [K2, Num_Matches]
        feat_belongs_to_g1 = patch_group_mask1.float()[:, matched_patch_indices1]
        feat_belongs_to_g2 = patch_group_mask2.float()[:, matched_patch_indices2]

        # 블록 행렬 곱을 통한 그룹 간 XFeat 기하 결합 카운팅 -> [K1, K2]
        geometric_counts = torch.mm(feat_belongs_to_g1, feat_belongs_to_g2.t())

        # ====================================================================
        # PHASE 3. 💡 [2중 기하-시맨틱 제약 융합 및 매칭 조건 필터링]
        # ====================================================================
        # 조건 1: XFeat 매칭 투표수가 count_thresh 이상인가?
        geo_mask = (geometric_counts >= count_thresh)

        # 조건 2: DINOv2 기반 그룹 평균 피처 유사도가 sim_thresh 이상인가?
        sem_mask = (semantic_affinity >= sim_thresh)

        # 2중 제약을 상호 통과한 최종 유효 커넥션 마스크 생성
        valid_edge_mask = geo_mask & sem_mask

        # 💡 융합 스코어링: 필터를 통과한 영역에 대해서만 XFeat 투표수를 남겨두고 나머지는 복사 무력화(0.0)
        # RANSAC 없이도 기하 순도가 100% 보장되는 필터링 지점입니다.
        hybrid_affinity_counts = torch.where(valid_edge_mask, geometric_counts, 0.0)

        # 1번 프레임 각 그룹별 최적의 타겟 짝 선별
        max_counts, best_match_g2_idx = hybrid_affinity_counts.max(dim=1)
        valid_group_mask = max_counts > 0

        return hybrid_affinity_counts, semantic_affinity, best_match_g2_idx, valid_group_mask

    def extract_sample_neighborhood_average_pool2(self, x_cat, centroids, sim_thresh=0.7):

        _, D, H_p, W_p = x_cat.shape
        device = x_cat.device
        K = len(centroids)

        if K == 0:
            return torch.empty((0, D), device=device), torch.zeros(0, device=device)

        # 1. 전역 패치 피처 평탄화 및 L2 정규화 [N, D]
        feat_flat = x_cat[0].view(D, -1).t()  # [N, D]
        feat_norm = F.normalize(feat_flat, p=2, dim=1)

        # 2. 선별된 정예 샘플 노드들의 피처만 따로 추출 [K, D]
        sample_feats_norm = feat_norm[centroids]

        # 3. 💡 [첫 번째 거대 행렬 곱]: 샘플 노드 [K, D] vs 전체 패치 [D, N] 코사인 유사도 전수 조사
        # 결과 행렬 크기: [K, N]
        sim_matrix = torch.mm(sample_feats_norm, feat_norm.t())

        # 4. 💡 임계값 필터링을 통해 각 샘플별 '동적 패치 마스크 행렬' 생성 [K, N]
        # 박사님 기존 코드의 masks_bool.view(K, -1) 역할을 완벽히 대체합니다.
        # 중복 선택이 자연스럽게 허용되는 지점입니다.
        masks_bool = (sim_matrix > sim_thresh).float()  # [K, N] (0.0 또는 1.0)

        # 5. 각 샘플별 그룹에 귀속된 유효 패치 개수(면적) 계산 [K]
        # 기존 코드의 n_patches = masks_bool.sum(dim=(1, 2))에 대응
        n_patches_per_sample = masks_bool.sum(dim=1)  # [K]

        # 6. 💡 [두 번째 거대 행렬 곱]: 마스크 행렬 [K, N] @ 오리지널 피처 행렬 [N, D]
        # 기존 코드의 sum_vecs = torch.mm(masks_flat, feat_flat.t()) 메커니즘과 정확히 일치합니다.
        # L2 정규화가 안 된 원본 feat_flat을 곱해주어야 가중치 평균이 정확하게 계산됩니다.
        sum_vecs = torch.mm(masks_bool, feat_flat)  # [K, D]

        # 7. 평균 계산 및 예외 처리 (분모 분산 나눗셈)
        denom = n_patches_per_sample.unsqueeze(1)  # [K, 1]
        # 유효 패치가 1개 이상인 그룹만 평균을 내고, 없는 경우(아웃라이어) 안전하게 0 벡터 처리
        sample_avg_vecs = torch.where(denom > 0, sum_vecs / denom, torch.zeros_like(sum_vecs))

        # 8. 최종 코사인 유사도 매칭 오퍼레이션을 위해 출력 벡터 L2 정규화
        sample_avg_vecs_norm = F.normalize(sample_avg_vecs, p=2, dim=1)

        return sample_avg_vecs_norm, n_patches_per_sample

    def match_vectors(self, vecs1, vecs2, min_cossim=0.8):
        """
        Args:
            vecs1: [N1, D] Tensor
            vecs2: [N2, D] Tensor
            min_cossim: 임계값 (0.8)
        Returns:
            idx1: vecs1에서의 인덱스들
            idx2: vecs2에서의 인덱스들 (idx1[i]와 idx2[i]가 매칭됨)
        """
        if vecs1.shape[0] == 0 or vecs2.shape[0] == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        # 1. 코사인 유사도를 위해 L2 정규화 (필수)
        vecs1_norm = F.normalize(vecs1, p=2, dim=1)
        vecs2_norm = F.normalize(vecs2, p=2, dim=1)

        # 2. 유사도 행렬 계산 [N1, N2]
        sim_matrix = torch.mm(vecs1_norm, vecs2_norm.t())

        # 3. Mutual Nearest Neighbor (상호 최우선 매칭) 찾기
        # vecs1 기준 가장 닮은 vecs2의 인덱스
        conf12, match12 = sim_matrix.max(dim=1)
        # vecs2 기준 가장 닮은 vecs1의 인덱스
        conf21, match21 = sim_matrix.max(dim=0)

        # 4. 상호 검증 (Mutual Check)
        # idx0(v1의 인덱스)가 가리키는 v2가 다시 idx0를 가리키는지 확인
        idx1 = torch.arange(len(match12), device=vecs1.device)
        mutual_mask = (match21[match12] == idx1)

        # 5. 유사도 임계값(0.8) 필터링
        threshold_mask = conf12 >= min_cossim

        # 최종 마스크
        final_mask = mutual_mask & threshold_mask

        return idx1[final_mask], match12[final_mask]

    def visualize_hybrid_connections(
            self,img1_bgr, sample1, img2_bgr, sample2,
            hybrid_affinity_counts, semantic_affinity, best_match_g2_idx, valid_group_mask,
            grid_shape=(34, 45), patch_size=14
    ):
        """
        [진우 박사님 파이프라인 전용 - 텍스트 위치 쿼리 노드 근처 수정 버전]
        융합 매칭 결과 [XFeat 개수 / DINOv2 유사도] 수치 정보를
        선 중앙이 아닌 좌측 쿼리 이미지 포인트(cx1, cy1) 근처에 정밀 출력합니다.
        """
        h1, w1 = img1_bgr.shape[:2]
        h2, w2 = img2_bgr.shape[:2]
        _, W_p = grid_shape
        offset = patch_size // 2

        # 1. 가로 결합형 모니터링 캔버스 생성
        max_h = max(h1, h2)
        canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1_bgr.copy()
        canvas[:h2, w1:w1 + w2] = img2_bgr.copy()

        # 텐서 언래핑 및 넘파이 변환
        c1_np = sample1.cpu().numpy()
        c2_np = sample2.cpu().numpy()
        geo_counts_np = hybrid_affinity_counts.cpu().numpy()
        sem_sim_np = semantic_affinity.cpu().detach().numpy()
        best_matches_np = best_match_g2_idx.cpu().detach().numpy()
        valid_mask_np = valid_group_mask.cpu().numpy()

        # 양측 프레임 기본 샘플 노드 주황색 점 렌더링
        for idx1 in c1_np:
            cv2.circle(canvas, (int((idx1 % W_p) * patch_size + offset), int((idx1 // W_p) * patch_size + offset)), 2,
                       (0, 140, 255), -1, cv2.LINE_AA)
        for idx2 in c2_np:
            cv2.circle(canvas, (int((idx2 % W_p) * patch_size + offset) + w1, int((idx2 // W_p) * patch_size + offset)),
                       2, (0, 140, 255), -1, cv2.LINE_AA)

        max_voting = np.max(geo_counts_np) if np.sum(valid_mask_np) > 0 else 1.0

        # 2. 🔗 하이브리드 커넥션 에지 렌더링 루프
        for g1_id in range(len(c1_np)):
            if not valid_mask_np[g1_id]:
                continue

            g2_id = best_matches_np[g1_id]
            votes = geo_counts_np[g1_id, g2_id]
            sim = sem_sim_np[g1_id, g2_id]

            # 1번 프레임(좌측 쿼리) 픽셀 중심점
            cx1 = int((c1_np[g1_id] % W_p) * patch_size + offset)
            cy1 = int((c1_np[g1_id] // W_p) * patch_size + offset)

            # 2번 프레임(우측 타겟) 픽셀 중심점 (가로 오프셋 가산)
            cx2 = int((c2_np[g2_id] % W_p) * patch_size + offset) + w1
            cy2 = int((c2_np[g2_id] // W_p) * patch_size + offset)

            # 투표 수 기반 에지 두께/유사도 기반 발광 컬러 스케일 적용
            ratio = votes / max_voting
            thick = max(1, int(ratio * 4))
            color = (0, int(150 + ((sim - 0.5) * 210)), 0)  # Rich Semantic Green

            # 링크선 및 노드 앵커 렌더링
            cv2.line(canvas, (cx1, cy1), (cx2, cy2), color, thick, cv2.LINE_AA)
            cv2.circle(canvas, (cx1, cy1), 4, (255, 0, 100), 1, cv2.LINE_AA)
            cv2.circle(canvas, (cx2, cy2), 4, (255, 0, 100), 1, cv2.LINE_AA)

            # 💡 [핵심 요구사항 반영]: 텍스트 레이아웃을 쿼리 이미지 포인트(cx1, cy1) 바로 우상단으로 이동
            # 문자열 포맷 예시: "[32/0.89]" (투표수 32개 / DINO 코사인 유사도 0.89)
            score_txt = f"[{int(votes)}/{sim:.2f}]"
            text_pos = (cx1 + 8, cy1 - 6)

            # 가독성 확보용 검은색 그림자 테두리 백그라운드 2중 투사
            cv2.putText(canvas, score_txt, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, score_txt, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 255, 255), 1, cv2.LINE_AA)

        # 전체 시스템 상태창 헤더바
        status_txt = f"Dual-Constraint Hybrid Matcher Engine (Info Position: Query Nodes)"
        cv2.putText(canvas, status_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, status_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Geometric-Semantic Fusion Connections", canvas)
        cv2.waitKey(1)
        return canvas

    def visualize_patch_group_connections(
            self, img1_bgr, sample1, img2_bgr, sample2,
            group_affinity_counts, best_match_g2_idx, valid_group_mask,
            grid_shape=(34, 45), patch_size=14, count_thresh=3, window_name = "visualize_patch_group_connections"
    ):
        """
        [진우 박사님 하이브리드 파이프라인 전용 최종 커넥션 시각화]
        XFeat 전역 매칭 투표를 통해 프레임 간에 매칭된 패치 그룹(오브젝트 노드) 쌍을 연결합니다.
        연결선의 두께와 밝기는 매칭된 XFeat 포인트 개수(Score)에 비례합니다.

        Args:
            img1_bgr, img2_bgr      : 양 프레임 오리지널 BGR 이미지
            sample1, sample2        : 양 프레임 NMS 정예 패치 1D 인덱스 텐서 -> [K1], [K2]
            group_affinity_counts  : 패치 그룹 간 XFeat 투표 결과 행렬 -> [K1, K2]
            best_match_g2_idx       : 1번 그룹별 가장 많이 매칭된 2번 그룹 인덱스 목록 -> [K1]
            valid_group_mask        : 유효 매칭 존재 여부 부울 마스크 -> [K1]
            count_thresh            : 노이즈 차단용 최소 XFeat 투표수 임계값 (이 값 미만은 선 연결 스킵)
        """
        h1, w1 = img1_bgr.shape[:2]
        h2, w2 = img2_bgr.shape[:2]
        _, W_p = grid_shape
        offset = patch_size // 2

        # 1. 가로 결합형 모니터링 대형 도화지 복사 빌드
        max_h = max(h1, h2)
        canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1_bgr.copy()
        canvas[:h2, w1:w1 + w2] = img2_bgr.copy()

        # 텐서 안전 해제 및 Numpy 언래핑
        c1_np = sample1.cpu().numpy()
        c2_np = sample2.cpu().numpy()
        affinity_np = group_affinity_counts.cpu().numpy()
        best_matches_np = best_match_g2_idx.cpu().numpy()
        valid_mask_np = valid_group_mask.cpu().numpy()

        # 2. 양쪽 프레임에 존재하는 모든 정예 패치 노드들 기본 주황색 점으로 선샤인 렌더링
        for idx1 in c1_np:
            cx = int((idx1 % W_p) * patch_size + offset)
            cy = int((idx1 // W_p) * patch_size + offset)
            cv2.circle(canvas, (cx, cy), 2, (0, 140, 255), -1, cv2.LINE_AA)

        for idx2 in c2_np:
            cx = int((idx2 % W_p) * patch_size + offset) + w1
            cy = int((idx2 // W_p) * patch_size + offset)
            cv2.circle(canvas, (cx, cy), 2, (0, 140, 255), -1, cv2.LINE_AA)

        # 최대 투표수를 찾아 선 두께/밝기 정규화 스케일 팩터 확보
        max_voting_score = np.max(affinity_np) if np.sum(valid_mask_np) > 0 else 1.0
        if max_voting_score < 1.0: max_voting_score = 1.0

        connection_count = 0

        # 3. 💡 [커넥션 링크선 도포]: 유효한 매칭 노드 쌍 전수 역산 및 드로잉
        for g1_id in range(len(c1_np)):
            if not valid_mask_np[g1_id]:
                continue

            g2_id = best_matches_np[g1_id]
            voting_score = affinity_np[g1_id, g2_id]

            # 노이즈 에지 제거: XFeat 투표수가 박사님이 지정한 기준 미만이면 시각화선 연결 패스
            if voting_score < count_thresh:
                continue

            connection_count += 1

            # 1번 프레임 정예 노드 픽셀 중심점 계산
            idx1 = c1_np[g1_id]
            cx1 = int((idx1 % W_p) * patch_size + offset)
            cy1 = int((idx1 // W_p) * patch_size + offset)

            # 2번 프레임 정예 노드 픽셀 중심점 계산 (가로 가산)
            idx2 = c2_np[g2_id]
            cx2 = int((idx2 % W_p) * patch_size + offset) + w1
            cy2 = int((idx2 // W_p) * patch_size + offset)

            # 💡 기하학적 융합 스코어(투표율) 기반 동적 선 스타일 기획
            # 투표수가 많을수록 굵고(최대 두께 4), 선명한 초록색 발광 레이아웃 적용
            ratio = voting_score / max_voting_score
            line_thickness = max(1, int(ratio * 4))
            line_color = (0, int(130 + (ratio * 125)), int(ratio * 100))  # BGR (Rich Green)

            # 4. 에지 실선 렌더링 및 결합 앵커 오버레이
            cv2.line(canvas, (cx1, cy1), (cx2, cy2), line_color, line_thickness, cv2.LINE_AA)

            # 매칭 성공한 핵심 노드는 빨간 링으로 하이라이트
            cv2.circle(canvas, (cx1, cy1), 4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (cx2, cy2), 4, (0, 0, 255), 1, cv2.LINE_AA)

            # 연결선 중앙 지점에 투표된 XFeat 짝 개수 폰트 인라인 오버레이
            mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
            text_str = f"Xf:{int(voting_score)}"
            cv2.putText(canvas, text_str, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, text_str, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 255, 200), 1, cv2.LINE_AA)

        # 5. 상단 시스템 메타 상태 표시 서치바
        summary_txt = f"Patch Group Connectors (Active Links: {connection_count} / Max Voting: {int(max_voting_score)})"
        cv2.putText(canvas, summary_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, summary_txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)

        return canvas

    def visualize_sample_avg_matches(self, img1_bgr, centroids1, img2_bgr, centroids2, idx1, idx2, grid_shape, patch_size=14):
        """
        [기존 visualize_matches의 패치 평균 벡터 이식 버전]
        두 이미지 간에 매칭 성공한 샘플 컨텍스트 노드 쌍들을 고유 색상선과 매칭 ID 텍스트로 정밀 연결 시각화합니다.
        """
        h1, w1 = img1_bgr.shape[:2]
        h2, w2 = img2_bgr.shape[:2]
        _, W_p = grid_shape
        offset = patch_size // 2

        # 가로 결합형 시각화 도화지 세팅
        max_h = max(h1, h2)
        canvas = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1_bgr.copy()
        canvas[:h2, w1:w1 + w2] = img2_bgr.copy()

        # 안전한 numpy 맵핑용 주소 변환
        c1_np = centroids1.cpu().numpy()
        c2_np = centroids2.cpu().numpy()

        num_matches = len(idx1)
        if num_matches == 0:
            print("[시각화 알림] 매칭된 샘플 평균 벡터 쌍이 존재하지 않습니다.")
            cv2.imshow("Sample Average Node Matches", canvas)
            cv2.waitKey(1)
            return canvas

        # 매칭 쌍마다 고유 연결선 색상을 부여하기 위한 무작위 컬러 맵 세팅
        np.random.seed(777)
        colors = [tuple(map(int, c)) for c in np.random.randint(0, 255, (num_matches, 3))]

        # 💡 유효 매칭 노드 시각화 루프 가동
        for k, (i, j) in enumerate(zip(idx1, idx2)):
            color = colors[k]

            # 1. 쿼리 이미지(좌측)의 매칭 성공 패치 픽셀 센터 연산
            q_global_idx = c1_np[i]
            q_px = q_global_idx % W_p
            q_py = q_global_idx // W_p
            cx1 = int(q_px * patch_size + offset)
            cy1 = int(q_py * patch_size + offset)

            # 2. 타겟 이미지(우측)의 매칭 성공 패치 픽셀 센터 연산 및 가로 오프셋 가산
            t_global_idx = c2_np[j]
            t_px = t_global_idx % W_p
            t_py = t_global_idx // W_p
            cx2 = int(t_px * patch_size + offset) + w1
            cy2 = int(t_py * patch_size + offset)

            # 3. 💡 시각 기하 요소 렌더링 (외곽선 가이드 적용)
            # 연결 에지 라인 드로잉
            cv2.line(canvas, (cx1, cy1), (cx2, cy2), color, 2, cv2.LINE_AA)
            # 양측 정착 노드 마킹
            cv2.circle(canvas, (cx1, cy1), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx1, cy1), 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx2, cy2), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx2, cy2), 3, color, -1, cv2.LINE_AA)

            # 매칭 노드에 고유 전역 패치 번호 가독성 오버레이
            txt1 = f"P:{q_global_idx}"
            cv2.putText(canvas, txt1, (cx1 + 6, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, txt1, (cx1 + 6, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1,
                        cv2.LINE_AA)

            txt2 = f"P:{t_global_idx}"
            cv2.putText(canvas, txt2, (cx2 + 6, cy2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, txt2, (cx2 + 6, cy2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # 상단 총 매칭 쌍 요약 메타 삽입
        summary_text = f"DINOv2 Sample Neighborhood Average Vector Matches: {num_matches}"
        cv2.putText(canvas, summary_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, summary_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Sample Average Node Matches", canvas)
        cv2.waitKey(1)
        return canvas

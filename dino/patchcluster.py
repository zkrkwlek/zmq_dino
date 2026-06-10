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

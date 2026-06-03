import numpy as np
import torch
import torch.nn.functional as F
import cv2


def generate_pseudo_labels_by_local_density(features_all, sampled_centroids, spatial_radius=3.0, sim_thresh=0.7):
    """
    [접근법 B] 샘플 패치 주변 국소 영역 내에서 '나와 닮은 패치의 비율(밀도)'을 측정합니다.
    높은 밀도 -> 특징이 밋밋하게 밀려버린 배경/벽면(1), 낮은 밀도 -> 디테일이 살아있는 객체(0)
    """
    device = features_all.device
    N_all = features_all.shape[0]
    K_sampled = sampled_centroids.shape[0]

    # 1. 전역 픽셀 좌표계 및 전수 물리 거리 행렬 빌드 [K, N_all]
    W_p = 45
    offset = 14 // 2
    all_indices = torch.arange(N_all, device=device)
    coords_all = torch.stack([all_indices % W_p * 14 + offset, all_indices // W_p * 14 + offset], dim=1).float()

    # 샘플 좌표 [K, 2] vs 전역 좌표 [N_all, 2] 거리 계산
    spatial_dist = torch.cdist(coords_all[sampled_centroids.long()], coords_all)  # [K, N_all]

    # 💡 조건 1: 내 주변 영역 마스크 (지정한 픽셀 반경 이내인가?)
    local_mask = (spatial_dist < (spatial_radius * 14)).float()

    # 2. 특징 공간 코사인 유사도 연산 [K, N_all]
    feats_norm = F.normalize(features_all.float(), p=2, dim=-1)
    cross_sim = torch.mm(feats_norm[sampled_centroids.long()], feats_norm.t())

    # 💡 조건 2: 나와 시맨틱적으로 닮았는가?
    semantic_mask = (cross_sim > sim_thresh).float()

    # 3. 💡 [2중 조건 결합]: 내 주변에 '있으면서' 동시에 나와 '닮은' 패치들의 개수 산출
    # local_mask & semantic_mask 원소별 곱셈 후 행 합산
    overlapping_patches = (local_mask * semantic_mask).sum(dim=1)  # [K]

    # 내 주변에 존재하는 총 패치 수로 나누어 밀도(Ratio) 계산
    total_local_patches = local_mask.sum(dim=1)
    local_density = overlapping_patches / torch.clamp(total_local_patches, min=1.0)  # [K]

    # 4. 밀도가 높다 -> 국소 영역 전체가 나랑 똑같은 민자 벽면이다 -> 배경 (1)
    #    밀도가 낮다 -> 주변에 다른 텍스처나 경계선이 섞여 있다 -> 객체 (0)
    threshold = torch.median(local_density)
    pseudo_labels = (local_density > threshold).int()

    return pseudo_labels, local_density
def generate_pseudo_labels_vectorized(features_all, sampled_centroids, k=8):
    device = features_all.device
    N_all = features_all.shape[0]  # 전체 패치 수 (예: 1530)
    K_sampled = sampled_centroids.shape[0]  # 정예 샘플 수 (예: 300)

    # 1. 전역 코사인 유사도 연산 (L2 정규화 후 매트릭스 곱)
    feats_norm = F.normalize(features_all.float(), p=2, dim=-1)  # [N_all, D]

    # 💡 [핵심 연산]: 선택된 정예 패치 피처 [K, D]와 전체 패치 피처 [D, N_all] 곱셉
    # 행렬 크기: [K_sampled, N_all]
    cross_sim = torch.mm(feats_norm[sampled_centroids.long()], feats_norm.t())

    # 자기 자신 패치가 이웃으로 다시 잡히는 것을 방지 (행렬 대각 성분이 아닌 고유 인덱스 지점 마스킹)
    rows = torch.arange(K_sampled, device=device)
    cross_sim[rows, sampled_centroids.long()] = -1.0

    # 2. 각 정예 패치별로 이미지 전역 패치 중 가장 닮은 Top-K 인덱스 일괄 수확
    # topk_idx shape: [K_sampled, k] (내부 값은 0 ~ N_all-1 사이의 전역 인덱스)
    _, topk_idx = torch.topk(cross_sim, k=k, dim=1, largest=True, sorted=False)

    # 3. 💡 [전역 좌표계 하이패스 역산]: 14픽셀 간격의 전역 픽셀 좌표 행렬 생성 [N_all, 2]
    H_p, W_p = 34, 45  # 1530 패치 기준 해상도 격자
    offset = 14 // 2

    all_indices = torch.arange(N_all, device=device)
    all_px = all_indices % W_p
    all_py = all_indices // W_p

    # coords_all shape: [N_all, 2] -> 각 전역 인덱스 패치의 실제 (cx, cy) 픽셀 좌표
    all_cx = all_px * 14 + offset
    all_cy = all_py * 14 + offset
    coords_all = torch.stack([all_cx, all_cy], dim=1).float()

    # 4. 💡 루프 없이 topk_idx를 활용해 전역 좌표계에서 이웃 패치들의 2D 픽셀 좌표 일괄 수집
    # neighbor_xy shape: [K_sampled, k, 2]
    neighbor_xy = coords_all[topk_idx]

    # 5. 각 정예 샘플 노드 기준 이웃들의 전역 공간 분산(std) 연산
    # std(dim=1) 결과 [K_sampled, 2] -> mean(dim=1) 결과 [K_sampled]
    spatial_spread = torch.std(neighbor_xy, dim=1, unbiased=False).mean(dim=1)

    # 6. 중간값 기준 전경(0) / 배경(1) 분기
    threshold = torch.median(spatial_spread)
    pseudo_labels = (spatial_spread > threshold).int()

    return pseudo_labels, spatial_spread

def generate_pseudo_labels(feats_norm, sampled_xy, k=8):
    """
    features   : (N, D) DINOv2 patch features
    sampled_xy : (N, 2) pixel coordinates

    반환: pseudo_labels (0=foreground, 1=background)
    """
    N = len(feats_norm)
    # feature 공간 cosine similarity

    sim_matrix = (feats_norm @ feats_norm.T).detach().cpu().numpy()  # (N, N)
    np.fill_diagonal(sim_matrix, -1)

    spatial_spread = np.zeros(N)

    for i in range(N):
        # feature 공간 top-k 이웃
        topk_idx = np.argsort(sim_matrix[i])[-k:]

        # 그 이웃들의 공간적 분산
        neighbor_xy = sampled_xy[topk_idx]  # (k, 2)
        spread = neighbor_xy.std(axis=0).mean()
        spatial_spread[i] = spread

    # 높은 spread → 배경 (1), 낮은 spread → 객체 (0)
    threshold = np.median(spatial_spread)
    pseudo_labels = (spatial_spread > threshold).astype(int)

    return pseudo_labels, spatial_spread

def visualize_pseudo_labels_opencv(img_bgr, centroids, pseudo_labels, spatial_spread, grid_shape, patch_size=14,
                                   window_name="DINOv2 Spatial-Spread Pseudo Labels"):
    """
    Args:
        img_bgr (np.ndarray): 원본 BGR 이미지 (H, W, 3)
        centroids (np.ndarray or torch.Tensor): 샘플 노드들의 1D 격자 인덱스 [N]
        pseudo_labels (np.ndarray): generate_pseudo_labels의 결과 (0 또는 1) [N]
        spatial_spread (np.ndarray): 각 패치별 계산된 이웃 std 평균값 [N]
        grid_shape (tuple): 패치 격자 크기 (H_p, W_p) 예: (34, 45)
    """
    canvas = img_bgr.copy()
    H_p, W_p = grid_shape
    offset = patch_size // 2

    # 💡 [핵심 해결 포인트]: PyTorch 텐서(CUDA/CPU)가 들어오면 안전하게 .numpy()로 언래핑
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    elif isinstance(centroids, list):
        centroids = np.array(centroids)

    if isinstance(pseudo_labels, torch.Tensor):
        pseudo_labels = pseudo_labels.detach().cpu().numpy()

    if isinstance(spatial_spread, torch.Tensor):
        spatial_spread = spatial_spread.detach().cpu().numpy()

    # 이제 100% 순수 Numpy 영역이므로 np.sum() 연산이 절대 패닉을 내지 않습니다.
    num_fg = int(np.sum(pseudo_labels == 0))
    num_bg = int(np.sum(pseudo_labels == 1))
    N = len(centroids)

    print(f"\n[Visualizer] {N}개 샘플 패치 의사 라벨 시각화 중...")
    print(f" - 전경 (Foreground, Object) 개수: {num_fg}개")
    print(f" - 배경 (Background, Wall) 개수: {num_bg}개")

    for i, idx in enumerate(centroids):
        label = pseudo_labels[i]
        spread_val = spatial_spread[i]

        # 1D 인덱스 기반 픽셀 좌표 역산
        py = idx // W_p
        px = idx % W_p
        cx = int(px * patch_size + offset)
        cy = int(py * patch_size + offset)

        # 전경 / 배경 가이드 마커 드로잉
        if label == 0:
            color = (0, 102, 255)  # Orange-Red (Foreground)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 2, color, -1, cv2.LINE_AA)
        else:
            color = (255, 128, 0)  # Cyan-Blue (Background/Wall)
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (cx - 3, cy), (cx + 3, cy), color, 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy - 3), (cx, cy + 3), color, 1, cv2.LINE_AA)

        # 분산 스코어 텍스트 표기
        text_str = f"{spread_val:.1f}"
        cv2.putText(canvas, text_str, (cx + 5, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, text_str, (cx + 5, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    # 상단 메타 가이드 바 오버레이
    cv2.putText(canvas, f"FG(Object): Orange | BG(Wall): Blue Cross", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)
    return canvas
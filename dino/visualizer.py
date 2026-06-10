import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class UnifiedAttentionDirectVectorVisualizer:
    """
    [진우 박사님 전용 - 중복 연산 제로 & 이미 구해진 평균 패치 벡터 직결형 종합 디버거]
    내부에서 평균 벡터를 다시 구하지 않고, 박사님이 완성하신 sample_avg_vecs를 그대로 전달받아
    셀프 및 크로스 어텐션 히트맵을 1:1 순정 크기로 투사합니다.
    """

    def __init__(self, grid_shape=(34, 45)):
        self.Hp, self.Wp = grid_shape
        self.N = self.Hp * self.Wp

        # 순정 데이터 버퍼
        self.img_base = None
        self.img_neigh = None
        self.feat_base = None
        self.feat_neigh = None

        # 💡 [박사님 핵심 지시 버퍼]: 외부에서 완공된 평균 벡터 및 세력권 매핑 매트릭스
        self.sample_avg_vecs = None  # [K, D] 이미 구해놓은 사물별 대표 평균 벡터 행렬
        self.affinity_base = None  # [K, N] 또는 [N, N] 패치가 어느 사물(K)에 속해있는지 나타내는 매핑 마스크

        self.base_h, self.base_w = 0, 0
        self.neigh_h, self.neigh_w = 0, 0
        self.window_name = "JinWoo_Doctor_DirectVector_Viewer"

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= self.base_w:
                return

            # 원본 스케일 기반 패치 격자 역산
            step_w = self.base_w / self.Wp
            step_h = self.base_h / self.Hp

            patch_x = max(0, min(int(x / step_w), self.Wp - 1))
            patch_y = max(0, min(int(y / step_h), self.Hp - 1))
            target_patch_idx = patch_y * self.Wp + patch_x

            # ⚡ 외부 주입 벡터 기반으로 원샷 히트맵 리프레시
            self._render_heatmaps_with_precomputed_vector(target_patch_idx, patch_x, patch_y)

    def _render_heatmaps_with_precomputed_vector(self, target_patch_idx, patch_x, patch_y):
        """💡 [중복 계산 금지 구역]: 전달받은 sample_avg_vecs에서 선택된 사물의 대표 벡터를 즉시 룩업합니다."""

        # 1. 사용자가 클릭한 패치가 '몇 번 사물 인덱스(k)'에 속해있는지 조사
        if self.affinity_base.shape[0] == self.N:
            # [N, N] 구조인 경우: 자기 자신이 곧 앵커 인덱스
            target_k_idx = target_patch_idx
            # 시각화 가이드를 위한 마스크 면적 계산
            n_patches = self.affinity_base[target_patch_idx].float().sum().item()
        else:
            # [K, N] 구조인 경우: 해당 패치 열에서 1이 켜진 사물 행(k)을 룩업
            connected_rows = torch.where(self.affinity_base[:, target_patch_idx] > 0)[0]
            if connected_rows.numel() == 0:
                print(f"⚠️ [직결 엔진] 패치 {target_patch_idx}가 포섭된 사물 채널(K)을 찾을 수 없습니다.")
                return
            target_k_idx = connected_rows[0].item()
            n_patches = self.affinity_base[target_k_idx].float().sum().item()

        # 2. 🎯 [박사님 핵심 의도]: 다시 더하고 나눌 필요 없이, 이미 구해서 인풋으로 넘어온 고유 평균 벡터 슬라이싱!
        # [K, D] -> [1, D]
        pure_avg_vec_base = self.sample_avg_vecs[target_k_idx].view(1, -1)

        # 4. 🚀 [교정 셀프 어텐션] -> 왼쪽 캔버스 배포
        self_scores = torch.mm(self.feat_base, pure_avg_vec_base.t()).squeeze()
        self_map_2d = self_scores.view(self.Hp, self.Wp).cpu().numpy()

        # 5. 🚀 [교정 크로스 어텐션] -> 오른쪽 캔버스 배포 (위치 고착 버그 완전 폭파 지점)
        cross_scores = torch.mm(self.feat_neigh, pure_avg_vec_base.t()).squeeze()
        cross_map_2d = cross_scores.view(self.Hp, self.Wp).cpu().numpy()

        # 6. 고해상도 오버레이 렌더링 컴파일
        def _generate_heatmap_overlay(attn_2d, orig_img, target_h, target_w):
            attn_2d = np.clip(attn_2d, 0.0, 1.0)
            a_min, a_max = attn_2d.min(), attn_2d.max()
            a_norm = (attn_2d - a_min) / (a_max - a_min + 1e-8)
            h_src = (a_norm * 255).astype(np.uint8)
            h_res = cv2.resize(h_src, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            h_col = cv2.applyColorMap(h_res, cv2.COLORMAP_JET)
            return cv2.addWeighted(orig_img, 0.6, h_col, 0.4, 0, dtype=cv2.CV_8U)

        vis_left_bgr = _generate_heatmap_overlay(self_map_2d, self.img_base, self.base_h, self.base_w)
        vis_right_bgr = _generate_heatmap_overlay(cross_map_2d, self.img_neigh, self.neigh_h, self.neigh_w)

        # 원본 해상도 위에 그릴 투명 그리드 캔버스 생성
        grid_overlay = vis_left_bgr.copy()
        step_w = self.base_w / self.Wp
        step_h = self.base_h / self.Hp

        # 세로 격자선 드로잉 (X축 이동)
        for w_idx in range(1, self.Wp):
            x_pos = int(w_idx * step_w)
            cv2.line(grid_overlay, (x_pos, 0), (x_pos, self.base_h), (255, 255, 0), 1, cv2.LINE_AA)
        # 가로 격자선 드로잉 (Y축 이동)
        for h_idx in range(1, self.Hp):
            y_pos = int(h_idx * step_h)
            cv2.line(grid_overlay, (0, y_pos), (self.base_w, y_pos), (255, 255, 0), 1, cv2.LINE_AA)

        if self.affinity_base.shape[0] == self.N:
            # [N, N] 구조인 경우 각 행의 합이 0보다 큰지 검사
            valid_mask_1d = (self.affinity_base.sum(dim=1) > 0).cpu().numpy()
        else:
            # [K, N] 구조인 경우 각 열(패치)에 1이라도 켜져 있는지 전역 논리합 추출 [N]
            valid_mask_1d = torch.any(self.affinity_base > 0, dim=0).cpu().numpy()

        valid_indices = np.where(valid_mask_1d)[0]

        # 1D 인덱스들을 2D 격자 좌표 (wp_idx, hp_idx)로 일괄 환원
        wp_indices = valid_indices % self.Wp
        hp_indices = valid_indices // self.Wp

        # 넘파이 벡터 연산으로 원소별 중앙 픽셀 좌표 리스트 빌드
        center_xs = ((wp_indices + 0.5) * step_w).astype(np.int32)
        center_ys = ((hp_indices + 0.5) * step_h).astype(np.int32)

        # OpenCV 고속 드로잉 바인딩 (C++ 가속단으로 바로 전달되므로 지연시간 제로)
        for pt_x, pt_y in zip(center_xs, center_ys):
            cv2.circle(grid_overlay, (pt_x, pt_y), 2, (0, 165, 255), -1, cv2.LINE_AA)

        # 원본 히트맵과 격자망을 85:15 비율로 블렌딩하여 은은하게 노출 (시야 방해 제거)
        vis_left_bgr = cv2.addWeighted(vis_left_bgr, 0.85, grid_overlay, 0.15, 0)

        # 7. 클릭 마커 드로잉
        step_w = self.base_w / self.Wp
        step_h = self.base_h / self.Hp
        x1, y1 = int(patch_x * step_w), int(patch_y * step_h)
        x2, y2 = int((patch_x + 1) * step_w), int((patch_y + 1) * step_h)
        cv2.rectangle(vis_left_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

        # 8. 👑 [순정 마스터 캔버스 합성 및 강제 크기 리사이즈 동기화]
        master_canvas = np.zeros((self.base_h, self.base_w + self.neigh_w, 3), dtype=np.uint8)
        master_canvas[:, :self.base_w] = vis_left_bgr
        master_canvas[:, self.base_w:] = vis_right_bgr

        # 안내 자막 임베딩
        info_left = f"FRAME 1 (SELF) | Clicked: {target_patch_idx} (Object Channel k: {target_k_idx}) | Area: {int(n_patches)}"
        info_right = "FRAME 2 (CROSS - PURE MATCHING)"
        cv2.putText(master_canvas, info_left, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(master_canvas, info_right, (self.base_w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.line(master_canvas, (self.base_w, 0), (self.base_w, self.base_h), (255, 255, 255), 2)

        # 화면 해상도 1:1 칼같이 강착
        cv2.resizeWindow(self.window_name, self.base_w + self.neigh_w, self.base_h)
        cv2.imshow(self.window_name, master_canvas)

    def start_unified_direct_vector_viewer(
            self, img_bgr_base, img_bgr_neighbor, feat_base, feat_neighbor, sample_avg_vecs, affinity_base
    ):
        """💡 [호출 인터페이스 커스텀]: 박사님이 구하신 sample_avg_vecs를 다이렉트로 전달받습니다."""
        self.base_h, self.base_w, _ = img_bgr_base.shape
        self.neigh_h, self.neigh_w, _ = img_bgr_neighbor.shape

        self.img_base = img_bgr_base.copy()
        self.img_neigh = img_bgr_neighbor.copy()
        self.feat_base = feat_base
        self.feat_neigh = feat_neighbor

        # 👑 박사님 고유 전처리 자산 바인딩
        self.sample_avg_vecs = sample_avg_vecs
        self.affinity_base = affinity_base

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.base_w + self.neigh_w, self.base_h)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # 가동 시 초기 정중앙 패치 기준으로 팝업 개설
        self._render_heatmaps_with_precomputed_vector((self.Hp // 2) * self.Wp + (self.Wp // 2), self.Wp // 2,
                                                      self.Hp // 2)

        while True:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(self.window_name)
        return

class DinoPatchVisualizer:
    def __init__(self):
        self.colors = self.generate_max_distinct_colors()

    def generate_max_distinct_colors(self, num_colors=50):

        colors_bgr = []

        for i in range(num_colors):
            # 1. 색상(Hue)을 num_colors만큼 균등 분할 (0 ~ 179 범위)
            h = int((i * 180) / num_colors)

            # 2. 이웃한 색상들이 뭉쳐 보이지 않도록 채도(S)와 명도(V)를 격차 분배
            if i % 2 == 0:
                s = 255 - (i % 3) * 30  # 진한 색 계열
                v = 240 - (i % 2) * 40
            else:
                s = 180 + (i % 3) * 25  # 중간/연한 색 계열
                v = 255 - (i % 2) * 30

            # 3. 단일 픽셀 HSV 매트릭스를 빌드하여 OpenCV 내장 함수로 BGR 변환
            hsv_pixel = np.array([[[h, s, v]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)

            colors_bgr.append(bgr_pixel[0, 0].tolist())

        # 4. (선택 사항) 인접한 인덱스 간의 색상 차이를 더 벌리기 위한 결정론적 셔플링
        # 박사님 시스템의 데이터 재현성을 위해 시드는 고정합니다.
        np.random.seed(1337)
        colors_bgr = np.array(colors_bgr, dtype=np.uint8)
        shuffle_indices = np.random.permutation(num_colors)
        distinct_colors = colors_bgr[shuffle_indices].tolist()

        return distinct_colors

    def visualize_anchor_relations(self, img_bgr, centroids, inhibition_mask, anchor_links, grid_shape, alpha=0.4):
        H_img, W_img, _ = img_bgr.shape
        H_p, W_p = grid_shape
        K = centroids.shape[0]

        # 💡 [구조 변경]: 원본 BGR 이미지를 복사하여 캔버스로 사용 (Matplotlib과 달리 BGR 유지)
        canvas_img = img_bgr.copy()
        mask_overlay = np.zeros_like(img_bgr, dtype=np.uint8)

        centroids_np = centroids.cpu().numpy()
        hard_mask_kn = inhibition_mask[centroids].cpu().numpy()  # [K, N]
        anchor_links_np = anchor_links.cpu().numpy()  # [K, K]

        np.random.seed(42)  # 색상 고정

        scale_y = H_img / H_p
        scale_x = W_img / W_p

        # ====================================================================
        # PHASE 1: [K, N] 앵커 세력권 마스크 컬러링
        # ====================================================================
        for k_idx in range(K):
            color = self.colors[k_idx]  # OpenCV는 BGR 순서로 채색
            member_patch_indices = np.where(hard_mask_kn[k_idx])[0]

            for p_idx in member_patch_indices:
                y_p = p_idx // W_p
                x_p = p_idx % W_p

                y_start, y_end = int(y_p * scale_y), int((y_p + 1) * scale_y)
                x_start, x_end = int(x_p * scale_x), int((x_p + 1) * scale_x)

                mask_overlay[y_start:y_end, x_start:x_end] = color

        # 💡 [원샷 합성]: 마스크 오버레이 블렌딩
        fused_img = cv2.addWeighted(canvas_img, 1.0, mask_overlay, alpha, 0)

        # ====================================================================
        # PHASE 2: 앵커 중심점 좌표 계산 및 [K, K] 링크 위상선 작도 (OpenCV 전산화)
        # ====================================================================
        anchor_coords_pixel = []
        for c_idx in centroids_np:
            y_p = c_idx // W_p
            x_p = c_idx % W_p
            pixel_y = int((y_p + 0.5) * scale_y)
            pixel_x = int((x_p + 0.5) * scale_x)
            anchor_coords_pixel.append((pixel_x, pixel_y))

        # 1. [K, K] 연결 위상선 렌더링 (주황색 투명도 처리를 위해 선 전용 오버레이 사용)
        line_overlay = fused_img.copy()
        for i in range(K):
            for j in range(K):
                if anchor_links_np[i, j]:
                    pt1 = anchor_coords_pixel[i]
                    pt2 = anchor_coords_pixel[j]
                    # OpenCV 선 작도: cv2.line(이미지, 시작점, 끝점, 색상(BGR), 두께, 선종류)
                    # #FF5722 -> BGR: (34, 87, 255)
                    cv2.line(line_overlay, pt1, pt2, (34, 87, 255), 2, cv2.LINE_AA)

        # 선에 대한 알파 투명도(0.8) 융합
        fused_img = cv2.addWeighted(fused_img, 0.2, line_overlay, 0.8, 0)

        # 2. 정예 앵커 중심점 및 텍스트 플로팅
        for idx, (pt_x, pt_y) in enumerate(anchor_coords_pixel):
            # 테두리 검은색 원 (두께 4)
            cv2.circle(fused_img, (pt_x, pt_y), 7, (0, 0, 0), 4, cv2.LINE_AA)
            # 내부 녹색 원 (두께 -1은 채우기, #00FF00 -> BGR: (0, 255, 0))
            cv2.circle(fused_img, (pt_x, pt_y), 5, (0, 255, 0), -1, cv2.LINE_AA)

        # 3. 최상단 디버깅용 타이틀 바 및 텍스트 렌더링
        title_text = f"Localized Scene Topology Graph (Detected Object Nodes: {K})"
        cv2.putText(
            fused_img,
            title_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ====================================================================
        # PHASE 3: 출력 인터페이스 제어
        # ====================================================================
        # 실시간 창 띄우기가 필요할 경우 아래 두 줄 주석을 해제하세요.
        cv2.imshow("Scene Topology Graph", fused_img)
        cv2.waitKey(1)

        return fused_img

    def visualize_sample_to_sample_similarity(self, img_bgr, centroids, sample_vecs_norm, affinity, grid_shape,
                                              sim_threshold=0.7, alpha=0.5):
        """
        [진우 박사님 디버깅 버전 - 텍스트 위치를 실제 대표 샘플(Centroid Patch) 중심으로 고정]
        """
        H_img, W_img, _ = img_bgr.shape
        H_p, W_p = grid_shape
        K, N = affinity.shape  # K: 샘플(앵커) 개수, N: 전체 패치 개수

        # 1. [K, C] x [C, K] -> [K, K] 행렬곱으로 샘플 간 상호 유사도 도출
        sample_sim_matrix = torch.matmul(sample_vecs_norm, sample_vecs_norm.T)
        sample_sim_np = sample_sim_matrix.cpu().numpy()

        # 2. [샘플 간 군집화 트리거]: 유사도가 높은 샘플끼리 같은 Group ID 부여
        group_assignments = np.arange(K)  # 처음에는 각 샘플이 독립 군집 (0 ~ K-1)

        for i in range(K):
            for j in range(i + 1, K):
                if sample_sim_np[i, j] >= sim_threshold:
                    root_i = group_assignments[i]
                    group_assignments[group_assignments == group_assignments[j]] = root_i

        # 유니크한 그룹 개수 파악 후 고유 색상표 생성
        unique_groups = np.unique(group_assignments)
        num_groups = len(unique_groups)

        np.random.seed(42)  # 구조적 색상 고정

        # 그룹 ID와 고유 컬러 매핑 딕셔너리 빌드
        color_map = {g_id: self.colors[idx] for idx, g_id in enumerate(unique_groups)}

        # 3. 도화지 및 패치 소속 레이아웃 전개
        mask_overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        hard_mask_kn = affinity.cpu().numpy()  # [K, N] 어피니티 맵 활용

        scale_y = H_img / H_p
        scale_x = W_img / W_p

        # 💡 [구조 변경]: 실제 대표 샘플 패치(Centroid)의 좌표 정보를 추출
        centroids_np = centroids.cpu().numpy() if hasattr(centroids, "cpu") else np.array(centroids)
        sample_text_positions = []

        # ====================================================================
        # PHASE 1: 그룹화된 샘플 세력권 컬러링 및 대표 샘플 좌표 수집
        # ====================================================================
        for k_idx in range(K):
            belonging_group = group_assignments[k_idx]
            color = color_map[belonging_group]

            # 💡 [핵심 해결 포인트]: 하위 패치 평균 대신, 진짜 대표 샘플(Centroid) 패치 인덱스를 가져옴
            c_patch_idx = centroids_np[k_idx]

            # 1차원 패치 번호를 2D 격자 좌표(y, x)로 변환
            c_y_p = c_patch_idx // W_p
            c_x_p = c_patch_idx % W_p

            # 💡 대표 샘플 패치의 정중앙 물리 픽셀 좌표 산출
            sample_center_x = int((c_x_p + 0.5) * scale_x)
            sample_center_y = int((c_y_p + 0.5) * scale_y)

            # 글씨 위치 정보 저장 (실제 대표 샘플 패치 중심)
            sample_text_positions.append((sample_center_x, sample_center_y, belonging_group))

            # 격자 마스크 채색 진행
            member_patch_indices = np.where(hard_mask_kn[k_idx] > 0.5)[0]
            for p_idx in member_patch_indices:
                y_p = p_idx // W_p
                x_p = p_idx % W_p

                y_start, y_end = int(y_p * scale_y), int((y_p + 1) * scale_y)
                x_start, x_end = int(x_p * scale_x), int((x_p + 1) * scale_x)

                mask_overlay[y_start:y_end, x_start:x_end] = color

        # ====================================================================
        # PHASE 2: 원본 이미지 융합 및 디버깅 데이터 인쇄
        # ====================================================================
        fused_img = cv2.addWeighted(img_bgr, 1.0 - alpha, mask_overlay, alpha, 0)

        # ====================================================================
        # PHASE 2: 원본 이미지 융합 및 동일 그룹 간 위상선 작도
        # ====================================================================
        # 💡 [추가]: 선명한 드로잉을 위해 선 전용 오버레이 도화지 복사
        line_overlay = fused_img.copy()

        # 💡 [추가]: 같은 그룹으로 묶인 샘플 간 징검다리 연결선 작도
        for i in range(len(sample_text_positions)):
            for j in range(i + 1, len(sample_text_positions)):
                pt1_x, pt1_y, g_id1 = sample_text_positions[i]
                pt2_x, pt2_y, g_id2 = sample_text_positions[j]

                # 동일한 그룹 ID를 공유하는 별개의 두 샘플 노드라면 선 연결
                if g_id1 == g_id2:
                    # 해당 그룹의 고유 컬러로 연결선 채색
                    line_color = color_map[g_id1]
                    # 선 굵기 2, 부드러운 안티앨리어싱 효과 포함
                    cv2.line(line_overlay, (pt1_x, pt1_y), (pt2_x, pt2_y), line_color, 2, cv2.LINE_AA)

        # 위상 연결선 레이어를 70% 투명도로 메인 이미지에 융합 (배경 가림 방지)
        fused_img = cv2.addWeighted(fused_img, 0.3, line_overlay, 0.7, 0)

        # '각각의 소속 대표 샘플 패치 위치'마다 그룹 ID 분할 출력
        for s_x, s_y, g_id in sample_text_positions:
            id_text = f"G:{g_id}"

            # 검은색 그림자 테두리 (두께 2)
            cv2.putText(fused_img, id_text, (s_x - 12, s_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            # 흰색 본문 글자 (두께 1)
            cv2.putText(fused_img, id_text, (s_x - 12, s_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 시스템 모니터링 안내 문구
        title_text = f"Sample-to-Sample Similarity Map (Total Anchors: {K} -> Merged Groups: {num_groups})"
        cv2.putText(
            fused_img,
            title_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("Sample-to-Sample Similarity", fused_img)
        cv2.waitKey(1)
        return fused_img

    def visualize_global_scene_matching(
            self,
            img1_rgb,
            img2_rgb,
            final_point_pairs_mask,
            match12_table,
            A_k1_f1,
            A_k2_f2,
            xfeat_kpts1,
            xfeat_kpts2,
            grid_shape1=(34, 45),
            grid_shape2=(34, 45),
            save_path="./global_scene_match.png",
    ):
        """[진우 박사님 지적 반영 - match12_table을 이용한 오른쪽 정예 포인트 룩업 정형화]"""
        if save_path:
            out_dir = os.path.dirname(save_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        device = final_point_pairs_mask.device
        K1, K2, _ = final_point_pairs_mask.shape

        match_counts = final_point_pairs_mask.sum(dim=-1)
        valid_k1_all, valid_k2_all = torch.where(match_counts > 0)

        h1, w1, _ = img1_rgb.shape
        h2, w2, _ = img2_rgb.shape
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1_rgb
        canvas[:h2, w1: w1 + w2] = img2_rgb

        x_offset = w1
        p_h1, p_w1 = h1 / grid_shape1[0], w1 / grid_shape1[1]
        p_h2, p_w2 = h2 / grid_shape2[0], w2 / grid_shape2[1]

        kpts1_np = (
            xfeat_kpts1.cpu().numpy()
            if torch.is_tensor(xfeat_kpts1)
            else xfeat_kpts1
        )
        kpts2_np = (
            xfeat_kpts2.cpu().numpy()
            if torch.is_tensor(xfeat_kpts2)
            else xfeat_kpts2
        )

        print(
            f"🎨 [앵커 위상 융합] match12_table 정밀 룩업 기반 {len(valid_k1_all)}개 객체 쌍 시각화 가동"
        )

        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(len(valid_k1_all), 3), dtype=int)

        for pair_idx, (k1, k2) in enumerate(
                zip(valid_k1_all.tolist(), valid_k2_all.tolist())
        ):
            color = tuple(int(c) for c in colors[pair_idx])

            # ----------------------------------------------------------------
            # 1. 💡 [매치테이블 대조인프라 가동]: 진짜 매칭된 F1, F2 인덱스만 정확히 슬라이싱
            # ----------------------------------------------------------------
            valid_f1_indices = torch.where(final_point_pairs_mask[k1, k2])[0]
            # 🔥 여기서 주입받은 match12_table을 사용하여 짝꿍 F2 인덱스를 정확하게 찾습니다!
            valid_f2_indices = match12_table[k1, k2, valid_f1_indices]

            f1_idx_np = valid_f1_indices.cpu().numpy()
            f2_idx_np = valid_f2_indices.cpu().numpy()

            # 매칭된 포인트가 없으면 스킵
            if len(f1_idx_np) < 5:
                continue

            # 💡 [박사님 피드백 핵심]: 전체 귀속 포인트가 아니라, '실제 매칭에 성공한 정예 점들'의 좌표만 추출
            k1_pts = kpts1_np[f1_idx_np]
            k2_pts = kpts2_np[f2_idx_np]

            # 2. 정예 포인트들로만 2D Bounding Box 타이트하게 계산
            bbox1 = (
                int(k1_pts[:, 0].min()),
                int(k1_pts[:, 1].min()),
                int(k1_pts[:, 0].max()),
                int(k1_pts[:, 1].max()),
            )
            bbox2 = (
                int(k2_pts[:, 0].min()),
                int(k2_pts[:, 1].min()),
                int(k2_pts[:, 0].max()),
                int(k2_pts[:, 1].max()),
            )

            cv2.rectangle(
                canvas, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), color, 2
            )
            cv2.rectangle(
                canvas,
                (bbox2[0] + x_offset, bbox2[1]),
                (bbox2[2] + x_offset, bbox2[3]),
                color,
                2
            )

            cv2.putText(
                canvas,
                f"K1:{k1}",
                (bbox1[0], bbox1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            cv2.putText(
                canvas,
                f"K2:{k2}",
                (bbox2[0] + x_offset, bbox2[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

            # 3. 매칭 성공 패치 격자 및 포인트 점 작도
            for pt in k1_pts:
                grid_x, grid_y = int(pt[0] / p_w1), int(pt[1] / p_h1)
                cv2.rectangle(
                    canvas,
                    (int(grid_x * p_w1), int(grid_y * p_h1)),
                    (int((grid_x + 1) * p_w1), int((grid_y + 1) * p_h1)),
                    color,
                    1,
                )
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, color, -1)

            for pt in k2_pts:
                grid_x, grid_y = int(pt[0] / p_w2), int(pt[1] / p_h2)
                cv2.rectangle(
                    canvas,
                    (int(grid_x * p_w2 + x_offset), int(grid_y * p_h2)),
                    (
                        int((grid_x + 1) * p_w2 + x_offset),
                        int((grid_y + 1) * p_h2),
                    ),
                    color,
                    1,
                )
                cv2.circle(
                    canvas, (int(pt[0] + x_offset), int(pt[1])), 2, color, -1
                )

            # 4. 사물 앵커 중심선 연결
            center1 = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
            center2 = (
                int((bbox2[0] + bbox2[2]) / 2 + x_offset),
                int((bbox2[1] + bbox2[3]) / 2),
            )

            cv2.line(canvas, center1, center2, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center1, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center2, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        # 메타 정보 렌더링
        cv2.putText(
            canvas,
            f"Global Object-level Link Map (Table Verified) | Active Links: {len(valid_k1_all)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if save_path:
            cv2.imwrite(save_path, canvas)
            print(f"💾 [테이블 역산 완결] 정밀 링킹 시각화 결과 저장 완료: {save_path}")

        try:
            cv2.imshow("Global Object-Patch-Point Fusion Map", canvas)
            cv2.waitKey(1)
        except Exception:
            pass

    def visualize_comparison_triple_canvas(
            self,
            img1_rgb,
            img2_rgb,
            final_point_pairs_mask,
            match12_table,
            xfeat_idx1,
            xfeat_idx2,
            xfeat_kpts1,
            xfeat_kpts2,
            grid_shape=(34, 45),
            save_path=None,
            smat=None,  # 💡 박사님이 연산하신 [K1, K2] 순수 객체 시맨틱 유사도 행렬 다이렉트 주입!
            A_k1_f1=None,  # bbox 역산을 위한 사물-포인트 귀속 행렬들
            A_k2_f2=None,
            smat_thresh=0.60  # 기하 검증 전, 시맨틱 매칭으로 인정할 문턱값
    ):
        """
        [진우 박사님 시각화 진짜 종결 버전 - smat 시맨틱 대조군 탑재 3단 엔진]

        1단 (Top)   : 순정 XFeat 디스크립터 매칭 (기하 전수조사 노이즈 확인)
        2단 (Middle): 💡 [smat 반영] 기하 검증 전, 순수 DINOv2 객체 시맨틱 매칭 선 (smat > thresh)
        3단 (Bottom): 우리 하이브리드 배치 기반 사물 위상 링킹 (최종 정제 매칭)
        """
        import os
        if save_path:
            out_dir = os.path.dirname(save_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        h, w, _ = img1_rgb.shape
        x_offset = w

        # CPU Numpy 캐싱 및 텐서 변환
        kpts1_np = xfeat_kpts1.cpu().numpy() if torch.is_tensor(xfeat_kpts1) else xfeat_kpts1
        kpts2_np = xfeat_kpts2.cpu().numpy() if torch.is_tensor(xfeat_kpts2) else xfeat_kpts2
        xf_idx1_np = xfeat_idx1.cpu().numpy() if torch.is_tensor(xfeat_idx1) else xfeat_idx1
        xf_idx2_np = xfeat_idx2.cpu().numpy() if torch.is_tensor(xfeat_idx2) else xfeat_idx2

        A_k1_f1_np = A_k1_f1.cpu().numpy() > 0 if A_k1_f1 is not None else None
        A_k2_f2_np = A_k2_f2.cpu().numpy() > 0 if A_k2_f2 is not None else None

        # ====================================================================
        # 🖼️ 1단: Pure XFeat Descriptor Matching
        # ====================================================================
        canvas_top = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas_top[:, :w] = img1_rgb
        canvas_top[:, w:] = img2_rgb

        matched_set1 = set(xf_idx1_np.tolist())
        matched_set2 = set(xf_idx2_np.tolist())

        # 프레임 1 미매칭 점들 -> 은은한 어두운 분홍색/빨간색 (0, 0, 150)
        for i1 in range(len(kpts1_np)):
            if i1 not in matched_set1:
                pt1 = (int(kpts1_np[i1, 0]), int(kpts1_np[i1, 1]))
                cv2.circle(canvas_top, pt1, 1, (0, 0, 140), -1)

        # 프레임 2 미매칭 점들 -> 은은한 어두운 하늘색/블루 (150, 0, 0)
        for i2 in range(len(kpts2_np)):
            if i2 not in matched_set2:
                pt2 = (int(kpts2_np[i2, 0] + x_offset), int(kpts2_np[i2, 1]))
                cv2.circle(canvas_top, pt2, 1, (140, 0, 0), -1)

        # 1-2. 매칭에 성공한 정예 포인트쌍을 그 위에 오버레이 (선명하게 보이도록 나중에 그림)
        for i1, i2 in zip(xf_idx1_np, xf_idx2_np):
            pt1 = (int(kpts1_np[i1, 0]), int(kpts1_np[i1, 1]))
            pt2 = (int(kpts2_np[i2, 0] + x_offset), int(kpts2_np[i2, 1]))

            # 매칭 성공점은 강조 컬러로 렌더링
            cv2.circle(canvas_top, pt1, 3, (100, 0, 255), -1)  # 원본 매칭 마젠타 계열
            cv2.circle(canvas_top, pt2, 3, (255, 100, 0), -1)  # 원본 매칭 시안 계열
            cv2.line(canvas_top, pt1, pt2, (0, 255, 255), 1, lineType=cv2.LINE_AA)

        cv2.putText(canvas_top, f" [METHOD A] Pure XFeat Descriptor Matching (Total: {len(xf_idx1_np)} pts)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # ====================================================================
        # 🖼️ 2단: 💡 [박사님 핵심 요청] Pure DINOv2 Object Semantic Matching (smat)
        # ====================================================================
        canvas_middle = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas_middle[:, :w] = img1_rgb
        canvas_middle[:, w:] = img2_rgb

        smat_link_count = 0
        if smat is not None and A_k1_f1_np is not None and A_k2_f2_np is not None:
            smat_np = smat.cpu().numpy() if torch.is_tensor(smat) else smat
            # smat 주소 공간에서 문턱값을 넘긴 모든 시맨틱 매칭 후보 쌍 추출 [K1, K2]
            s_k1_list, s_k2_list = np.where(smat_np > smat_thresh)

            # 은은하게 대비를 보여주기 위해 일괄 단색(예: 하늘색) 선 렌더링
            semantic_color = (255, 191, 0)  # Deep Sky Blue 느낌

            for sk1, sk2 in zip(s_k1_list.tolist(), s_k2_list.tolist()):
                k1_pts = kpts1_np[A_k1_f1_np[sk1]]
                k2_pts = kpts2_np[A_k2_f2_np[sk2]]

                if len(k1_pts) == 0 or len(k2_pts) == 0:
                    continue

                # 사물 영역 앵커 중심점 역산
                c1_x = int((k1_pts[:, 0].min() + k1_pts[:, 0].max()) / 2)
                c1_y = int((k1_pts[:, 1].min() + k1_pts[:, 1].max()) / 2)
                c2_x = int((k2_pts[:, 0].min() + k2_pts[:, 0].max()) / 2 + x_offset)
                c2_y = int((k2_pts[:, 1].min() + k2_pts[:, 1].max()) / 2)

                # 아직 기하학적으로 검증되지 않아 "모호한 상태"의 시맨틱 링킹 선 작도
                cv2.line(canvas_middle, (c1_x, c1_y), (c2_x, c2_y), semantic_color, 1, lineType=cv2.LINE_AA)
                cv2.circle(canvas_middle, (c1_x, c1_y), 3, (255, 255, 255), -1)
                cv2.circle(canvas_middle, (c2_x, c2_y), 3, (255, 255, 255), -1)
                smat_link_count += 1

            cv2.putText(canvas_middle,
                        f" [METHOD B] Pure DINOv2 Object Semantic Matching via 'smat' (Thresh: {smat_thresh}, Links: {smat_link_count})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, semantic_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas_middle, " [METHOD B] smat or affinity matrices not provided",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # ====================================================================
        # 🖼️ 3단: Our Object-Aware Topology Linking (최종 테이블 검증 완료 단)
        # ====================================================================
        canvas_bottom = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas_bottom[:, :w] = img1_rgb
        canvas_bottom[:, w:] = img2_rgb

        p_h, p_w = grid_shape

        match_counts = final_point_pairs_mask.sum(dim=-1)
        valid_k1_all, valid_k2_all = torch.where(match_counts > 0)

        np.random.seed(42)
        colors = np.random.randint(50, 255, size=(len(valid_k1_all) + 1, 3), dtype=int)

        for pair_idx, (k1, k2) in enumerate(zip(valid_k1_all.tolist(), valid_k2_all.tolist())):
            color = tuple(int(c) for c in colors[pair_idx])

            valid_f1_indices = torch.where(final_point_pairs_mask[k1, k2])[0]
            valid_f2_indices = match12_table[k1, k2, valid_f1_indices]

            f1_np = valid_f1_indices.cpu().numpy()
            f2_np = valid_f2_indices.cpu().numpy()

            if len(f1_np) == 0:
                continue

            k1_pts = kpts1_np[f1_np]
            k2_pts = kpts2_np[f2_np]

            bbox1 = (int(k1_pts[:, 0].min()), int(k1_pts[:, 1].min()), int(k1_pts[:, 0].max()), int(k1_pts[:, 1].max()))
            bbox2 = (int(k2_pts[:, 0].min()), int(k2_pts[:, 1].min()), int(k2_pts[:, 0].max()), int(k2_pts[:, 1].max()))

            cv2.rectangle(canvas_bottom, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), color, 2)
            cv2.rectangle(canvas_bottom, (bbox2[0] + x_offset, bbox2[1]), (bbox2[2] + x_offset, bbox2[3]), color, 2)

            for pt in k1_pts:
                gx, gy = int(pt[0] / p_w), int(pt[1] / p_h)
                cv2.rectangle(canvas_bottom, (int(gx * p_w), int(gy * p_h)), (int((gx + 1) * p_w), int((gy + 1) * p_h)),
                              color, 1)
                cv2.circle(canvas_bottom, (int(pt[0]), int(pt[1])), 2, color, -1)
            for pt in k2_pts:
                gx, gy = int(pt[0] / p_w), int(pt[1] / p_h)
                cv2.rectangle(canvas_bottom, (int(gx * p_w + x_offset), int(gy * p_h)),
                              (int((gx + 1) * p_w + x_offset), int((gy + 1) * p_h)), color, 1)
                cv2.circle(canvas_bottom, (int(pt[0] + x_offset), int(pt[1])), 2, color, -1)

            c1 = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
            c2 = (int((bbox2[0] + bbox2[2]) / 2 + x_offset), int((bbox2[1] + bbox2[3]) / 2))
            cv2.line(canvas_bottom, c1, c2, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas_bottom, c1, 4, (255, 255, 255), -1)
            cv2.circle(canvas_bottom, c2, 4, (255, 255, 255), -1)

        cv2.putText(canvas_bottom, f" [METHOD C] Our Object-Aware Topology Linking (Active Links: {len(valid_k1_all)})",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # ====================================================================
        # 3단 조립
        # ====================================================================
        triple_frame = np.vstack([canvas_top, canvas_middle, canvas_bottom])

        if save_path:
            cv2.imwrite(save_path, triple_frame)
            print(f"💾 [3단 smat 대조 완료] 결과물 저장 완료: {save_path}")

        try:
            cv2.imshow("Triple Verification: XFeat (Top) vs smat Semantic (Mid) vs Ours (Bottom)", triple_frame)
            cv2.waitKey(1)
        except Exception:
            pass

    def interactive_single_image_heatmap(self, img_rgb, x_cat, detections, mat, patch_size=14, sim_threshold=0.65):
        """
        단일 이미지 내에서 특정 패치를 클릭하여,
        해당 패치와 이미지 내 다른 모든 객체/패치 간의 Self-Similarity를 시각화합니다.
        """
        # Feature Map 차원 정보 추출
        _, D, H_p, W_p = x_cat.shape
        H, W = img_rgb.shape[:2]

        # 연산 속도 향상을 위해 벡터들 사전 정규화
        mat_norm = F.normalize(mat, p=2, dim=1) if mat.shape[0] > 0 else None

        # 패치 단위 비교를 위해 x_cat Flatten 및 정규화: [1, D, H_p, W_p] -> [D, H_p*W_p]
        x_cat_flat = x_cat.view(D, -1)
        x_cat_norm = F.normalize(x_cat_flat, p=2, dim=0)

        # 1x3 Subplot 설정
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))

        axes[0].imshow(img_rgb)
        axes[0].set_title("1. Original Image (Click Here!)", fontsize=13, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(img_rgb)
        axes[1].set_title("2. Object-level Self-Similarity", fontsize=13, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(img_rgb)
        axes[2].set_title(f"3. Patch-level Self-Similarity (>{sim_threshold})", fontsize=13, fontweight='bold')
        axes[2].axis('off')

        # 상태 저장용 딕셔너리
        plot_state = {'point': None, 'heatmap_obj': None, 'heatmap_patch': None, 'cbar_obj': None, 'cbar_patch': None}

        def onclick(event):
            if event.inaxes != axes[0]:
                return

            x, y = int(event.xdata), int(event.ydata)

            # 1. 좌표 변환 및 클릭된 지점의 'Query Vector' 추출
            px = min(max(x // patch_size, 0), W_p - 1)
            py = min(max(y // patch_size, 0), H_p - 1)

            query_vec = x_cat[0, :, py, px].unsqueeze(0)  # [1, D]
            query_vec = F.normalize(query_vec, p=2, dim=1)

            # ==========================================
            # [Visual 2] Object-level Similarity (동일 이미지 내)
            # ==========================================
            obj_sim_overlay = np.zeros((H, W), dtype=np.float32)
            if mat_norm is not None:
                obj_similarities = torch.mm(query_vec, mat_norm.t()).squeeze(0).cpu().numpy()
                for k in range(len(obj_similarities)):
                    mask = detections.mask[k]
                    obj_sim_overlay[mask] = np.maximum(obj_sim_overlay[mask], obj_similarities[k])

            # ==========================================
            # [Visual 3] Patch-level Similarity (동일 이미지 내)
            # ==========================================
            # [1, D] @ [D, H_p*W_p] -> [1, H_p*W_p]
            patch_similarities = torch.mm(query_vec, x_cat_norm).view(H_p, W_p).cpu().numpy()

            # 원본 해상도로 리사이즈 (Bilinear 보간법 사용)
            patch_sim_resized = cv2.resize(patch_similarities, (W, H), interpolation=cv2.INTER_LINEAR)

            # ==========================================
            # 화면 갱신 (Rendering)
            # ==========================================
            if plot_state['point']: plot_state['point'].remove()
            if plot_state['heatmap_obj']: plot_state['heatmap_obj'].remove()
            if plot_state['heatmap_patch']: plot_state['heatmap_patch'].remove()

            # 1. 클릭 위치 표시
            plot_state['point'], = axes[0].plot(x, y, 'r*', markersize=15, markeredgecolor='white')

            # 2. 객체 히트맵 그리기 (마스크 없는 부분은 투명하게)
            masked_obj_heatmap = np.ma.masked_where(obj_sim_overlay == 0, obj_sim_overlay)
            plot_state['heatmap_obj'] = axes[1].imshow(masked_obj_heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)

            # 3. 패치 히트맵 그리기 (Threshold 미만인 부분은 투명하게)
            masked_patch_heatmap = np.ma.masked_where(patch_sim_resized < sim_threshold, patch_sim_resized)
            plot_state['heatmap_patch'] = axes[2].imshow(masked_patch_heatmap, cmap='jet', alpha=0.6,
                                                         vmin=sim_threshold,
                                                         vmax=1)

            # 컬러바 생성 (최초 1회)
            if plot_state['cbar_obj'] is None:
                plot_state['cbar_obj'] = fig.colorbar(plot_state['heatmap_obj'], ax=axes[1], fraction=0.046, pad=0.04)
            if plot_state['cbar_patch'] is None:
                plot_state['cbar_patch'] = fig.colorbar(plot_state['heatmap_patch'], ax=axes[2], fraction=0.046,
                                                        pad=0.04)

            fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

    def interactive_layered_heatmap(self,   img1_rgb, x_cat1, img2_rgb, x_cat2, detections2, mat2, patch_size=14,
                                    sim_threshold=0.7):
        """
        타겟 이미지 하나에 객체 유사도와 패치 유사도를 동일한 색상 스케일(0~1, jet)로 겹쳐서 시각화합니다.
        """
        # 쿼리 및 타겟 Feature Map 차원 정보
        _, D, H_p1, W_p1 = x_cat1.shape
        _, _, H_p2, W_p2 = x_cat2.shape
        H2, W2 = img2_rgb.shape[:2]

        # ==========================================
        # [사전 연산] 쿼리 및 타겟 전체 피처 맵 정규화
        # ==========================================
        # 1. Query 피처 맵(x_cat1) 전체를 채널(D) 차원 기준으로 미리 정규화 ([1, D, H, W] -> dim=1)
        x_cat1_norm = F.normalize(x_cat1, p=2, dim=1)

        # 2. 타겟 객체 벡터(mat2) 사전 정규화
        mat2_norm = F.normalize(mat2, p=2, dim=1) if mat2.shape[0] > 0 else None

        # 3. 타겟 패치 피처 맵(x_cat2) Flatten 및 정규화
        x_cat2_flat = x_cat2.view(D, -1)
        x_cat2_norm = F.normalize(x_cat2_flat, p=2, dim=0)

        # 1x2 Subplot 설정
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        axes[0].imshow(img1_rgb)
        axes[0].set_title("1. Query Image (Click Here!)", fontsize=13, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(img2_rgb)
        axes[1].set_title("2. Target Similarity (Object + Patch Overlay)", fontsize=13, fontweight='bold')
        axes[1].axis('off')

        # 상태 저장용 딕셔너리
        plot_state = {
            'point': None,
            'heatmap_obj': None,
            'heatmap_patch': None,
            'cbar': None
        }

        def onclick(event):
            if event.inaxes != axes[0]:
                return

            x, y = int(event.xdata), int(event.ydata)

            # 1. 좌표 변환
            px = min(max(x // patch_size, 0), W_p1 - 1)
            py = min(max(y // patch_size, 0), H_p1 - 1)

            # 2. 이미 정규화된 쿼리 피처 맵에서 클릭한 위치의 패치 벡터만 추출
            query_vec = x_cat1_norm[0, :, py, px].unsqueeze(0)  # 차원: [1, D]

            # ==========================================
            # [Layer 1] Object-level Similarity
            # ==========================================
            obj_sim_overlay = np.zeros((H2, W2), dtype=np.float32)
            if mat2_norm is not None:
                obj_similarities = torch.mm(query_vec, mat2_norm.t()).squeeze(0).cpu().numpy()
                for k in range(len(obj_similarities)):
                    mask = detections2.mask[k]
                    obj_sim_overlay[mask] = np.maximum(obj_sim_overlay[mask], obj_similarities[k])

            # ==========================================
            # [Layer 2] Patch-level Similarity
            # ==========================================
            patch_similarities = torch.mm(query_vec, x_cat2_norm).view(H_p2, W_p2).cpu().numpy()
            patch_sim_resized = cv2.resize(patch_similarities, (W2, H2), interpolation=cv2.INTER_LINEAR)

            # ==========================================
            # 화면 갱신 (Rendering)
            # ==========================================
            # 기존 플롯 제거
            if plot_state['point']: plot_state['point'].remove()
            if plot_state['heatmap_obj']: plot_state['heatmap_obj'].remove()
            if plot_state['heatmap_patch']: plot_state['heatmap_patch'].remove()

            # 클릭 위치 렌더링
            plot_state['point'], = axes[0].plot(x, y, 'r*', markersize=15, markeredgecolor='white')

            # [Layer 1 렌더링] 배경 베이스용 객체 히트맵 (투명도 0.4, 0~1 스케일)
            masked_obj_heatmap = np.ma.masked_where(obj_sim_overlay == 0, obj_sim_overlay)
            plot_state['heatmap_obj'] = axes[1].imshow(
                masked_obj_heatmap, cmap='jet', alpha=0.4, vmin=0, vmax=1
            )

            # [Layer 2 렌더링] 패치 히트맵 (투명도 0.8, 0~1 스케일)
            masked_patch_heatmap = np.ma.masked_where(patch_sim_resized < sim_threshold, patch_sim_resized)
            plot_state['heatmap_patch'] = axes[1].imshow(
                masked_patch_heatmap, cmap='jet', alpha=0.8, vmin=0, vmax=1
            )

            # 값이 항상 존재하는 객체 히트맵을 기준으로 컬러바 1회 생성
            if plot_state['cbar'] is None:
                plot_state['cbar'] = fig.colorbar(plot_state['heatmap_obj'], ax=axes[1], fraction=0.046, pad=0.04)
                plot_state['cbar'].set_label("Cosine Similarity", rotation=270, labelpad=15)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

    def visualize_cls_attention_opencv(self, img_bgr, cls_attn, grid_shape, alpha=0.6, window_name="CLS Attention Map"):
        """
        DINOv2의 CLS Attention 맵을 원본 이미지 위에 JET 컬러맵으로 오버레이하여 시각화합니다.

        Args:
            img_bgr (np.ndarray): OpenCV 원본 이미지 (H, W, 3)
            cls_attn (torch.Tensor): extract_features_with_attention에서 나온 (B, N) 또는 (N,) 형태의 텐서
            grid_shape (tuple): 패치 그리드 크기 (H_p, W_p) 예: (34, 45)
            alpha (float): 어텐션 맵의 투명도 (0.0 ~ 1.0)
        """
        H, W, _ = img_bgr.shape
        H_p, W_p = grid_shape

        # 1. 텐서 안전하게 정제 (Batch 차원 제거 및 CPU/Numpy 변환)
        if isinstance(cls_attn, torch.Tensor):
            if cls_attn.dim() == 2:
                cls_attn = cls_attn[0]  # Batch 0 선택 -> [N]
            cls_attn_np = cls_attn.detach().cpu().numpy()
        else:
            cls_attn_np = cls_attn

        # 2. 1D 패치 배열을 2D 그리드 형태로 리셰이프 -> [H_p, W_p]
        attn_grid = cls_attn_np.reshape(H_p, W_p)

        # 3. 민맥스 정규화 (0.0 ~ 1.0 스케일링) 및 8비트 이미지화
        attn_min = attn_grid.min()
        attn_max = attn_grid.max()
        if (attn_max - attn_min) > 1e-8:
            attn_norm = (attn_grid - attn_min) / (attn_max - attn_min)
        else:
            attn_norm = np.zeros_like(attn_grid)

        attn_8bit = (attn_norm * 255).astype(np.uint8)

        # 4. 패치 해상도(34x45)를 원본 이미지 해상도(H, W)로 고품질 확대 (INTER_CUBIC)
        attn_resized = cv2.resize(attn_8bit, (W, H), interpolation=cv2.INTER_CUBIC)

        # 5. 의사 컬러맵(Color Map) 적용 (낮은 값: 파란색, 높은 값/어텐션 집중: 빨간색)
        heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)

        # 6. 원본 이미지와 히트맵 블렌딩 (Overlay)
        blended = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)

        # 7. 시각화 텍스트 삽입 (디버깅용 정보)
        cv2.putText(blended, f"Attention Grid: {W_p}x{H_p}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 화면에 띄우기
        cv2.imshow(window_name, blended)
        cv2.waitKey(1)  # ZeroMQ나 실시간 스트리밍 루프 내에서 블로킹 방지

        return blended

    def visualize_new_structural_grouping(self, img_bgr, centroids, affinity, group_assignments, heatmap_sim_np,
                                          n_components, grid_shape=(34, 45), alpha=0.5):

        H_img, W_img, _ = img_bgr.shape
        H_p, W_p = grid_shape
        K, N = affinity.shape  # K: 대표 앵커 개수, N: 전체 패치 개수(1530)

        # 1. 고유 그룹 분류 및 결정론적 색상표 룩업
        unique_groups = np.unique(group_assignments)

        # 디버깅 가시성을 위해 무작위 고유 컬러 맵 구축
        color_map = {g_id: self.colors[idx] for idx, g_id in enumerate(unique_groups)}

        # 2. 채색 도화지 및 기하학적 스케일 팩터 준비
        mask_overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        hard_mask_kn = affinity.cpu().numpy() if hasattr(affinity, "cpu") else np.array(affinity)
        centroids_np = centroids.cpu().numpy() if hasattr(centroids, "cpu") else np.array(centroids)

        scale_y = H_img / H_p
        scale_x = W_img / W_p
        sample_text_positions = []

        # ====================================================================
        # PHASE 1: 외부 주입 group_assignments 기반 세력권 채색 및 센트로이드 추출
        # ====================================================================
        for k_idx in range(K):
            belonging_group = group_assignments[k_idx]
            color = color_map[belonging_group]

            # 💡 진짜 대표 샘플(Centroid) 패치 인덱스를 가져와 2D 픽셀 중심 좌표 산출
            c_patch_idx = centroids_np[k_idx]
            c_y_p = c_patch_idx // W_p
            c_x_p = c_patch_idx % W_p

            sample_center_x = int((c_x_p + 0.5) * scale_x)
            sample_center_y = int((c_y_p + 0.5) * scale_y)

            # 위상 매핑 주소록에 저장
            sample_text_positions.append((sample_center_x, sample_center_y, belonging_group, k_idx))

            # 해당 시드 채널에 포섭된 하위 자식 패치 영역들 일괄 채색
            member_patch_indices = np.where(hard_mask_kn[k_idx] > 0.5)[0]
            for p_idx in member_patch_indices:
                y_p = p_idx // W_p
                x_p = p_idx % W_p

                y_start, y_end = int(y_p * scale_y), int((y_p + 1) * scale_y)
                x_start, x_end = int(x_p * scale_x), int((x_p + 1) * scale_x)
                mask_overlay[y_start:y_end, x_start:x_end] = color

        # 원본 이미지 위에 알파 블렌딩 합성
        fused_img = cv2.addWeighted(img_bgr, 1.0 - alpha, mask_overlay, alpha, 0)

        # ====================================================================
        # PHASE 2: 💥 [박사님 요구 스펙] 동일 그룹 간 위상 연결선 및 날것의 코사인 유사도 텍스트 작도
        # ====================================================================
        line_overlay = fused_img.copy()

        for i in range(len(sample_text_positions)):
            for j in range(i + 1, len(sample_text_positions)):
                pt1_x, pt1_y, g_id1, k_i = sample_text_positions[i]
                pt2_x, pt2_y, g_id2, k_j = sample_text_positions[j]

                # 두 시드가 동일 그룹 제국으로 판정되어 묶였다면 위상선 개설
                if g_id1 == g_id2:
                    line_color = color_map[g_id1]
                    cv2.line(line_overlay, (pt1_x, pt1_y), (pt2_x, pt2_y), line_color, 2, cv2.LINE_AA)

                    # 💡 [핵심 가치 추가]: 두 시드가 '실제 몇의 코사인 유사도'로 묶였는지 선 정중앙에 수치 인쇄
                    cos_sim_val = heatmap_sim_np[k_i, k_j]
                    mid_x, mid_y = int((pt1_x + pt2_x) / 2), int((pt1_y + pt2_y) / 2)
                    sim_text = f"{cos_sim_val:.2f}"
                    cv2.putText(line_overlay, sim_text, (mid_x - 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                (0, 255, 255), 1, cv2.LINE_AA)

        # 선 레이어 투명도 동기화 융합
        fused_img = cv2.addWeighted(fused_img, 0.3, line_overlay, 0.7, 0)

        # ====================================================================
        # PHASE 3: 센트로이드 노드 좌표계 위에 명확한 그룹 번호 안착
        # ====================================================================
        for s_x, s_y, g_id, _ in sample_text_positions:
            id_text = f"G:{g_id}"
            # 검은색 테두리 그림자
            cv2.putText(fused_img, id_text, (s_x - 12, s_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2,
                        cv2.LINE_AA)
            # 흰색 본문 글자
            cv2.putText(fused_img, id_text, (s_x - 12, s_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # 시스템 모니터링 전역 헤더 박스 임베딩
        title_text = f"New Structural Equivalence Map (Anchors: {K} -> Merged Entities: {n_components})"
        cv2.putText(fused_img, title_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 💡 독립된 단독 OpenCV 창으로 분리 팝업
        cv2.imshow("New_Structural_Equivalence_Map_Window", fused_img)
        cv2.waitKey(1)

        return fused_img

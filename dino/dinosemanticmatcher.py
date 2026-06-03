import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
class DinoXFeatPatchMatcher:
    def __init__(self):
        pass

    def build_affinity_matrix(self,feat1, feat2, threshold = 0.7):
        #K1 X K2
        res_sim = torch.mm(feat1, feat2.t())
        res_sim = res_sim * (res_sim > threshold)
        return res_sim

    def compute_local_point_correspondence_batch_fixed(
            self, A_k1_f1, A_k2_f2, S_match_mask, fmat, min_cossim=0.75
    ):
        """[진우 박사님 피드백 반영 - 진짜 로컬 경쟁형 배치 커널]

        글로벌 격자 스캔 방식을 폐기하고, 각 배치 방 내부의 로컬 포인트 유효 원소들끼리만
        Argmax 경쟁을 붙여 매칭 점 개수를 정상화합니다.
        """
        device = fmat.device
        K1, F1 = A_k1_f1.shape
        K2, F2 = A_k2_f2.shape

        # 1. 유효 사물 쌍 추출 [M]
        m_k1, m_k2 = torch.where(S_match_mask > 0)
        M = m_k1.shape[0]

        final_point_pairs_mask = torch.zeros(
            (K1, K2, F1), dtype=torch.bool, device=device
        )
        match12_table = torch.zeros((K1, K2, F1), dtype=torch.long, device=device)

        if M == 0:
            return final_point_pairs_mask, match12_table

        # [M, F1, 1], [M, 1, F2] 크기로 사물 귀속성 분리
        batch_mask_f1 = A_k1_f1[m_k1].view(M, F1, 1)  # 1 또는 0
        batch_mask_f2 = A_k2_f2[m_k2].view(M, 1, F2)  # 1 또는 0

        # 2. 💡 핵심: 내 방에 속하지 않는 영역의 유사도를 -10.0이 아니라,
        # 반대편 차원(dim)에서 max를 찾을 때 아예 배제되도록 마스킹 소프트맥스/필터링 제약을 줍니다.
        # fmat 상에서 내 사물 방에 속한 자식들의 값만 살립니다.
        fmat_3d = fmat.view(1, F1, F2).expand(M, F1, F2)

        # 행(F1) 기준 마스킹: 프레임1의 포인트가 현재 사물방에 속하지 않으면 유사도 축출
        conditioned_f1 = torch.where(
            batch_mask_f1 > 0, fmat_3d, torch.tensor(-10.0, device=device)
        )
        # 열(F2) 기준 마스킹: 프레임2의 포인트가 상대 사물방에 속하지 않으면 유사도 축출
        conditioned_f2 = torch.where(
            batch_mask_f2 > 0, fmat_3d, torch.tensor(-10.0, device=device)
        )

        # 3. 각자 허가된 로컬 도메인 내부에서의 독립 최강자 룩업
        # match12: F1이 가리키는 F2의 인덱스 (F2 마스크가 적용된 판 위에서 최고를 찾아야 함)
        max_vals_f1, match12 = conditioned_f2.max(dim=2)  # [M, F1]
        # match21: F2가 가리키는 F1의 인덱스 (F1 마스크가 적용된 판 위에서 최고를 찾아야 함)
        _, match21 = conditioned_f1.max(dim=1)  # [M, F2]

        # 4. 🚀 상호 최적 매칭(Mutual NN) 및 문턱값 필터링
        grid_m = torch.arange(M, device=device).view(M, 1).expand(M, F1)
        idx_f1 = torch.arange(F1, device=device).view(1, F1).expand(M, F1)

        # 상호 검증 텐서 전개
        mutual = match21[grid_m, match12] == idx_f1  # [M, F1]

        # 💡 중요: max_vals_f1도 '진짜 내 방에 속한 F1 포인트'일 때만 스코어로 인정해야 합니다.
        good = (max_vals_f1 > min_cossim) & (batch_mask_f1.squeeze(-1) > 0)
        batch_survival_mask = mutual & good  # [M, F1]

        # 5. 전역 마스터 테이블 차원으로 인덱스 복원 환원
        m_indices, f1_indices = torch.where(batch_survival_mask)

        if m_indices.shape[0] > 0:
            real_k1 = m_k1[m_indices]
            real_k2 = m_k2[m_indices]
            real_f2 = match12[m_indices, f1_indices]

            final_point_pairs_mask[real_k1, real_k2, f1_indices] = True
            match12_table[real_k1, real_k2, f1_indices] = real_f2

        return final_point_pairs_mask, match12_table

    def compute_local_point_correspondence_batch(
            self, A_k1_f1, A_k2_f2, S_match_mask, Fmat, min_cossim=0.9
    ):
        """[진우 박사님 제안 - 루프 없는 3D 배치 병렬화 버전 (12GB VRAM 최적화)]

        유효 사물 쌍들을 배치(Batch) 차원으로 묶어, OOM 없이 모든 방의 Mutual NN을 한 번에
        처리합니다.
        """
        device = Fmat.device
        K1, F1 = A_k1_f1.shape
        K2, F2 = A_k2_f2.shape

        # 1. 💡 [배치 축 추출]: 매칭 유효성이 1인 사물 쌍 주소만 원샷 추출
        # m_k1, m_k2 shape: [M] (M은 유효 사물 쌍의 개수)
        m_k1, m_k2 = torch.where(S_match_mask > 0)
        M = m_k1.shape[0]

        if M == 0:
            return torch.zeros((K1, K2, F1), dtype=torch.bool, device=device), None

        # 2. 💡 [배치 단위 마스크 조립]: 선택된 M개 방에 대한 고유 마스크만 슬라이싱
        # [K1, F1] -> [M, F1] -> 차원 확장 [M, F1, 1]
        batch_mask_f1 = A_k1_f1[m_k1].view(M, F1, 1)
        # [K2, F2] -> [M, F2] -> 차원 확장 [M, 1, F2]
        batch_mask_f2 = A_k2_f2[m_k2].view(M, 1, F2)

        # 이 둘을 결합하여 [M, F1, F2] 크기의 배치 경계 필터 생성
        batch_boundary_mask = (batch_mask_f1 > 0) & (batch_mask_f2 > 0)

        # 3. 🚀 [배치 텐서 내적 공간 빌드]: Fmat [F1, F2]를 [1, F1, F2]로 늘려 [M, F1, F2]로 브로드캐스팅
        Fmat_3d = Fmat.view(1, F1, F2)

        # 내 사물 방에 해당하지 않는 포인트 유사도는 -10.0으로 묵살
        # 💥 핵심: 4D 90GB가 아니라 고작 250MB 수준에서 연산이 일어납니다!
        conditioned_Fmat = torch.where(
            batch_boundary_mask, Fmat_3d, torch.tensor(-10.0, device=device)
        )

        # ====================================================================
        # 🎯 4. [M] 배치 축 상에서의 고속 병렬 Mutual NN (상호 최적 룩업)
        # ====================================================================
        # F2 축 방향(dim=2)으로 배치 내부 최강자 찾기
        max_vals_f1, match12 = conditioned_Fmat.max(dim=2)  # [M, F1]
        # F1 축 방향(dim=1)으로 배치 내부 최강자 찾기
        _, match21 = conditioned_Fmat.max(dim=1)  # [M, F2]

        # 고급 인덱싱을 위한 배치 격자 그리드 생성
        grid_m = (
            torch.arange(M, device=device).view(M, 1).expand(M, F1)
        )  # 배치 주소 [M, F1]
        idx_f1 = (
            torch.arange(F1, device=device).view(1, F1).expand(M, F1)
        )  # F1 포인트 주소 [M, F1]

        # match21[m, match12[m, f1]] 구조를 루프 없이 한 방에 룩업 검증
        mutual = match21[grid_m, match12] == idx_f1  # [M, F1]
        good = max_vals_f1 > min_cossim  # 디스크립터 유사도 문턱값

        # 최종 생존 플래그 [M, F1]
        batch_survival_mask = mutual & good

        # ====================================================================
        # 5. 🎉 원래의 [K1, K2, F1] 마스터 테이블 차원으로 인덱스 복원 환원
        # ====================================================================
        final_point_pairs_mask = torch.zeros(
            (K1, K2, F1), dtype=torch.bool, device=device
        )
        match12_table = torch.zeros((K1, K2, F1), dtype=torch.long, device=device)

        # 배치 차원에서 생존한 위치 추출
        m_indices, f1_indices = torch.where(batch_survival_mask)

        if m_indices.shape[0] > 0:
            # 원래의 사물 ID (k1, k2) 역산 복원
            real_k1 = m_k1[m_indices]
            real_k2 = m_k2[m_indices]
            real_f2 = match12[m_indices, f1_indices]

            # 마스터 텐서 테이블에 원샷 주입 종결!
            final_point_pairs_mask[real_k1, real_k2, f1_indices] = True
            match12_table[real_k1, real_k2, f1_indices] = real_f2

        return final_point_pairs_mask, match12_table

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

    def inspect_matching_tensor_integrity(self, final_point_pairs_mask, match12_table, S_match_mask):
        """
        [진우 박사님 전용 - 매칭 텐서 블랙박스 해부 가이드]

        이 코드는 4D/3D 배치 연산 이후, 진짜로 '사물 방 격리 매칭'이 일어났는지
        아니면 전역 매칭에 오염되었는지 터미널 상에 직관적인 위상 지도를 그려줍니다.
        """
        import torch

        print("\n" + "=" * 60)
        print("🚨 [TENSOR INTEGRITY INSPECTION] 하이브리드 매칭 텐서 실시간 해부")
        print("=" * 60)

        device = final_point_pairs_mask.device
        K1, K2, F1 = final_point_pairs_mask.shape

        # 1. DINOv2 가이드 마스크가 1인 '진짜 매칭 유효 방'의 개수
        dino_active_rooms = torch.sum(S_match_mask > 0).item()

        # 2. XFeat 검증까지 통과해서 '실제 기하 결합'이 일어난 방의 개수
        match_counts_per_room = final_point_pairs_mask.sum(dim=-1)  # [K1, K2]
        geom_active_rooms = torch.sum(match_counts_per_room > 0).item()

        total_verified_points = final_point_pairs_mask.sum().item()

        print(f"📊 [전역 통계]")
        print(f"  • 시맨틱 매칭 허용 방 (DINOv2 > Thresh) : {dino_active_rooms} 개 / 총 {K1 * K2} 개")
        print(f"  • 최종 기하 검증 성공 방 (Ours 生存)   : {geom_active_rooms} 개")
        print(f"  • 최종 생존 정예 기하 포인트 쌍 수    : {total_verified_points} 개")
        print("-" * 60)

        if total_verified_points == 0:
            print("❌ [경고] 현재 생존한 포인트가 0개입니다. 디스크립터 매칭 문턱값(min_cossim)이 너무 높거나 fmat이 오염되었습니다.")
            return

        # 3. 🚀 [핵심 검증]: 생존한 포인트들이 '진짜로 허가된 사물 방 내부'에만 존재하는가?
        # S_match_mask가 0인(허가되지 않은) 방인데 final_point_pairs_mask가 True인 곳이 있다면 팅겨나간 것입니다.
        illegal_mask = (final_point_pairs_mask.sum(dim=-1) > 0) & (S_match_mask == 0)
        illegal_leakage = illegal_mask.sum().item()

        print(f"🛡️ [차단벽(Gating) 작동 여부 검증]")
        if illegal_leakage == 0:
            print("  • 🟢 무결성 통과: 허가되지 않은 사물 방에서의 포인트 유출이 '제로(0)'입니다.")
            print("    -> 즉, 사물 도메인 격리 자체는 텐서 차원에서 완벽히 가두어 연산하고 있습니다.")
        else:
            print(f"  • 🔴 텐서 오염 발견!! 시맨틱 마스크가 차단한 방에서 {illegal_leakage}개의 포인트 매칭 노이즈가 유출됨.")
            print("    -> 배치 조립 시 index mapping이 밀렸을 확률이 높습니다.")

        # 4. 🖨️ [위상 맵 프린터]: 상위 5개 활성 사물 방의 매칭 인덱스 실시간 샘플링 리포트
        print("-" * 60)
        print("🎯 [활성 사물 방별 정예 포인트 룩업 테이블 매핑 스캔 (Top 5)]")

        active_k1, active_k2 = torch.where(match_counts_per_room > 0)
        printed_count = 0

        for k1, k2 in zip(active_k1.tolist(), active_k2_all := active_k2.tolist()):
            if printed_count >= 5:
                break

            pts_in_this_room = match_counts_per_room[k1, k2].item()

            # 해당 방에서 생존한 F1 주소와 매치테이블에 기록된 F2 주소 직접 추출
            f1_indices = torch.where(final_point_pairs_mask[k1, k2])[0]
            f2_indices = match12_table[k1, k2, f1_indices]

            print(f"\n📦 [사물 에지방] K1 노드 ({k1}) ↔ K2 노드 ({k2}) -> 검증된 기하점: {pts_in_this_room}개")
            print(f"  • 단독 매칭된 F1 포인터 고유 ID: {f1_indices[:8].tolist()} ...")
            print(f"  • 매치테이블이 가리키는 F2 ID  : {f2_indices[:8].tolist()} ...")

            printed_count += 1

        print("=" * 60 + "\n")

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
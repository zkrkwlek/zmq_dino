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
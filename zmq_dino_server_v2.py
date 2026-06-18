"""
zmq_dino_server_v2.py
────────────────────────────────────────────────────────────────
기존 zmq_dino_server.py 의 업그레이드 버전.

주요 변경:
  1. DinoSemanticObjectExtractorV2 사용
       → 신규 파이프라인 (compute_anchor_patch_context / merge_anchors_heatmap
          / _apply_group_and_recompute / filter_memory_anchors_cross_frame)
       → avg_vec를 pure 패치 기반으로 계산 (TS1 수정)

  2. CA 제거
       → 메모리 앵커 품질을 단일 프레임 앵커 정제로 확보하는 방향
       → 향후 메모리 뱅크 벡터를 현재 프레임에 반영하는 별도 구조로 대체 예정

  3. GlobalObjectBank 추가 (I3 수정)
       → gid 등록 / EMA 갱신 / 코사인 유사도 Re-ID

  4. pick_best_target — src 기반 다중 기기 필터 (TS3 수정)

  5. Multi-Response Masking → matching 전처리 연결 (I2 수정)
────────────────────────────────────────────────────────────────
"""

import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zmq
from queue import Queue

from dino.dinoencoder import DinoV2Encoder
from dino.patchcluster_v2 import DinoSemanticObjectExtractorV2
from communication.commuprocessmanager import NotificationManager, DownloadDataManager
from communication.inferencedatamanager import (
    ImageDataManager, XFeatDataManager, SaladDataManager, DINODataManager
)
from dino.dinosemanticmatcher import DinoXFeatPatchMatcher
from dino.visualizer import DinoPatchVisualizer
from dino.pipeline_debug_visualizer import PipelineDebugVisualizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_PATH    = '../dinov2'
DINO_MODEL   = 'dinov2_vits14'
WEIGHTS_PATH = './dinov2_vits14.pth'

SERVER_NAME = b"dino_server"
RECV_KW = [b"image", b"salad_res", b"xfeat_kp_res", b"xfeat_desc_res"]

# ── 디버그 시각화 설정 ────────────────────────────────────────────────────────
DEBUG_VIS        = False          # True 로 바꾸면 각 Phase 시각화 저장
DEBUG_VIS_DIR    = './debug_v2'   # 저장 경로
DEBUG_VIS_PHASES = None           # None=전체  예: {1, 4, 6} 일부만 활성화

task_queue = Queue(maxsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# GlobalObjectBank
# ─────────────────────────────────────────────────────────────────────────────

class GlobalObjectBank:
    """
    gid(글로벌 객체 ID) 기반 객체 메모리 뱅크 (I3 구현).

    - register_or_update : 앵커 벡터를 받아 gid 부여 또는 EMA 갱신
    - query              : 현재 앵커와 가장 유사한 gid 반환 (Re-ID)
    - get_all_vecs       : 전체 뱅크 벡터 반환

    threshold:
        reid_threshold  : 동일 객체 판단 코사인 유사도 하한 (기본 0.75)
        ema_alpha       : EMA 갱신 시 새 벡터 비율 (기본 0.3)
    """

    def __init__(self, reid_threshold: float = 0.75, ema_alpha: float = 0.3):
        self.reid_threshold = reid_threshold
        self.ema_alpha      = ema_alpha
        self._bank: dict[int, dict] = {}
        self._next_gid = 0

    @property
    def size(self) -> int:
        return len(self._bank)

    def register_or_update(self, avg_vecs: torch.Tensor,
                           frame_id: str, src: bytes) -> list[int]:
        """
        Args:
            avg_vecs : [K, D] L2 정규화 앵커 벡터
        Returns:
            gids     : [K] int
        """
        K = avg_vecs.shape[0]
        if K == 0:
            return []

        if len(self._bank) == 0:
            return [self._new_gid(avg_vecs[k], frame_id, src) for k in range(K)]

        bank_vecs = self._get_bank_matrix(avg_vecs.device)  # [M, D]
        bank_gids = list(self._bank.keys())
        sim       = torch.mm(avg_vecs, bank_vecs.t())        # [K, M]

        gids = []
        for k in range(K):
            best_sim, best_idx = sim[k].max(dim=0)
            if best_sim.item() >= self.reid_threshold:
                matched_gid = bank_gids[best_idx.item()]
                self._update_gid(matched_gid, avg_vecs[k], frame_id, src)
                gids.append(matched_gid)
            else:
                gids.append(self._new_gid(avg_vecs[k], frame_id, src))
        return gids

    def query(self, avg_vecs: torch.Tensor):
        if len(self._bank) == 0 or avg_vecs.shape[0] == 0:
            return [], torch.zeros(avg_vecs.shape[0])
        bank_vecs = self._get_bank_matrix(avg_vecs.device)
        bank_gids = list(self._bank.keys())
        sim = torch.mm(avg_vecs, bank_vecs.t())
        best_sim, best_idx = sim.max(dim=1)
        matched = [bank_gids[i.item()] if s.item() >= self.reid_threshold else -1
                   for i, s in zip(best_idx, best_sim)]
        return matched, best_sim

    def get_all_vecs(self, device):
        if len(self._bank) == 0:
            return torch.zeros((0, 384), device=device), []
        gids = list(self._bank.keys())
        vecs = torch.stack([self._bank[g]['vec'] for g in gids]).to(device)
        return vecs, gids

    def _new_gid(self, vec, frame_id, src):
        gid = self._next_gid
        self._next_gid += 1
        self._bank[gid] = {
            'vec': vec.detach().cpu(),
            'obs_count': 1,
            'last_frame': frame_id,
            'src': src,
        }
        return gid

    def _update_gid(self, gid, new_vec, frame_id, src):
        old = self._bank[gid]['vec'].to(new_vec.device)
        updated = (1 - self.ema_alpha) * old + self.ema_alpha * new_vec
        self._bank[gid]['vec']        = F.normalize(updated, p=2, dim=0).detach().cpu()
        self._bank[gid]['obs_count'] += 1
        self._bank[gid]['last_frame'] = frame_id

    def _get_bank_matrix(self, device):
        gids = list(self._bank.keys())
        return torch.stack([self._bank[g]['vec'] for g in gids]).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 타겟 프레임 선택 (src 기반 다중 기기 필터)
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_target(list_neigh_frames, cur_src: bytes, cur_fid: bytes,
                     prefer_other_device: bool = True,
                     max_temporal_dist: int = 500):
    """
    SALAD 인접 프레임 중 최적 타겟 선택.

    prefer_other_device=True  : 다른 기기(src) 우선 → 다중 기기 cross-view
    prefer_other_device=False : 동일 기기 temporal 매칭
    """
    cur_fid_int = int(cur_fid.decode()) if cur_fid else 0

    other_device = []
    same_device  = []

    for (tkey, _sim) in list_neigh_frames:   # list_neigh = (key, sim) 튜플
        tsrc, tfid = tkey
        try:
            tfid_int = int(tfid.decode())
        except Exception:
            tfid_int = 0
        temporal_dist = abs(cur_fid_int - tfid_int)

        if tsrc != cur_src:
            other_device.append((tsrc, tfid, temporal_dist))
        elif temporal_dist < max_temporal_dist:
            same_device.append((tsrc, tfid, temporal_dist))

    if prefer_other_device and other_device:
        return other_device[0][:2]   # SALAD 순서 = 유사도 순

    if same_device:
        same_device.sort(key=lambda x: -x[2])
        return same_device[0][:2]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# inference_loop
# ─────────────────────────────────────────────────────────────────────────────

def inference_loop(zmq_socket):
    with torch.no_grad():
        img_mgr    = ImageDataManager()
        xfeat_mgr  = XFeatDataManager()
        salad_mgr  = SaladDataManager()
        dino_mgr   = DINODataManager()
        matcher    = DinoXFeatPatchMatcher()
        obj_bank   = GlobalObjectBank(reid_threshold=0.75)
        visualizer = DinoPatchVisualizer()
        vis_dbg    = PipelineDebugVisualizer(
            output_dir=DEBUG_VIS_DIR,
            enable_phases=DEBUG_VIS_PHASES,
        ) if DEBUG_VIS else None

        while True:
            bundle = task_queue.get()
            src, fid, data = bundle
            stime = time.time()

            if vis_dbg is not None:
                vis_dbg.set_frame_id(fid.decode())

            # ── 1. 데이터 수신 ────────────────────────────────────────────
            img_mgr.process(src, fid, data[b'image'][1])
            salad_mgr.process(src, fid, data[b'salad_res'][1])
            xfeat_mgr.process(src, fid, (data[b'xfeat_kp_res'][1],
                                          data[b'xfeat_desc_res'][1]))

            img1     = img_mgr.get(src, fid)
            img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            kp1, desc1 = xfeat_mgr.get(src, fid)
            kp1, desc1 = kp1.cuda(), desc1.cuda()

            # ── 2. SALAD 인접 프레임 탐색 ─────────────────────────────────
            salad_mgr.calc_neighbor_frames(src, fid)
            list_neigh = salad_mgr.get(src, fid)

            # ── 3. DINOv2 피처 추출 ───────────────────────────────────────
            tensor1, (W, H) = model.preprocess_cv2(img_rgb1)
            x_cat1, attn_1 = model.extract_features_with_attention(
                tensor1.cuda(), model.out_indices,
                patch_size=14, head_indices=[5])
            feat1, H_p, W_p = objpatcher._prepare_features(x_cat1)
            grid_shape = (H_p, W_p)
            t_dino = time.time()

            # Phase 1 디버그
            if vis_dbg is not None:
                vis_dbg.phase1_attention(img1, attn_1[0], grid_shape)
                _bmat_vis = dino_mgr.bind_xfeat_to_patch(kp1, grid_shape)
                vis_dbg.phase1_xfeat_patches(img1, kp1, _bmat_vis, grid_shape)

            # ── 4. 단일 프레임 앵커 정제 ──────────────────────────────────

            # 4-1. NMS 앵커 샘플링
            mask1, sim_mat1 = objpatcher.generate_mask(
                feat1, grid_shape, spatial_radius=3, sim_thresh=0.7)
            sample1 = objpatcher.generate_seeds(attn_1, mask1)          # unified API

            # 4-2. 앵커 컨텍스트 계산
            ctx1 = objpatcher.compute_anchor_patch_context(sample1, feat1)

            # 4-3. 앵커 병합
            group_labels, n_comp = objpatcher.group_anchors(ctx1)       # unified API

            # 4-4. 그룹 확정 + pure 기반 avg_vec
            new_sample_1, new_vom1, group_pure1, new_avg_vec1, new_ctx1 = \
                objpatcher.compute_anchor_response(                       # unified API
                    sample1, group_labels, n_comp, ctx1,
                    feat1, x_cat1, attn_1, grid_shape=grid_shape)

            K_cur = new_sample_1.shape[0]
            print(f"[단일프레임] 앵커 {sample1.shape[0]} → 그룹 {K_cur}")

            # 4-5. Multi-Response Masking
            if K_cur > 0:
                sim_for_mask = torch.mm(new_avg_vec1, feat1.t())
                I_n, overlap_counts = objpatcher.detect_multiresponse(   # unified API
                    sim_for_mask, th_sim=0.60, th_margin=0.12)
                clean_patch_mask = (I_n < 1)
            else:
                N = feat1.shape[0]
                I_n = torch.zeros(N, dtype=torch.long, device=feat1.device)
                overlap_counts = torch.zeros(N, device=feat1.device)
                clean_patch_mask = torch.ones(N, dtype=torch.bool, device=feat1.device)

            # Phase 4 디버그
            if vis_dbg is not None and K_cur > 0:
                vis_dbg.phase4_multiresponse(img1, I_n, grid_shape)

            # 4-6. XFeat 귀속
            bind_xfeat_mat1 = dino_mgr.bind_xfeat_to_patch(kp1, grid_shape)
            mat_sample_xfeat1 = torch.matmul(new_vom1.float(), bind_xfeat_mat1)

            t_single = time.time()

            # ── 5. 글로벌 객체 뱅크 갱신 ─────────────────────────────────
            if K_cur > 0:
                gids = obj_bank.register_or_update(
                    new_avg_vec1, fid.decode(), src)
                print(f"[ObjectBank] 크기={obj_bank.size}  "
                      f"gids={gids[:5]}{'...' if len(gids) > 5 else ''}")

            # ── 6. 메모리 저장 ────────────────────────────────────────────
            dino_mgr.process(src, fid, (
                x_cat1.cpu(),
                new_sample_1.cpu(),
                new_vom1.cpu(),
                new_avg_vec1.cpu(),
                bind_xfeat_mat1.cpu(),
            ))

            # ── 7. 다중 프레임 / 다중 기기 매칭 ──────────────────────────
            best_target = pick_best_target(list_neigh, src, fid,
                                           prefer_other_device=True)

            if best_target is not None and K_cur > 0:
                tsrc, tfid = best_target
                print(f"[타겟] {src.decode()}:{fid.decode()} ↔ "
                      f"{tsrc.decode()}:{tfid.decode()}")

                # 7-1. 메모리 앵커 로드
                x_cat2, sample2, mask2, avg_vec2, bind_xfeat2 = map(
                    lambda x: x.cuda(), dino_mgr.get(tsrc, tfid))
                feat2, _, _ = objpatcher._prepare_features(x_cat2)
                kp2, desc2  = xfeat_mgr.get(tsrc, tfid)
                kp2, desc2  = kp2.cuda(), desc2.cuda()

                # 7-2. 메모리 앵커 현재 프레임 투영 + 품질 필터
                valid_mask2, pure_area2, vom_mem_cur, _ = \
                    objpatcher.project_memory_to_frame(              # unified API
                        avg_vec2, feat1, min_pure_response=3)

                K_mem_before = avg_vec2.shape[0]
                K_mem_after  = valid_mask2.sum().item()
                print(f"[cross-frame] {K_mem_before} → {K_mem_after} "
                      f"(배제: {K_mem_before - K_mem_after})")

                avg_vec2_f = avg_vec2[valid_mask2]   # [K_mem', D]
                sample2_f  = sample2[valid_mask2]

                if avg_vec2_f.shape[0] > 0:
                    # 7-3. XFeat 귀속 (메모리 앵커, Multi-Response Masking 적용)
                    vom_f = vom_mem_cur[valid_mask2]                     # [K_mem', N]
                    clean_vom_f = vom_f & clean_patch_mask.unsqueeze(0)  # [K_mem', N]
                    mat_sample_xfeat2_f = torch.matmul(
                        clean_vom_f.float(), bind_xfeat2)

                    # 7-4. Cross-View Matching
                    smat = matcher.build_affinity_matrix(new_avg_vec1, avg_vec2_f)
                    fmat = matcher.build_affinity_matrix(desc1, desc2)
                    match_mask, match12 = \
                        matcher.compute_local_point_correspondence_batch_fixed(
                            mat_sample_xfeat1, mat_sample_xfeat2_f, smat, fmat)

                    print(f"[매칭] match_count={match_mask.sum().item()}")

                    # Phase 6 디버그
                    if vis_dbg is not None:
                        vis_dbg.phase6_cross_match(
                            img1, img_mgr.get(tsrc, tfid),
                            match_mask, match12, kp1, kp2)

                    # 7-5. 시각화
                    img2 = img_mgr.get(tsrc, tfid)
                    dynamic_path = (f"./matches/frame_{fid.decode()}"
                                    f"_{tfid.decode()}.png")
                    visualizer.visualize_global_scene_matching(
                        img1, img2, match_mask, match12,
                        mat_sample_xfeat1, mat_sample_xfeat2_f,
                        kp1, kp2, grid_shape, grid_shape, dynamic_path)

            t_total = time.time()
            print(f"[타임] dino={t_dino - stime:.3f}s  "
                  f"single={t_single - t_dino:.3f}s  "
                  f"total={t_total - stime:.3f}s")

            # ── 8. 단일 프레임 시각화 ─────────────────────────────────────
            if K_cur > 0:
                visualizer.visualize_exclusive_master_groups(
                    img1, new_sample_1, new_vom1, grid_shape)
                visualizer.visualize_mixed_boundary_patches(
                    img1, I_n, grid_shape, overlap_counts)

            cv2.waitKey(1)


# ─────────────────────────────────────────────────────────────────────────────
# ZMQ worker
# ─────────────────────────────────────────────────────────────────────────────

def run_worker():
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, SERVER_NAME)
    sock.connect("tcp://143.248.96.81:37001")

    notify_mgr = NotificationManager(RECV_KW)
    data_mgr   = DownloadDataManager(RECV_KW)

    sock.send_multipart([b"", b"RECV_REG"] + RECV_KW + [SERVER_NAME, b"ALL"])
    threading.Thread(target=inference_loop, args=(sock,), daemon=True).start()

    loop_count = 0
    while True:
        msg = sock.recv_multipart()
        if msg[1] == b"NOTIFY":
            _, _, kw, src, fid = msg
            targets = notify_mgr.register_notify(kw, src, fid)
            if targets:
                for p_src, p_kw in targets:
                    sock.send_multipart([b"", b"DOWNLOAD", p_kw, p_src, fid])

        elif msg[1] == b"DATA_REPLY":
            _, _, kw, src, fid, data_bytes = msg
            completed = data_mgr.register_data(kw, src, fid, data_bytes)
            if completed:
                task_queue.put((src, fid, completed))

        loop_count += 1
        if loop_count % 100 == 0:
            loop_count = 0
            notify_mgr.clear_old_fid(expire_time_sec=10)
            data_mgr.clear_old_data(expire_time_sec=15)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    return DinoV2Encoder()


if __name__ == '__main__':
    model      = load_model()
    objpatcher = DinoSemanticObjectExtractorV2()
    print("모델 로드 완료 (DinoSemanticObjectExtractorV2)")
    run_worker()

"""
zmq_dino_server_v3.py
────────────────────────────────────────────────────────────────
v2 대비 주요 변경:
  파이프라인 우선순위 변경
    1순위 (크로스 프레임): SALAD 타겟 + 메모리 앵커 존재 시
         → filter_memory_anchors_cross_frame → filter_by_quality
         → 유효 앵커 >= MIN_CROSS_ANCHORS 이면 크로스 프레임 결과 사용
    2순위 (단일 프레임 폴백): 최초 프레임 / SALAD 없음 / 유효 앵커 부족
         → sample_patch → merge_anchors_heatmap → _apply_group_and_recompute
    공통 후처리:
         → filter_by_quality (단일 프레임 결과에도 동일 기준 적용)
         → GlobalObjectBank 갱신 (품질 통과 앵커만)
         → dino_mgr 저장
         → XFeat 귀속 + Cross-View Matching
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
from dino.global_object_bank import GlobalObjectBank
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

# 크로스 프레임 경로를 유지하는 최소 유효 앵커 수
MIN_CROSS_ANCHORS = 3

# ── PHASE 2: SALAD 메모리 풀 구성 파라미터 (아키텍처 v2) ──────────────────────
POOL_MAX_FRAMES   = 5      # 풀에 포함할 최대 인접 프레임 수
POOL_DEDUP_SIM    = 0.90   # 객체 벡터 중복 제거 코사인 임계값
POOL_MAX_ANCHORS  = 64     # 풀에서 선택할 최대 객체 벡터 수

# ── 앵커 객체 표현 방식 (ablation 토글, objpatcher.compute_object_vectors) ────
#   ANCHOR_REPR  : "avg"(패치 평균) | "patch"(대표 시드 패치 피처)
#   AVG_WEIGHT   : "uniform"(단순평균) | "attn"(CLS attention 가중) | "sim"(유사도 가중)
#   OVERLAP_MODE : "exclude"(pure, 중복 배제) | "keep"(vom 전체) | "argmax"(중복은 최고 sim 앵커)
ANCHOR_REPR  = "patch"
AVG_WEIGHT   = "uniform"
OVERLAP_MODE = "exclude"

# ── 디버그 시각화 설정 ────────────────────────────────────────────────────────
DEBUG_VIS        = False          # True 로 바꾸면 각 Phase 시각화 저장
DEBUG_VIS_DIR    = './debug_v3'   # 저장 경로
DEBUG_VIS_PHASES = {3,4,5,6}           # None=전체  예: {1, 4, 5} 일부만 활성화

task_queue = Queue(maxsize=10)

# GlobalObjectBank 는 global_object_bank.py 에서 import

# 발산 entry 정리 주기 (프레임 단위)
CLEANUP_INTERVAL = 30

# ─────────────────────────────────────────────────────────────────────────────
# 타겟 프레임 선택
# ─────────────────────────────────────────────────────────────────────────────

def pick_best_target(list_neigh_frames, cur_src: bytes, cur_fid: bytes,
                     prefer_other_device: bool = True,
                     max_temporal_dist: int = 500):
    cur_fid_int = int(cur_fid.decode()) if cur_fid else 0
    other_device, same_device = [], []

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
        return other_device[0][:2]
    if same_device:
        same_device.sort(key=lambda x: -x[2])
        return same_device[0][:2]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 메모리 풀(build_object_vector_pool / select_object_vectors)과
# 객체 벡터 표현(compute_object_vectors)은 patchcluster_v2(objpatcher)로 이관됨.
# 서버는 데이터(dino_mgr)만 넘기고 호출한다.
# ─────────────────────────────────────────────────────────────────────────────


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
        objpatcher = DinoSemanticObjectExtractorV2(timing=True)
        objpatcher.set_stage_impl("filter_anchors", "test_filter")
        objpatcher.set_stage_impl("compute_anchor_response", "test_group")
        obj_bank   = GlobalObjectBank(
            reid_threshold=0.75,
            ema_alpha=0.3,
            stability_threshold=0.3,
            min_frames_to_judge=5,
            centroid_var_threshold=4.0,
            max_spatial_std=10.0,
        )
        visualizer    = DinoPatchVisualizer()
        vis_dbg       = PipelineDebugVisualizer(
            output_dir=DEBUG_VIS_DIR,
            enable_phases=DEBUG_VIS_PHASES,
        ) if DEBUG_VIS else None
        frame_counter = 0

        while True:
            bundle = task_queue.get()
            src, fid, data = bundle
            stime = time.time()

            frame_counter += 1
            if vis_dbg is not None:
                vis_dbg.set_frame_id(fid.decode())

            # ── 1. 데이터 수신 ────────────────────────────────────────────
            img_mgr.process(src, fid, data[b'image'][1])
            salad_mgr.process(src, fid, data[b'salad_res'][1])
            xfeat_mgr.process(src, fid, (data[b'xfeat_kp_res'][1],
                                          data[b'xfeat_desc_res'][1]))

            img1       = img_mgr.get(src, fid)
            img_rgb1   = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            kp1, desc1 = xfeat_mgr.get(src, fid)
            kp1, desc1 = kp1.cuda(), desc1.cuda()

            # ── 2. SALAD 인접 프레임 탐색 ─────────────────────────────────
            salad_mgr.calc_neighbor_frames(src, fid)
            list_neigh  = salad_mgr.get(src, fid)
            best_target = pick_best_target(list_neigh, src, fid,
                                           prefer_other_device=True)

            # Phase 2 디버그 — 인접 프레임 이미지 로드 (있는 것만)
            if vis_dbg is not None and list_neigh:
                # 인접 프레임만 (self 는 list_neigh 단계에서 이미 제외됨)
                _neigh_imgs, _neigh_sims, _labels = [], [], []
                for (_nkey, _nsim) in list_neigh[:5]:   # (key, sim)
                    _ns, _nf = _nkey
                    try:
                        _neigh_imgs.append(img_mgr.get(_ns, _nf))
                        _neigh_sims.append(_nsim)        # 실제 SALAD 코사인 sim
                        _labels.append(f"{_ns.decode()}:{_nf.decode()}")
                    except Exception:
                        break
                if _neigh_imgs:
                    _sel = [0] if (best_target is not None and list_neigh[0][0] == best_target) else []
                    vis_dbg.phase2_neighbor_frames(_neigh_imgs, _neigh_sims,
                                                   selected_idx=_sel, labels=_labels)

            # ── 3. DINOv2 피처 추출 (항상) ────────────────────────────────
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

            # ── 4. 앵커 확보: 크로스 프레임 우선 / 단일 프레임 폴백 ──────

            used_cross   = False   # 이번 프레임 경로 표시용
            work_sample  = None    # [K] 앵커 시드
            work_vom     = None    # [K, N] vom
            work_pure    = None    # [K, N] pure
            work_avg_vec = None    # [K, D]

            # ── 4-A. 크로스 프레임 경로 (PHASE 2 풀 기반) ────────────────
            target_has_memory = False
            avg_vec_mem  = None    # [M, D] 풀에서 선택된 메모리 객체 벡터
            work_mem_vec = None    # [K_cross, D] 매칭(smat)용 메모리 벡터

            if best_target is not None:
                # PHASE 2: 인접 프레임 풀 구성 → 중복 제거 → 대표 벡터 선택
                #   list_neigh 는 (key, sim) 튜플 → key 만 추출, best_target 우선 포함
                neigh_keys = [k for (k, _s) in list_neigh]
                pool_neigh = [best_target] + [k for k in neigh_keys if k != best_target]
                pool_vecs, _pool_meta = objpatcher.build_object_vector_pool(
                    dino_mgr, pool_neigh, repr=ANCHOR_REPR, max_frames=POOL_MAX_FRAMES)
                sel_vecs, _keep_idx   = objpatcher.select_object_vectors(
                    pool_vecs, sim_thresh=POOL_DEDUP_SIM, max_k=POOL_MAX_ANCHORS)
                if sel_vecs is not None and sel_vecs.shape[0] > 0:
                    avg_vec_mem       = sel_vecs.cuda()
                    target_has_memory = True

                # Phase 2 디버그 — 메모리 풀 + 선택 결과 (PCA 산점도)
                if vis_dbg is not None and pool_vecs is not None:
                    _selm = torch.zeros(pool_vecs.shape[0], dtype=torch.bool)
                    if _keep_idx is not None:
                        _selm[_keep_idx] = True
                    vis_dbg.phase2_memory_pool(pool_vecs, selected_mask=_selm)

            if target_has_memory:
                tsrc, tfid = best_target            # 매칭/시각화 기준 프레임 (단일 유지)

                # 매칭용 타겟 프레임 데이터 (XFeat 쪽)
                kp2, desc2 = xfeat_mgr.get(tsrc, tfid)
                kp2, desc2 = kp2.cuda(), desc2.cuda()
                _, _, _, _, bind_xfeat2 = map(
                    lambda x: x.cuda(), dino_mgr.get(tsrc, tfid))

                # 메모리 풀 앵커를 현재 프레임에 투영 — 단일/크로스 공용 컨텍스트.
                #   2D 쿼리(메모리 벡터) → self-mask 없음. vom/pure/heatmap_sim 모두 산출.
                ctx_mem  = objpatcher.compute_anchor_patch_context(avg_vec_mem, feat1)
                vom_cur  = ctx_mem["vom"]       # [M, N]
                pure_cur = ctx_mem["pure"]      # [M, N]
                # heatmap_sim = ctx_mem["heatmap_sim"]  # [M, M] 등록 판단/merge 에 활용 가능

                valid_mask = objpatcher.filter_anchors(              # = filter_by_quality
                    vom_cur, pure_cur,
                    min_pure_response=MIN_CROSS_ANCHORS,
                    min_pure_vom_ratio=0.10,
                    max_spatial_std=8.0,
                    grid_shape=grid_shape,
                )

                K_cross = valid_mask.sum().item()
                print(f"[크로스프레임] 풀 앵커 {avg_vec_mem.shape[0]} → 유효 {K_cross}")

                if vis_dbg is not None:
                    vis_dbg.phase3_cross_frame(img1, vom_cur, pure_cur,
                                               valid_mask, grid_shape)
                    vis_dbg.phase3_cross_context(img1, ctx_mem, grid_shape,
                                                 valid_mask=valid_mask)

                if K_cross >= MIN_CROSS_ANCHORS:
                    # 유효 앵커만 사용
                    vom_f  = vom_cur[valid_mask]    # [K', N]
                    pure_f = pure_cur[valid_mask]   # [K', N]

                    # 현재 프레임 pure 기반으로 avg_vec 재계산
                    avg_vec_cur, _ = objpatcher.recompute_anchor_vecs(   # unified API
                        x_cat1, pure_f.float(), attn_1)

                    # 풀에는 단일 시드가 없으므로 현재 프레임 투영 응답 피크를
                    # 대표 패치(위치 참조용)로 사용
                    work_sample  = pure_f.float().argmax(dim=1).long()   # [K']
                    work_vom     = vom_f
                    work_pure    = pure_f
                    work_avg_vec = avg_vec_cur
                    work_mem_vec = avg_vec_mem[valid_mask]   # smat용 메모리 벡터
                    used_cross   = True

            # ── 4-B. 단일 프레임 폴백 ────────────────────────────────────
            if not used_cross:
                print(f"[단일프레임] 폴백" + (" (최초)" if best_target is None else " (앵커 부족)"))

                mask1, sim_mat1 = objpatcher.generate_mask(
                    feat1, grid_shape, spatial_radius=3, sim_thresh=0.7)
                sample1 = objpatcher.generate_seeds(attn_1, mask1)          # unified API

                ctx1 = objpatcher.compute_anchor_patch_context(sample1, feat1)
                group_labels, n_comp = objpatcher.group_anchors(ctx1)       # unified API

                work_sample, work_vom, work_pure, work_avg_vec, _ = \
                    objpatcher.compute_anchor_response(                       # unified API
                        sample1, group_labels, n_comp, ctx1,
                        feat1, x_cat1, attn_1, grid_shape=grid_shape)

                print(f"[단일프레임] 앵커 {sample1.shape[0]} → {work_sample.shape[0]}")

                if vis_dbg is not None and work_sample.shape[0] > 0:
                    vis_dbg.phase3_anchors(img1, work_sample, work_vom,
                                           work_pure, grid_shape, suffix="single")

            # ── 5. 품질 필터 (공통 — 단일/크로스 모두 동일 기준) ─────────
            K_work = work_sample.shape[0] if work_sample is not None else 0

            if K_work > 0:
                keep = objpatcher.filter_anchors(                    # unified API
                    work_vom, work_pure,
                    min_pure_response=1,
                    min_pure_vom_ratio=0.10,
                    max_spatial_std=8.0,
                    grid_shape=grid_shape,
                )
                if vis_dbg is not None:
                    _qm = {"pure_area": work_pure.sum(dim=1).long().tolist()}
                    vis_dbg.phase4_quality_filter(img1, work_vom, work_pure,
                                                  keep, grid_shape,
                                                  quality_metrics=_qm)
                if keep.any() and not keep.all():
                    work_sample  = work_sample[keep]
                    work_vom     = work_vom[keep]
                    work_pure    = work_pure[keep]
                    work_avg_vec = work_avg_vec[keep]

                K_final = work_sample.shape[0]
                print(f"[품질필터] {K_work} → {K_final}  "
                      f"({'크로스' if used_cross else '단일'})")
            else:
                K_final = 0

            # ── 5-b. 앵커 객체 표현 계산 (표현/가중/중복 토글) ────────────
            #   work_avg_vec : 항상 avg 표현(메모리 저장 slot3 → 풀 재구성용)
            #   work_repr    : downstream(매칭·뱅크·multiresponse)에서 쓰는 표현
            if K_final > 0:
                work_avg_vec = objpatcher.compute_object_vectors(
                    feat1, work_sample, work_vom, attn=attn_1,
                    repr="avg", weight=AVG_WEIGHT, overlap=OVERLAP_MODE)
                if ANCHOR_REPR == "patch":
                    work_repr = F.normalize(feat1[work_sample], p=2, dim=1)
                else:
                    work_repr = work_avg_vec
            else:
                work_repr = work_avg_vec

            t_anchor = time.time()

            # ── 6. Multi-Response Masking ────────────────────────────────
            N = feat1.shape[0]
            if K_final > 0:
                sim_for_mask = torch.mm(work_repr, feat1.t())
                I_n, overlap_counts = objpatcher.detect_multiresponse(  # unified API
                    sim_for_mask, th_sim=0.60, th_margin=0.12)
                clean_patch_mask = (I_n < 1)
                if vis_dbg is not None:
                    vis_dbg.phase4_multiresponse(img1, I_n, grid_shape)
            else:
                I_n             = torch.zeros(N, dtype=torch.long, device=feat1.device)
                overlap_counts  = torch.zeros(N, device=feat1.device)
                clean_patch_mask = torch.ones(N, dtype=torch.bool, device=feat1.device)

            # ── 7. GlobalObjectBank 갱신 (품질 통과 앵커만) ───────────────
            if K_final > 0:
                pure_areas_f = work_pure.sum(dim=1).float()
                gids = obj_bank.register_or_update(
                    avg_vecs=work_repr,
                    pure_areas=pure_areas_f,
                    frame_id=fid.decode(),
                    vom=work_vom,
                    grid_shape=grid_shape,
                )
                print(f"[ObjectBank] size={obj_bank.size}  "
                      f"gids={gids[:5]}{'...' if len(gids) > 5 else ''}")

            # Phase 5 디버그
            if vis_dbg is not None and K_final > 0:
                vis_dbg.phase5_bank_state(obj_bank)

            # 주기적 발산 entry 정리
            if frame_counter % CLEANUP_INTERVAL == 0:
                removed = obj_bank.remove_diverging()
                if removed:
                    print(f"[ObjectBank] 발산 entry 삭제: gids={removed}")
                obj_bank.print_summary()

            # ── 8. 메모리 저장 ────────────────────────────────────────────
            bind_xfeat_mat1 = dino_mgr.bind_xfeat_to_patch(kp1, grid_shape)
            if K_final > 0:
                mat_sample_xfeat1 = torch.matmul(work_vom.float(), bind_xfeat_mat1)
            else:
                mat_sample_xfeat1 = torch.zeros(
                    (0, bind_xfeat_mat1.shape[1]), device=feat1.device)

            # Phase 4 디버그 — 앵커별 귀속 XFeat 키포인트
            if vis_dbg is not None and K_final > 0:
                vis_dbg.phase4_anchor_xfeat(img1, mat_sample_xfeat1, kp1, grid_shape)

            dino_mgr.process(src, fid, (
                x_cat1.cpu(),
                work_sample.cpu()  if K_final > 0 else torch.zeros(0, dtype=torch.long),
                work_vom.cpu()     if K_final > 0 else torch.zeros((0, N), dtype=torch.bool),
                work_avg_vec.cpu() if K_final > 0 else torch.zeros((0, 384)),  # slot3=avg(풀 재구성용)
                bind_xfeat_mat1.cpu(),
            ))

            # ── 9. Cross-View Matching ────────────────────────────────────
            if used_cross and K_final > 0:
                vom_match = work_vom & clean_patch_mask.unsqueeze(0)
                mat_xfeat2_f = torch.matmul(vom_match.float(), bind_xfeat2)

                smat = matcher.build_affinity_matrix(work_repr, work_mem_vec)
                fmat = matcher.build_affinity_matrix(desc1, desc2)
                match_mask, match12 = \
                    matcher.compute_local_point_correspondence_batch_fixed(
                        mat_sample_xfeat1, mat_xfeat2_f, smat, fmat)

                print(f"[매칭] match_count={match_mask.sum().item()}")

                # Phase 6 디버그
                if vis_dbg is not None:
                    vis_dbg.phase6_cross_match(
                        img1, img_mgr.get(tsrc, tfid),
                        match_mask, match12, kp1, kp2)

                img2 = img_mgr.get(tsrc, tfid)
                dynamic_path = f"./matches/frame_{fid.decode()}_{tfid.decode()}.png"
                visualizer.visualize_global_scene_matching(
                    img1, img2, match_mask, match12,
                    mat_sample_xfeat1, mat_xfeat2_f,
                    kp1, kp2, grid_shape, grid_shape, dynamic_path)

            t_total = time.time()
            print(f"[타임] dino={t_dino-stime:.3f}s  "
                  f"anchor={t_anchor-t_dino:.3f}s  "
                  f"total={t_total-stime:.3f}s  "
                  f"경로={'크로스' if used_cross else '단일'}")

            # ── 10. 시각화 ────────────────────────────────────────────────
            if K_final > 0:
                visualizer.visualize_exclusive_master_groups(
                    img1, work_sample, work_vom, grid_shape)
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
    print("모델 로드 완료 (DinoSemanticObjectExtractorV2 / v3 파이프라인)")
    run_worker()

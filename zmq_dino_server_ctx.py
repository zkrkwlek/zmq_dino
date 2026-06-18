"""
zmq_dino_server_ctx.py — STEP 2 참조 구현
────────────────────────────────────────────────────────────────
inference_loop 를 통합 data(frame/memory) + ctx 계약으로 재작성한 초안.

전제:
  - STEP 1 의 CtxPipelineMixin 을 DinoSemanticObjectExtractorV2 에 믹스인:
        class V2Ctx(CtxPipelineMixin, DinoSemanticObjectExtractorV2): pass
  - dino_mgr 저장 슬롯에 gids 추가 (x_cat, sample, vom, avg_vec, bind_xfeat, gids)
  - 매칭/뱅크/시각화 등 downstream 은 ctx 필드(vec/vom/centroid/meta)에서 꺼내 사용

※ 실행 검증 전 초안. 통합 시 ctx 필드/슬롯 정합만 확인하면 됨.
   특히 크로스 경로의 "현재 프레임 대표 패치(rep)" 처리(주석 표시) 확인.
"""
import threading, time
import cv2, numpy as np, torch
import torch.nn.functional as F
import zmq
from queue import Queue

from dino.dinoencoder import DinoV2Encoder
from dino.patchcluster_v2 import DinoSemanticObjectExtractorV2
from communication.commuprocessmanager import NotificationManager, DownloadDataManager
from communication.inferencedatamanager import (
    ImageDataManager, XFeatDataManager, SaladDataManager, DINODataManager)
from dino.dinosemanticmatcher import DinoXFeatPatchMatcher
from dino.visualizer import DinoPatchVisualizer
from dino.global_object_bank import GlobalObjectBank
from dino.pipeline_debug_visualizer import PipelineDebugVisualizer

# STEP 1 모듈
from ctx_stages import (CtxPipelineMixin, make_frame_data, is_memory, new_ctx)


class V2Ctx(CtxPipelineMixin, DinoSemanticObjectExtractorV2):
    """ctx 파이프라인 믹스인 + 기존 V2 알고리즘."""
    pass


SERVER_NAME = b"dino_server"
RECV_KW = [b"image", b"salad_res", b"xfeat_kp_res", b"xfeat_desc_res"]
MIN_CROSS_ANCHORS = 3
POOL_MAX_FRAMES, POOL_DEDUP_SIM, POOL_MAX_ANCHORS = 5, 0.90, 64

# 객체 표현 토글
ANCHOR_REPR, AVG_WEIGHT, OVERLAP_MODE = "avg", "uniform", "exclude"

DEBUG_VIS, DEBUG_VIS_DIR, DEBUG_VIS_PHASES = True, "./debug_ctx", None
CLEANUP_INTERVAL = 30
task_queue = Queue(maxsize=10)


def pick_best_target(list_neigh, cur_src, cur_fid,
                     prefer_other_device=True, max_temporal_dist=500):
    cur_fid_int = int(cur_fid.decode()) if cur_fid else 0
    other, same = [], []
    for (tkey, _sim) in list_neigh:            # list_neigh = (key, sim)
        tsrc, tfid = tkey
        try:    tfid_int = int(tfid.decode())
        except Exception: tfid_int = 0
        td = abs(cur_fid_int - tfid_int)
        if tsrc != cur_src:                other.append((tsrc, tfid, td))
        elif td < max_temporal_dist:       same.append((tsrc, tfid, td))
    if prefer_other_device and other: return other[0][:2]
    if same: same.sort(key=lambda x: -x[2]); return same[0][:2]
    return None


def inference_loop(zmq_socket):
    with torch.no_grad():
        img_mgr, xfeat_mgr = ImageDataManager(), XFeatDataManager()
        salad_mgr, dino_mgr = SaladDataManager(), DINODataManager()
        matcher = DinoXFeatPatchMatcher()
        obj_bank = GlobalObjectBank(reid_threshold=0.75, ema_alpha=0.3,
                                    stability_threshold=0.3, min_frames_to_judge=5,
                                    centroid_var_threshold=4.0, max_spatial_std=10.0)
        visualizer = DinoPatchVisualizer()
        vis_dbg = (PipelineDebugVisualizer(output_dir=DEBUG_VIS_DIR,
                                           enable_phases=DEBUG_VIS_PHASES)
                   if DEBUG_VIS else None)

        objp.set_stage_impl("filter_anchors", "test_filter")
        objp.set_stage_impl("compute_anchor_response", "test_group")

        frame_counter = 0

        while True:
            src, fid, data = task_queue.get()
            stime = time.time(); frame_counter += 1
            if vis_dbg is not None: vis_dbg.set_frame_id(fid.decode())

            # ── 1. 데이터 수신 ─────────────────────────────────────
            img_mgr.process(src, fid, data[b'image'][1])
            salad_mgr.process(src, fid, data[b'salad_res'][1])
            xfeat_mgr.process(src, fid, (data[b'xfeat_kp_res'][1],
                                          data[b'xfeat_desc_res'][1]))
            img1 = img_mgr.get(src, fid)
            kp1, desc1 = xfeat_mgr.get(src, fid)
            kp1, desc1 = kp1.cuda(), desc1.cuda()

            # ── 2. DINOv2 피처 → 현재 프레임 data1 ─────────────────
            tensor1, _ = model.preprocess_cv2(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            x_cat1, attn_1 = model.extract_features_with_attention(
                tensor1.cuda(), model.out_indices, patch_size=14, head_indices=[5])
            feat1, H_p, W_p = objp._prepare_features(x_cat1)
            grid_shape = (H_p, W_p)
            data1 = make_frame_data(x_cat1, feat1, attn_1, grid_shape)
            t_dino = time.time()

            # ── 3. SALAD 인접 → 메모리 data2 (type="memory", meta 포함) ──
            salad_mgr.calc_neighbor_frames(src, fid)
            list_neigh  = salad_mgr.get(src, fid)
            best_target = pick_best_target(list_neigh, src, fid)
            mem = None
            if best_target is not None:
                neigh_keys = [k for (k, _s) in list_neigh]
                pool_neigh = [best_target] + [k for k in neigh_keys if k != best_target]
                mem = objp.build_memory_data(dino_mgr, pool_neigh,
                                             repr=ANCHOR_REPR, max_frames=POOL_MAX_FRAMES)
                mem = objp.select_memory(mem, sim_thresh=POOL_DEDUP_SIM, max_k=POOL_MAX_ANCHORS)
                if vis_dbg is not None and mem is not None:
                    vis_dbg.phase2_memory_pool(mem["feat"])

            # ── 4. 앵커 추출: 크로스 우선 / 단일 폴백 → ctx ──────────
            used_cross = False
            ctx = None
            if mem is not None and mem["feat"].shape[0] > 0:
                mem_cuda = dict(mem); mem_cuda["feat"] = mem["feat"].cuda()
                ctx = objp.compute_context(data1, mem_cuda)            # 메모리→현재프레임 투영(디스패치)
                ctx = objp.filter_anchors(data1, ctx, reduce=False,
                                          min_pure_response=MIN_CROSS_ANCHORS,
                                          min_pure_vom_ratio=0.10, max_spatial_std=8.0)
                K_cross = int(ctx["keep"].sum())
                if vis_dbg is not None:
                    vis_dbg.phase3_cross_frame(img1, ctx["vom"], ctx["pure"],
                                               ctx["keep"], grid_shape)
                    vis_dbg.phase3_cross_context(img1,
                        {"sim_matrix": ctx["sim"], "vom": ctx["vom"], "pure": ctx["pure"],
                         "oc": ctx["oc"], "heatmap_sim": ctx["heat"]},
                        grid_shape, valid_mask=ctx["keep"])
                if K_cross >= MIN_CROSS_ANCHORS:
                    objp._reduce_ctx_inplace(ctx, ctx["keep"])         # 유효만 남김 (mem 행 idx 보존)
                    # 크로스: 현재 프레임 대표 패치(rep)=pure 응답 피크 → object_vectors 가 사용
                    ctx["rep"] = ctx["pure"].float().argmax(dim=1).long()
                    ctx = objp.object_vectors(data1, ctx, repr=ANCHOR_REPR,
                                              weight=AVG_WEIGHT, overlap=OVERLAP_MODE)
                    used_cross = True

            if not used_cross:
                ctx = objp.generate_seeds(data1)
                ctx = objp.compute_context(data1, data1, ctx)          # 단일: data2=data1
                ctx = objp.group_anchors(data1, ctx)
                ctx = objp.compute_anchor_response(data1, ctx, grid_shape=grid_shape)
                ctx = objp.object_vectors(data1, ctx, repr=ANCHOR_REPR,
                                          weight=AVG_WEIGHT, overlap=OVERLAP_MODE)

            # ── 5. 공통 품질 필터 (ctx 축소) ───────────────────────
            ctx = objp.filter_anchors(data1, ctx, reduce=True,
                                      min_pure_response=1, min_pure_vom_ratio=0.10,
                                      max_spatial_std=8.0)
            K_final = 0 if ctx["vec"] is None else ctx["vec"].shape[0]
            t_anchor = time.time()

            # ── 6. Multi-Response (재가공) → clean_patch_mask ──────
            N = feat1.shape[0]
            if K_final > 0:
                I_n, overlap, clean_patch_mask = objp.detect_multiresponse(
                    data1, ctx, th_sim=0.60, th_margin=0.12)
            else:
                I_n = torch.zeros(N, dtype=torch.long, device=feat1.device)
                overlap = torch.zeros(N, device=feat1.device)
                clean_patch_mask = torch.ones(N, dtype=torch.bool, device=feat1.device)

            # ── 7. GlobalObjectBank 갱신 → gids (메모리 저장에 사용) ─
            gids = []
            if K_final > 0:
                gids = obj_bank.register_or_update(
                    avg_vecs=ctx["vec"], pure_areas=ctx["pure"].sum(dim=1).float(),
                    frame_id=fid.decode(), vom=ctx["vom"], grid_shape=grid_shape)

            if frame_counter % CLEANUP_INTERVAL == 0:
                obj_bank.remove_diverging(); obj_bank.print_summary()

            # ── 8. 메모리 저장 (gids 슬롯 추가) ────────────────────
            bind_xfeat1 = dino_mgr.bind_xfeat_to_patch(kp1, grid_shape)
            if K_final > 0:
                mat_sample_xfeat1 = torch.matmul(ctx["vom"].float(), bind_xfeat1)
                gids_t = torch.tensor(gids, dtype=torch.long)
            else:
                mat_sample_xfeat1 = torch.zeros((0, bind_xfeat1.shape[1]), device=feat1.device)
                gids_t = torch.zeros(0, dtype=torch.long)
            dino_mgr.process(src, fid, (
                x_cat1.cpu(),
                (ctx["centroid"].cpu() if K_final > 0 else torch.zeros(0, dtype=torch.long)),
                (ctx["vom"].cpu()      if K_final > 0 else torch.zeros((0, N), dtype=torch.bool)),
                (ctx["vec"].cpu()      if K_final > 0 else torch.zeros((0, 384))),
                bind_xfeat1.cpu(),
                gids_t,                                       # ← 슬롯6: gid (메모리 meta 용)
            ))

            # ── 9. Cross-View Matching (best_target 프레임 기준) ───
            if used_cross and K_final > 0 and best_target is not None:
                tsrc, tfid = best_target
                kp2, desc2 = xfeat_mgr.get(tsrc, tfid); kp2, desc2 = kp2.cuda(), desc2.cuda()
                _, _, _, _, bind_xfeat2, *_ = dino_mgr.get(tsrc, tfid)
                bind_xfeat2 = bind_xfeat2.cuda()
                vom_match = ctx["vom"] & clean_patch_mask.unsqueeze(0)
                mat_xfeat2 = torch.matmul(vom_match.float(), bind_xfeat2)
                # 메모리쪽 벡터: 살아남은 ctx 의 출처(meta) 또는 mem 행으로 정렬
                mem_vec = mem["feat"].cuda()[ctx["centroid"]] if mem is not None else ctx["vec"]
                smat = matcher.build_affinity_matrix(ctx["vec"], mem_vec)
                fmat = matcher.build_affinity_matrix(desc1, desc2)
                match_mask, match12 = matcher.compute_local_point_correspondence_batch_fixed(
                    mat_sample_xfeat1, mat_xfeat2, smat, fmat)
                if vis_dbg is not None:
                    vis_dbg.phase6_cross_match(img1, img_mgr.get(tsrc, tfid),
                                               match_mask, match12, kp1, kp2)

            t_total = time.time()
            print(f"[타임] dino={t_dino-stime:.3f}s anchor={t_anchor-t_dino:.3f}s "
                  f"total={t_total-stime:.3f}s 경로={'크로스' if used_cross else '단일'} K={K_final}")
            cv2.waitKey(1)


def run_worker():
    ctx_zmq = zmq.Context()
    sock = ctx_zmq.socket(zmq.DEALER); sock.setsockopt(zmq.IDENTITY, SERVER_NAME)
    sock.connect("tcp://143.248.96.81:37001")
    notify_mgr, data_mgr = NotificationManager(RECV_KW), DownloadDataManager(RECV_KW)
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
            if completed: task_queue.put((src, fid, completed))
        loop_count += 1
        if loop_count % 100 == 0:
            loop_count = 0
            notify_mgr.clear_old_fid(10); data_mgr.clear_old_data(15)


if __name__ == '__main__':
    model = DinoV2Encoder()
    objp = V2Ctx(timing=True)
    objp.use_ctx_stages()          # 디스패치를 ctx 어댑터로 매핑 → set_stage_impl 로 교체 가능
    print("모델 로드 완료 (V2Ctx / ctx 파이프라인)")
    run_worker()

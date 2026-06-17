import torch
import zmq
import threading
import numpy as np
from queue import Queue
import time

from dino.dinoencoder import DinoV2Encoder
from dino.patchcluster import DinoSemanticObjectExtractor

from communication.commuprocessmanager import NotificationManager, DownloadDataManager
from communication.inferencedatamanager import ImageDataManager, XFeatDataManager, SaladDataManager, DINODataManager
from dino.dinosemanticmatcher import DinoXFeatPatchMatcher
from dino.visualizer import DinoPatchVisualizer, UnifiedAttentionDirectVectorVisualizer
from dino.crossattention import FrameMatchedCrossAttention

from dino.segmentation import generate_pseudo_labels_by_local_density, visualize_pseudo_labels_opencv

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

#model
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_PATH = '../dinov2'
DINO_MODEL = 'dinov2_vits14'
WEIGHTS_PATH = './dinov2_vits14.pth'

SERVER_NAME = b"dino_server"
RECV_KW = [b"image",b"salad_res", b"xfeat_kp_res", b"xfeat_desc_res"] #[b"salad_res", b"xfeat_kp_res"]


#load model
def load_model():
    encoder = DinoV2Encoder()
    ##처음 GPU 돌리기
    img1 = cv2.imread('./7.png')
    img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    tensor1, (W, H) = encoder.preprocess_cv2(img_rgb1)
    x_cat1, attn_1 = encoder.extract_features_with_attention(tensor1.cuda(), encoder.out_indices, patch_size=14, head_indices=[5])
    return encoder

# 가상의 모델 (예: Segmentation)
task_queue = Queue(maxsize=10)

import cv2

def inference_loop(zmq_socket):
    with torch.no_grad():
        img_mgr = ImageDataManager()
        xfeat_mgr = XFeatDataManager()
        salad_mgr = SaladDataManager()
        dino_mgr = DINODataManager()
        matcher = DinoXFeatPatchMatcher()
        ca_mgr = FrameMatchedCrossAttention().cuda()

        while True:
            # 큐에서 작업 꺼내기
            bundle = task_queue.get()
            src, fid, data = bundle
            #print(data)

            stime = time.time()
            img_mgr.process(src, fid, data[b'image'][1])
            salad_mgr.process(src, fid, data[b'salad_res'][1])
            xfeat_mgr.process(src, fid, (data[b'xfeat_kp_res'][1], data[b'xfeat_desc_res'][1]))

            img1 = img_mgr.get(src, fid)
            img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            ##SALAD && Xfeat
            salad_mgr.calc_neighbor_frames(src, fid)
            list_neigh_frames = salad_mgr.get(src, fid)
            kp1, desc1 = xfeat_mgr.get(src, fid)
            kp1 = kp1.cuda()
            desc1 = desc1.cuda()

            best_target = None
            max_temporal_dist = -1
            current_frame_idx = int(fid.decode())  # 현재 프레임 ID 정수화

            for (tsrc, tfid) in list_neigh_frames:
                target_frame_idx = int(tfid.decode())
                temporal_dist = abs(current_frame_idx - target_frame_idx)

                # 룩업 테이블이나 매니저를 통해 미리 계산된 SALAD 점수를 가져온다고 가정
                # 만약 list_neigh_frames가 이미 SALAD 유사도 탑-N으로 정렬되어 들어온다면,
                # 이 중에서 시간축 거리가 가장 먼 녀석을 고르는 것이 최선입니다.
                if temporal_dist > max_temporal_dist and temporal_dist < 100:
                    max_temporal_dist = temporal_dist
                    best_target = (tsrc, tfid)

            t1 = time.time()
            ##DINOv2
            tensor1, (W, H) = model.preprocess_cv2(img_rgb1)
            x_cat1,attn_1 = model.extract_features_with_attention(tensor1.cuda(), model.out_indices, patch_size=14, head_indices=[5])
            feat1, H_p, W_p = objpatcher._prepare_features(x_cat1)
            t2 = time.time()

            ##cross attention
            #memory bank
            TOP_K = 1
            selected_frames = list_neigh_frames[:TOP_K]
            Nm = H_p*W_p*len(selected_frames)
            Np = H_p*W_p
            memory_bank_patches = torch.empty((Nm, 384), dtype=torch.float32, device=feat1.device)
            global_target_indices = []
            start_idx = 0
            for i, (tsrc, tfid) in enumerate(selected_frames):
                # 1) 현재 프레임의 DINO 패치 추출 (가상 텐서)
                x_cat2, sample2, mask2, avg_patch_vec2, bind_xfeat_mat2 = map(
                    lambda x: x.cuda(), dino_mgr.get(tsrc, tfid)
                )
                feat2,_,_ = objpatcher._prepare_features(x_cat2)
                M_new = avg_patch_vec2.shape[0]
                end_idx = start_idx + M_new
                memory_bank_patches[start_idx:end_idx, :] = avg_patch_vec2

                if M_new > 0:
                    adjusted_tensor = torch.arange(start_idx, start_idx+3, dtype=torch.long, device=avg_patch_vec2.device)
                    global_target_indices.append(adjusted_tensor)

                start_idx = end_idx
                """
                # 2) 텐서의 어느 위치에 넣을지 시작(start)과 끝(end) 인덱스 계산
                start_idx = i * Np  # 0, 1024, 2048
                end_idx = (i + 1) * Np  # 1024, 2048, 3072

                # 3) 만들어둔 빈 텐서의 해당 구간에 추출한 패치 덮어쓰기 (In-place 연산)
                memory_bank_patches[start_idx:end_idx, :] = feat2

                if sample2 is not None and len(sample2) > 0:
                    adjusted_tensor = sample2 + start_idx
                    global_target_indices.append(adjusted_tensor)
                """
                print(f"Frame {i}, {tfid.decode()} (인덱스 {start_idx}~{end_idx}) 채워넣기 완료")

            if len(list_neigh_frames) > 0:
                if global_target_indices:
                    global_target_indices = torch.cat(global_target_indices, dim=0).to(feat1.device)
                else:
                    global_target_indices = None
                afeat1, aattn_1 = ca_mgr(feat1, avg_patch_vec2, None)
                print("attn shape",aattn_1.shape, attn_1.shape)
                visualizer.visualize_cls_attention_opencv(img1, aattn_1, grid_shape)

            grid_shape = (H_p, W_p)
            mask1, sim_mat1 = objpatcher.generate_mask(feat1, grid_shape, spatial_radius=3, sim_thresh=0.7)
            sample1 = objpatcher.sample_patch(attn_1, mask1)

            affinity1 = objpatcher.build_anchor_to_patch_affinity(sample1, mask1, sim_mat1, exclusive=True)
            avg_patch_vec1, _ = objpatcher.extract_sample_neighborhood_average_pool(x_cat1, affinity1, attn_1)
            # patch cluster
            group1, n_comp1, heat1 = objpatcher.compile_structural_equivalence_vectorized(feat1, feat1[sample1])

            new_sample_1 = objpatcher.extract_new_group_centroids_vectorized(
                sample1_old=sample1, group_assignments=group1, n_components=n_comp1
            )
            #affinity11 = matcher.build_affinity_matrix(avg_patch_vec1, feat1)
            new_affinity1 = objpatcher.expand_patch_groups_exclusive_vectorized(
                feat_base=feat1,
                sample1_new=new_sample_1,
                sim_thresh=0.70  # 박사님의 순정 역치 유지
            )
            new_avg_patch_vec1, _ = objpatcher.extract_sample_neighborhood_average_pool(x_cat1, new_affinity1, attn_1)

            sim1 = torch.matmul(new_avg_patch_vec1, feat1.t())
            #사용 안할 듯
            ###공유 패치 기반 단일 프레임 관리
            I1, overlap1 = objpatcher.detect_mixed_boundary_patches_by_counting(sim1)

            ttt1 = time.time()

            ctx = objpatcher.compute_anchor_patch_context(
                sample1, feat1,
                th_sim=0.60, th_margin=0.12, sim_cutoff=0.60,
            )
            overlap_results = objpatcher.classify_anchor_overlaps_vectorized(ctx['vom'])
            # ── 방식 A: 히트맵 유사도 기반 병합 ─────────────────────────

            """
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
            """
            ttt2 = time.time()
            print("valid test = ",id, ttt2-ttt1)
            ### 공유 패치 시각화
            #visualizer.visualize_anchor_overlaps_interactive(img1, sample1, ctx['vom'], overlap_results, grid_shape)
            visualizer.save_anchor_overlaps_batch(
                base_output_dir="./output_overlaps",  # 🎯 이 폴더 안에 알아서 frame_xxx 폴더가 생성됩니다.
                frame_id=fid,  # bytes 혹은 str 원본 주입
                img_bgr=img1,
                centroids=ctx['centroids'],
                valid_overlap_mask=ctx['vom'],
                overlap_results=overlap_results,
                grid_shape=grid_shape
            )
            visualizer.visualize_pure_vs_overlap_patches(
                img1, ctx['vom'], ctx['pure'], grid_shape)
            """
            visualizer.visualize_anchor_refinement_comparison(
                img_bgr=img1,
                centroids=ctx['centroids'],
                valid_overlap_mask=ctx['vom'],  # 정제 전 (VOM)
                refined_masks=result_a['group_pure'][result_a['group_assign']],  # 💡 명칭 교정 완료
                valid_after=result_a['valid_groups'][result_a['group_assign']],
                grid_shape=grid_shape
            )
            """
            ###공유 패치 기반 단일 프레임 관리

            link1 = objpatcher.build_anchor_to_anchor_link(new_affinity1)
            bind_xfeat_mat1 = dino_mgr.bind_xfeat_to_patch(kp1, grid_shape)
            mat_sample_xfeat1 = torch.matmul(new_affinity1.float(), bind_xfeat_mat1)

            dino_mgr.process(src, fid, (x_cat1.cpu(), new_sample_1.cpu(), mask1.cpu(), new_avg_patch_vec1.cpu(), bind_xfeat_mat1.cpu()))
            #patch_mask1 = objpatcher.build_patch_group_mask(feat1, sample1)
            #avg_vec1, n_patches1 = objpatcher.extract_sample_neighborhood_average_pool(x_cat1, sample1)
            #dino_mgr.process(src, fid, x_cat1.cpu(), sample1.cpu(), patch_mask1.cpu())
            t3 = time.time()

            #labels, spatial_spread = generate_pseudo_labels_by_local_density(feat1, sample1)
            #visualize_pseudo_labels_opencv(img, sample1, labels, spatial_spread, grid_shape1)
            #cv2.waitKey()

            t4 = time.time()

            #visualizer.visualize_cls_attention_opencv(img1, attn_1, grid_shape)

            #visualizer.visualize_exclusive_master_groups(img1, new_sample_1, new_affinity1, grid_shape)
            #visualizer.visualize_mixed_boundary_patches(img1, I1, grid_shape, overlap1)

            #visualizer.visualize_anchor_relations(img1, new_sample_1, mask1, link1, grid_shape)
            #visualizer.visualize_sample_to_sample_similarity(img1, new_sample_1, new_avg_patch_vec1, new_affinity1, grid_shape)
            #visualizer.visualize_new_structural_grouping(img1, new_sample_1, new_affinity1, group1, heat1, n_comp1, grid_shape)
            #cv2.waitKey(0)
            #continue

            ##객체 벡터 매핑
            """
            for (tsrc, tfid) in list_neigh_frames:
                #print(tsrc, tfid)
                img2 = img_mgr.get(tsrc, tfid)
                kp2, desc2 = xfeat_mgr.get(tsrc, tfid)
                kp2 = kp2.cuda()
                desc2 = desc2.cuda()
                #idx1,idx2 = xfeat_mgr.match_xfeat_desc(desc1, desc2)
                x_cat2, sample2, mask2, avg_patch_vec2, bind_xfeat_mat2 = map(
                    lambda x: x.cuda(), dino_mgr.get(tsrc, tfid)
                )

                affinity2 = objpatcher.build_anchor_to_patch_affinity(sample2, mask2)
                mat_sample_xfeat2 = torch.matmul(affinity2.float(), bind_xfeat_mat2)
                smat = matcher.build_affinity_matrix(avg_patch_vec1, avg_patch_vec2)
                fmat = matcher.build_affinity_matrix(desc1,desc2)
                match_mask, match12 = matcher.compute_local_point_correspondence_batch(mat_sample_xfeat1, mat_sample_xfeat2, smat, fmat)
                dynamic_path = "./matches/frame_" + fid.decode() + ".png"
                visualizer.visualize_global_scene_matching(img1, img2, match_mask, match12, mat_sample_xfeat1, mat_sample_xfeat2, kp1, kp2, grid_shape, grid_shape, dynamic_path)
                #objpatcher.visualize_patch_group_connections(img1, sample1, img2, sample2, group_affinity_counts, best_match_g2_idx, valid_group_mask, grid_shape1)
                #print(kp2, desc2)
                break
            """
            """"""

            if False and best_target is not None:
                tsrc, tfid = best_target
                print(f"🎯 [최적 대조 씬 선별] 현재: {current_frame_idx} ↔ 타겟: {tfid.decode()} (프레임 간격: {max_temporal_dist})")

                img2 = img_mgr.get(tsrc, tfid)
                kp2, desc2 = xfeat_mgr.get(tsrc, tfid)
                kp2 = kp2.cuda()
                desc2 = desc2.cuda()

                # A. 박사님의 공간-시맨틱 배치 매칭 파이프라인 가동
                x_cat2, sample2, mask2, avg_patch_vec2, bind_xfeat_mat2 = map(
                    lambda x: x.cuda(), dino_mgr.get(tsrc, tfid)
                )
                feat2, _, _ = objpatcher._prepare_features(x_cat2)
                affinity2 = objpatcher.build_anchor_to_patch_affinity(sample2, mask2)
                mat_sample_xfeat2 = torch.matmul(affinity2.float(), bind_xfeat_mat2)

                smat = matcher.build_affinity_matrix(new_avg_patch_vec1, avg_patch_vec2)
                fmat = matcher.build_affinity_matrix(desc1, desc2)

                match_mask, match12 = matcher.compute_local_point_correspondence_batch_fixed(
                    mat_sample_xfeat1, mat_sample_xfeat2, smat, fmat
                )

                # B. 주석 해제된 오리지널 XFeat 순정 매칭 결과 추출 (박사님 원본 match_xfeat_desc 활용)
                idx1, idx2 = xfeat_mgr.match_xfeat_desc(desc1, desc2)

                #어텐션 비교
                unified_engine = UnifiedAttentionDirectVectorVisualizer(grid_shape=(34, 45))
                unified_engine.start_unified_direct_vector_viewer(
                    img1,  # Frame 1 이미지 변수
                    img2,  # Frame 2 이미지 변수
                    feat1,
                    feat2, new_avg_patch_vec1,
                    new_affinity1  # 기준 프레임 고유의 마스크 행렬
                    #,sample1, group1,n_comp1
                )
                # C. 💡 [듀얼 시각화 가동]: 두 매칭 방식을 비교하여 "./matches/frame_XXXX.png" 로 저장
                dynamic_path = "./matches5/frame_" + fid.decode()+"_"+tfid.decode() + ".png"
                visualizer.visualize_comparison_triple_canvas(
                    img1_rgb=img1,
                    img2_rgb=img2,
                    final_point_pairs_mask=match_mask,
                    match12_table=match12,
                    xfeat_idx1=idx1,
                    xfeat_idx2=idx2,
                    xfeat_kpts1=kp1,
                    xfeat_kpts2=kp2,
                    grid_shape=grid_shape,
                    smat = smat,
                    A_k1_f1= mat_sample_xfeat1,
                    A_k2_f2= mat_sample_xfeat2,
                    save_path=dynamic_path
                )

            etime = time.time()
            cv2.waitKey(1)
            print("처리", fid, etime-stime, len(list_neigh_frames), " salad 처리 = ", t1-stime, ", 디노 인코딩 = ", t2-t1, ", 디노 패치 벡터 = ", t3-t2, etime-t4)


def run_worker():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, SERVER_NAME)
    sock.connect("tcp://143.248.96.81:37001")

    notify_mgr  = NotificationManager(RECV_KW)
    data_mgr = DownloadDataManager(RECV_KW)

    # 1. 'image' 키워드 수신 등록
    sock.send_multipart([b"", b"RECV_REG"]+RECV_KW+[SERVER_NAME, b"ALL"])

    # 2. 추론 스레드 시작
    threading.Thread(target=inference_loop, args=(sock,), daemon=True).start()

    loop_count = 0
    while True:
        msg = sock.recv_multipart()
        # NOTIFY 받으면 즉시 DOWNLOAD 요청
        if msg[1] == b"NOTIFY":
            # 알림 패킷: [ID(zmq 제거), Empty, NOTIFY(노티), KW, Source, Target, FID]
            _, _, kw, src, fid = msg

            targets = notify_mgr.register_notify(kw, src, fid)
            if targets:
                for p_src, p_kw in targets:
                    sock.send_multipart([b"", b"DOWNLOAD", p_kw, p_src, fid])

        # 실제 데이터 수신 시 큐에 삽입
        elif msg[1] == b"DATA_REPLY":
            # 응답 패킷 구성: [요청자ID(제거), Empty, DATA_REPLY, KW, Source, Target, FID, Data]
            _, _, kw, src, fid, data = msg

            completed_bundle = data_mgr.register_data(kw, src, fid, data)
            if completed_bundle:
                task_queue.put((src, fid, completed_bundle))
            #task_queue.put((None, kw, src, fid, data))

        loop_count += 1
        if loop_count % 100 == 0:
            loop_count = 0
            notify_mgr.clear_old_fid(expire_time_sec=10)
            data_mgr.clear_old_data(expire_time_sec=15)

if __name__ == '__main__':
    model = load_model()
    print("Load Model 완료")
    objpatcher = DinoSemanticObjectExtractor()
    visualizer = DinoPatchVisualizer()
    run_worker()
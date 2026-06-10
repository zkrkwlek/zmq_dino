import h5py
import cv2
import numpy as np
import torch
from pathlib import Path

from dino.dinoencoder import DinoV2Encoder
from dino.patchcluster import DinoSemanticObjectExtractor
from dino.visualizer import DinoPatchVisualizer
from dino.dinosemanticmatcher import DinoXFeatPatchMatcher
from communication.inferencedatamanager import ImageDataManager, XFeatDataManager, SaladDataManager, DINODataManager

import sys
import os
sys.path.append(os.path.abspath('D:\\UVR\\accelerated_features'))
from modules.xfeat import XFeat

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_PATH = '../../../dinov2'
DINO_MODEL = 'dinov2_vits14'
WEIGHTS_PATH = '../../../ZeroMQ_Server/dinov2_vits14.pth'

class ScannetExperimentVisualizer:
    def __init__(self):
        pass
class ScannetMultiViewExperimentEngine:
    def __init__(self, h5_path, scans_root):
        self.h5_path = h5_path
        self.scans_root = Path(scans_root)
        self.visualizer = DinoPatchVisualizer()
        self.dino_mgr = DINODataManager()
        self.xfeat_mgr = XFeatDataManager()

        top_k = 2048
        self.xfeat = XFeat(weights='../../../accelerated_features/weights/xfeat.pt', top_k=top_k,
                      detection_threshold=0.05).eval().cuda()
        self.matcher = DinoXFeatPatchMatcher()
        self.encoder = DinoV2Encoder(DINO_PATH = DINO_PATH, WEIGHTS_PATH= WEIGHTS_PATH)
        self.sampler = DinoSemanticObjectExtractor(patch_size=14)
        print("Load Model")

    def get_available_scenes(self):
        """H5 파일에 전처리되어 저장된 선점 씬 ID 리스트 반환"""
        with h5py.File(self.h5_path, 'r') as f:
            return list(f.keys())

    def _load_pure_image(self, scene_id, fid_str):
        """디스크에서 순정 이미지를 로드하여 PyTorch 텐서로 변환"""
        img_path = self.scans_root / scene_id / "color" / f"frame-{fid_str.zfill(6)}.color.640.png"
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        # BGR -> RGB 및 전산학적 텐서화 [3, H, W]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor1, (W, H) = self.encoder.preprocess_cv2(img)
        #img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, tensor1

    def _preprocessing(self, img, tensor, patch_size = 14, spatial_radius = 3, sim_thresh = 0.7):
        pred1 = self.xfeat.detectAndCompute(img)[0]
        kp1 = pred1['keypoints']
        desc1 = pred1['descriptors']
        x_cat1, attn_1 = self.encoder.extract_features_with_attention(tensor.cuda(), self.encoder.out_indices,
                                                                      patch_size=patch_size,
                                                                      head_indices=[5])  # 0,1,2,3,4,
        feat_1, Hp, Wp = self.sampler._prepare_features(x_cat1)
        grid_shape1 = (Hp, Wp)
        mask1 = self.sampler.generate_mask(feat_1, grid_shape1, spatial_radius=spatial_radius, sim_thresh=sim_thresh)

        centroid1 = self.sampler.sample_patch(attn_1, mask1)
        affinity1 = self.sampler.build_anchor_to_patch_affinity(centroid1, mask1)
        #avg_patch_vec1, _ = self.sampler.extract_sample_neighborhood_average_pool(x_cat1, affinity1)
        avg_patch_vec1, _ = self.sampler.extract_sample_neighborhood_pure_average(feat_1, affinity1)
#
        # 오직 패치 유사도와 시드 평균 벡터로만
        affinity11 = self.matcher.build_affinity_matrix(avg_patch_vec1, feat_1)
        #mask11 = self.sampler.generate_mask(feat_1, grid_shape1, spatial_radius=0, sim_thresh=sim_thresh)
        #affinity11 = self.sampler.build_anchor_to_patch_affinity(centroid1, mask1)

        link1 = self.sampler.build_anchor_to_anchor_link(affinity11)
        bind_xfeat_mat1 = self.dino_mgr.bind_xfeat_to_patch(kp1, grid_shape1)
        mat_sample_xfeat1 = torch.matmul(affinity11.float(), bind_xfeat_mat1)

        return {"kp":kp1,
                "desc":desc1,
                "dino_embedding":x_cat1,
                "dino_feat":feat_1,
                "grid_shape":grid_shape1,
                "dino_mask":mask1,
                "attention":attn_1,
                "dino_seed":centroid1,
                "dino_affinity":affinity11,
                "dino_avg_vec":avg_patch_vec1,
                "bind_patch_xfeat":bind_xfeat_mat1,
                "bind_seed_xfeat":mat_sample_xfeat1,
                "link_patch":link1}

    def run_scene_multi_view_experiment(self, scene_id, alpha=0.20, tau=0.75):
        """
        💡 [박사님 핵심 요구사항]: 선택된 씬 내부의 프레임들을 돌면서(for image)
           다른 뷰의 인접 이미지들과 전역 위상 연산 및 예외 케이스 방어력 검증 실행
        """
        print(f"\n[실험 엔진 가동] Target Scene ID: {scene_id}")

        with h5py.File(self.h5_path, 'r') as f:
            if scene_id not in f:
                print(f"❌ H5 파일 내에 {scene_id} 데이터가 없습니다.")
                return

            scene_grp = f[scene_id]
            base_fids = list(scene_grp["matches"].keys())

        print(f"  └── 발견된 총 기준 프레임 수: {len(base_fids)}개. 내역 순회(for image)를 시작합니다.")

        # 🔄 [for image]: 기준 프레임 이미지들을 순차적으로 타겟팅
        for base_fid in base_fids:
            # 1. 현재 기준 뷰 이미지 로드
            base_img ,base_img_tensor = self._load_pure_image(scene_id, base_fid)
            if base_img_tensor is None:
                continue

            with torch.no_grad():
                base_data = self._preprocessing(base_img, base_img_tensor)
                self.visualizer.visualize_cls_attention_opencv(base_img, base_data["attention"], base_data["grid_shape"])
                self.visualizer.visualize_anchor_relations(base_img, base_data["dino_seed"], base_data["dino_mask"], base_data["link_patch"], base_data["grid_shape"])
                self.visualizer.visualize_sample_to_sample_similarity(base_img, base_data["dino_seed"], base_data["dino_avg_vec"], base_data["dino_affinity"], base_data["grid_shape"])


            # 2. H5 토폴로지 저장소로부터 '다른 뷰의 인접 프레임 ID' 목록 룩업
            with h5py.File(self.h5_path, 'r') as f:
                neighbor_fids = list(f[scene_id]["matches"][base_fid].keys())

            if not neighbor_fids:
                continue

            print(f"\n[Current Frame: {base_fid}] ──> 다른 뷰({len(neighbor_fids)}개)와의 전역 유사도 교차 연산 전개")

            # 🔄 [for neighbor_image]: 다른 뷰의 인접 프레임들을 하나씩 꺼내어 상호 대조 비교
            for neigh_fid in reversed(neighbor_fids):
                neigh_img, neigh_img_tensor = self._load_pure_image(scene_id, neigh_fid)
                if neigh_img_tensor is None:
                    continue

                with torch.no_grad():
                    neigh_data = self._preprocessing(neigh_img, neigh_img_tensor)

                    smat = self.matcher.build_affinity_matrix(base_data["dino_avg_vec"], neigh_data["dino_avg_vec"])
                    fmat = self.matcher.build_affinity_matrix(base_data["desc"], neigh_data["desc"])

                    match_mask, match12 = self.matcher.compute_local_point_correspondence_batch_fixed(
                        base_data["bind_seed_xfeat"], neigh_data["bind_seed_xfeat"], smat, fmat
                    )

                    # B. 주석 해제된 오리지널 XFeat 순정 매칭 결과 추출 (박사님 원본 match_xfeat_desc 활용)
                    idx1, idx2 = self.xfeat_mgr.match_xfeat_desc(base_data["desc"], neigh_data["desc"])

                    # C. 💡 [듀얼 시각화 가동]: 두 매칭 방식을 비교하여 "./matches/frame_XXXX.png" 로 저장
                    dynamic_path = "./matches3/frame_" + base_fid + "_" + neigh_fid + ".png"
                    self.visualizer.visualize_comparison_triple_canvas(
                        img1_rgb=base_img,
                        img2_rgb=neigh_img,
                        final_point_pairs_mask=match_mask,
                        match12_table=match12,
                        xfeat_idx1=idx1,
                        xfeat_idx2=idx2,
                        xfeat_kpts1=base_data["kp"],
                        xfeat_kpts2=neigh_data["kp"],
                        grid_shape=base_data["grid_shape"],
                        smat=smat,
                        A_k1_f1=base_data["bind_seed_xfeat"],
                        A_k2_f2=neigh_data["bind_seed_xfeat"],
                        save_path=dynamic_path
                    )
                break
            cv2.waitKey(0)
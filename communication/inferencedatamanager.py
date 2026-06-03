import torch
import numpy as np
import cv2

class InferenceDataManager:
    def __init__(self):
        self.storage = {}
        self.count = 0
    def process(self, src, id, data):
        pass

    def get(self, src, id):
        return self.storage[src][id]

class ImageDataManager(InferenceDataManager):
    def __init__(self):
        super().__init__()
        self.img_size = {}

    def process(self, src, id, data):
        # numpy 형태로 저장
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if src not in self.storage:
            self.storage[src] = {}
            self.img_size[src] = img.shape[:2]
        self.storage[src][id] = img

class XFeatDataManager(InferenceDataManager):
    def __init__(self):
        super().__init__()
    
    #CPU에 보관
    def process(self, src, fid, data):
        kp_data, desc_data = data
        nparr = np.frombuffer(kp_data, np.float32)
        kp = torch.from_numpy(nparr)#.to('cuda')
        kp = kp.reshape(-1, 2)

        nparr = np.frombuffer(desc_data, np.float32)
        desc_tensor = torch.from_numpy(nparr)#.to('cuda')
        desc = desc_tensor.reshape(-1, 64)

        if src not in self.storage:
            #self.storage[(src, fid)] = [None, None]
            self.storage[src] = {}
        self.storage[src][fid]=(kp.cpu(),desc.cpu())
        #self.storage[src][fid][0] = kp  # 0번 인덱스에 포인트 저장
        #self.storage[src][fid][1] = desc  # 1번 인덱스에 기술자 저장

    @torch.inference_mode()
    def match_xfeat_desc(self, feats1, feats2, min_cossim=0.9):  # 0.82

        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    @torch.inference_mode()
    def match_xfeat_desc_batched(self, feats_curr, list_of_feats_ref, min_cossim=0.9):
        """
        feats_curr       : (N, D) GPU
        list_of_feats_ref: list of (Mi, D) GPU
        반환             : list of (idx_curr, idx_ref) — 각 ref에 대한 매칭 인덱스 쌍
        """
        if not list_of_feats_ref:
            return []

        # 1. 오프셋 계산 (CPU, 정수 연산)
        lens = [f.shape[0] for f in list_of_feats_ref]
        offsets = [0]
        for l in lens:
            offsets.append(offsets[-1] + l)

        # 2. 행렬 곱 한 번 (N × Total_M) — 핵심 비용
        feats_ref_all = torch.cat(list_of_feats_ref, dim=0)  # (Total_M, D)
        cossim_all = feats_curr @ feats_ref_all.t()  # (N, Total_M)

        idx_curr = torch.arange(feats_curr.shape[0], device=feats_curr.device)
        results = []

        # 3. ref별 슬라이싱 후 MNN (슬라이스는 view — 추가 복사 없음)
        for i in range(len(list_of_feats_ref)):
            s, e = offsets[i], offsets[i + 1]
            cossim_i = cossim_all[:, s:e]  # (N, Mi) — no copy

            val12, match12 = cossim_i.max(dim=1)  # curr → ref_i
            _, match21 = cossim_i.max(dim=0)  # ref_i → curr  ← 슬라이스 기준

            mutual = match21[match12] == idx_curr
            good = val12 > min_cossim

            mask = mutual & good
            results.append((idx_curr[mask], match12[mask]))

        return results

class SaladDataManager(InferenceDataManager):
    def __init__(self):
        super().__init__()
        self.salad_mat = torch.empty((0, 8448), device='cpu')
        self.metadata = {}
        self.src_to_idx = {}

    def process(self, src, id, data):
        nparr = np.frombuffer(data, np.float32)
        desc = torch.from_numpy(nparr).to('cpu')
        desc = desc.view(1, 8448)

        key = (src, id)
        if key not in self.src_to_idx:
            idx = self.count
            self.salad_mat = torch.cat((self.salad_mat, desc), dim=0)
            self.metadata[idx] = (src, id)
            self.src_to_idx[key] = idx
            self.count = self.count + 1
            return idx
        else:
            return self.src_to_idx[key]

    @torch.inference_mode()
    def match_topk(self, feat1, feats2, K=5, th=0.6):
        # feat1: 1 x N, feats2: M x N
        cossim = (feats2 @ feat1.t()).squeeze()  # (M,)
        # threshold 조건을 먼저 적용하여 mask 추출
        mask = cossim >= th
        # 조건에 맞는 인덱스와 값 추출
        valid_indices = torch.nonzero(mask, as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return torch.tensor([]), torch.tensor([])  # 조건 만족하는 값 없음
        # 조건을 만족하는 값 중 topK
        filtered_cossim = cossim[mask]
        topk_val, topk_relative_idx = torch.topk(filtered_cossim, min(K, filtered_cossim.numel()))
        topk_idx = valid_indices[topk_relative_idx]
        return topk_idx, topk_val

    def calc_neighbor_frames(self, src, fid, K = 20):
        key = (src, fid)
        curr_idx = self.src_to_idx[key]
        neigh_idx, frame_distances = self.match_topk(self.salad_mat[curr_idx, :], self.salad_mat, K)

        if src not in self.storage:
            self.storage[src] = {}
        ckeys = []
        for n_idx in neigh_idx:
            cand_src, cand_fid = self.metadata[n_idx.item()]
            ckey = (cand_src, cand_fid)
            if key == ckey:
                continue
            ckeys.append(ckey)
            #캔디데이트 키들도 데이터를 추가
            self.storage[cand_src][cand_fid].append(key)
        self.storage[src][fid] = ckeys

#cpu에 기록 후 필요할 때 load
class DINODataManager(InferenceDataManager):
    def __init__(self):
        super().__init__()
        #storage에 patch, sample, mask, link to link 필요한가?, affinity 생성 가능.
        #패치 Xfeat 필요.
        #(patch, sample, mask, xfeat, link)

    def process(self, src, fid, data):
        #xcat, sample, mask
        if src not in self.storage:
            self.storage[src] = {}
        self.storage[src][fid] = data

    def bind_xfeat_to_patch(self, xfeat_pts, grid_shape, patch_size = 14):
        device = xfeat_pts.device
        Hp, Wp = grid_shape
        Np = Hp*Wp
        F = xfeat_pts.shape[0]

        #if F == 0 or K == 0:
        #    return torch.zeros((K, F), device=device)

        # 1. 픽셀 좌표를 패치 크기(14)로 나누어 패치 그리드 상의 (x_p, y_p) 좌표로 변환
        x_p = (xfeat_pts[:, 0] / patch_size).long()
        y_p = (xfeat_pts[:, 1] / patch_size).long()

        # 그리드 경계 조건 방어 (이미지 외곽선 노이즈 처리)
        x_p = torch.clamp(x_p, 0, Wp - 1)
        y_p = torch.clamp(y_p, 0, Hp - 1)

        # 2. 2D 패치 좌표를 1D 패치 번호(0 ~ 1529)로 환원 🌟 (이것이 F에 대응하는 패치 인덱스 리스트!)
        # f_to_patch_indices shape: [F]
        f_to_patch_indices = y_p * Wp + x_p

        # ====================================================================
        # 💡 [N x F] 원핫 스파스 행렬 빌드
        # ====================================================================
        xfeat_patch_matrix = torch.zeros(
            (Np, F), device=device, dtype=torch.float32
        )

        # 각 포인트(0 ~ F-1)가 해당하는 패치 인덱스 위치에 1.0 주입
        # scatter_ 연산을 이용해 루프 없이 원샷 도포합니다.
        xfeat_patch_matrix.scatter_(
            dim=0,
            index=f_to_patch_indices.unsqueeze(0),
            src=torch.ones((1, F), device=device),
        )
        return xfeat_patch_matrix
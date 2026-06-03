import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import requests

from torchvision import transforms
from segmodel import segheader
from io import BytesIO


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DINO_PATH = '../dinov2'
DINO_MODEL = 'dinov2_vits14'
WEIGHTS_PATH = './dinov2_vits14.pth'

class DinoV2Encoder:
    def __init__(self, model_type='vits14'):  # 'vits14', 'vitb14', 'vitl14', 'vitg14'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 페이스북(Meta)의 공식 dinov2 모델 로드
        self.model = torch.hub.load(repo_or_dir=DINO_PATH, model=DINO_MODEL, source='local',weights=WEIGHTS_PATH).to(self.device)
        self.model.eval()
        self.model = torch.compile(self.model)

        # DINOv2 표준 전처리 (224x224, ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #segmentation head
        BACKBONE_NAME = "dinov2_vits14"  # dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14
        HEAD_DATASET = "ade20k"  # voc2012 | ade20k
        HEAD_TYPE = "ms"  # ms (multi-scale)

        EMBED_DIM_MAP = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        NUM_CLASSES_MAP = {
            "voc2012": 21,
            "ade20k": 150,
        }
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        self.out_indices = self.fetch_out_indices(BACKBONE_NAME, HEAD_DATASET, HEAD_TYPE, DINOV2_BASE_URL)
        EMBED_DIM = EMBED_DIM_MAP[BACKBONE_NAME]
        NUM_CLASSES = NUM_CLASSES_MAP[HEAD_DATASET]
        IN_CHANNELS = EMBED_DIM * len(self.out_indices)

        IMAGE_SIZE = 518  # 14의 배수
        PATCH_SIZE = 14
        self.head = segheader.BNHead(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
        self.load_head_checkpoint(self.head, BACKBONE_NAME, HEAD_DATASET, HEAD_TYPE, DINOV2_BASE_URL)
        self.head.eval().to(self.device)

        # 조립
        """
        self.seg_model = segheader.DINOv2SegmentationModel(
            backbone=self.model, head=head,
            out_indices=OUT_INDICES, patch_size=PATCH_SIZE,
        )
        self.seg_model = self.seg_model.to(self.device).eval()
        print(self.seg_model)
        """
    def extract_features(
            self,
            x: torch.Tensor,
            out_indices: list,
            patch_size: int = 14,
    ) -> torch.Tensor:
        """
        백본에서 feature map (x_cat) 추출.
        이 결과를 세그멘테이션 헤드에도, 다른 헤드에도 재사용 가능.

        Returns:
            x_cat: (B, 4*embed_dim, h_feat, w_feat)
        """
        B, C, H, W = x.shape
        h_feat = H // patch_size
        w_feat = W // patch_size

        raw_features = self.model.get_intermediate_layers(
            x,
            n=[11],#out_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )

        feature_maps = []
        for feat in raw_features:
            feat = feat.reshape(B, h_feat, w_feat, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(feat)

        x_cat = torch.cat(feature_maps, dim=1)  # (B, 4*embed_dim, h, w)
        return x_cat

    def extract_features_with_attention(
            self,
            x: torch.Tensor,
            layer_idx: int = 11,  # ViT-Base 기준 마지막 블록 인덱스
            patch_size: int = 14,
            head_indices: list = [0,1,2,3,4,5]
    ) -> tuple:
        """
        백본을 단 한 번만 포워드하여
        1) 세그멘테이션용 Feature Map(x_cat)과
        2) 해당 레이어의 CLS Attention, K 벡터를 한 번에 추출합니다.
        Returns:
            x_cat : (B, embed_dim, h_feat, w_feat) -> 후속 세그헤드 입력용
            cls_attn : (B, N) -> 뭉친 패치 필터링 및 객체 앵커 선별용 (N = h_feat * w_feat)
        """
        B, C, H, W = x.shape
        h_feat = H // patch_size
        w_feat = W // patch_size
        N_patch = h_feat*w_feat

        num_heads = 6  # vits14 모델의 Attention Head 개수는 6개입니다.
        dim = 384
        head_dim = dim // num_heads

        # 💡 1. 훅(Hook) 데이터 컨테이너 및 공유 훅 정의
        attention_map = {}
        hook_data = {}

        def qkv_hook(module, inp, out):
            # out shape: [B, N+1, 3 * embed_dim] -> [B, N+1, 1152]
            # (N+1은 CLS 토큰 1개 + 패치 N개)
            hook_data['qkv'] = out.detach()

        # 💡 2. MemEffAttention 내부에 존재하는 실제 레이어인 'qkv'에 훅 등록
        out_indices = [11]
        attn_layer_idx = out_indices[-1]
        target_block = self.model.blocks[attn_layer_idx]
        handle = target_block.attn.qkv.register_forward_hook(qkv_hook)

        # 💡 3. 순방향 연산 실행 (get_intermediate_layers 내부에서 훅이 트리거됨)
        raw_features = self.model.get_intermediate_layers(
            x,
            n=out_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )

        # 훅 즉시 제거 (메모리 누수 방지 및 다음 프레임 오염 차단)
        handle.remove()

        # 💡 4. Feature Map 성형 [B, D, H_p, W_p]
        feature_maps = []
        for feat in raw_features:
            feat = feat.reshape(B, h_feat, w_feat, -1).permute(0, 3, 1, 2).contiguous()
            feature_maps.append(feat)
        x_cat = torch.cat(feature_maps, dim=1)

        #QKV 파싱
        qkv = hook_data['qkv']  # [B, N+1, 1152]
        total_tokens = qkv.shape[1]  # N+1

        # [B, N+1, 3, num_heads, head_dim] 형태로 분할 후 Q, K, V 추출
        qkv = qkv.reshape(B, total_tokens, 3, num_heads, head_dim)
        q = qkv[:, :, 0]  # [B, N+1, num_heads, head_dim]
        k = qkv[:, :, 1]  # [B, N+1, num_heads, head_dim]

        # Scaled Dot-Product Attention 수학적 연산을 위해 축 변경 [B, num_heads, N+1, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        q = q[:, head_indices, :, :]
        k = k[:, head_indices, :, :]

        # 전역 어텐션 행렬 연산 수행 (Q @ K_T)
        # [B, num_heads, N+1, head_dim] @ [B, num_heads, head_dim, N+1] -> [B, num_heads, N+1, N+1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 💡 7. CLS 토큰(0번 행)이 나머지 패치(1번 열 이후)를 바라보는 지분만 슬라이싱
        cls_attn = attn_weights[:, :, 0, 1:]  # [B, num_heads, num_patches]
        cls_attn = cls_attn.mean(dim=1)

        """
        ##K의 어피니티를 활용. 아니면 K값만.
        k_patches = k[:, :, 1:, :]
        #k_patches = k_patches.transpose(1, 2).reshape(B, N_patch, 64*len(head_indices))
        B, num_heads, N, head_dim = k_patches.shape
        k_patches = k_patches.view(B, num_heads, h_feat, w_feat, head_dim)
        k_patches = k_patches.permute(0, 2,3,1,4).contiguous()
        k_patches = k_patches.view(B, h_feat, w_feat, num_heads*head_dim)
        k_patches = k_patches.permute(0, 3, 1, 2).contiguous()
        """
        #print(k_patches.shape)
        #k_patches = k_patches.view(N, D)

        #k_patches_norm = F.normalize(k_patches, p=2, dim=-1)
        #k_affinity_matrix = torch.matmul(k_patches_norm, k_patches_norm.transpose(-2, -1))

        #features = x_cat[0].view(384, -1).t()  # [N, D]
        #feat_last_64 = features[:, -64:]
        #is_same = torch.allclose(feat_last_64, K_flat, atol=1e-6)
        #print("test K = ", x_cat.shape,k_patches.shape, q.shape, cls_attn.shape)

        return x_cat, cls_attn


    def seg_head_forward(
            self,
            x_cat: torch.Tensor,  # extract_features()의 출력을 그대로 받음
            original_hw: tuple,  # (H, W) — upsample 목표 크기
    ) -> torch.Tensor:
        """
        x_cat을 받아서 세그멘테이션 logits 반환.
        백본 없이 헤드만 독립 실행 가능.
        """
        H, W = original_hw
        logits = self.head(x_cat)  # (B, num_classes, h, w)
        logits = F.interpolate(logits, size=(H, W),
                               mode='bilinear', align_corners=False)  # (B, num_classes, H, W)
        return logits

    def preprocess_cv2(self, img_rgb: np.ndarray, patch_size=14, target_size=None):
        H, W = img_rgb.shape[:2]

        if target_size is not None:
            # 짧은 변 기준 리사이즈 후 14 배수로 내림
            scale = target_size / min(H, W)
            new_W = int(W * scale // patch_size) * patch_size
            new_H = int(H * scale // patch_size) * patch_size
        else:
            # 리사이즈 없이 14 배수로만 내림 (640→630, 480→476)
            new_W = (W // patch_size) * patch_size
            new_H = (H // patch_size) * patch_size

        img_resized = cv2.resize(img_rgb, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)
        img_np = img_resized.astype(np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return tensor, (W, H)

    @torch.no_grad()
    def encode_image(self, image_bgr):
        pass

    @torch.no_grad()
    def encode_object(self, image_bgr, bbox, contour=None):
        """
        image_bgr : OpenCV BGR 이미지 (H, W, 3), numpy
        bbox      : [x1, y1, x2, y2], tensor or numpy
        contour   : (N, 2) tensor or numpy, 없으면 배경 제거 생략
        반환      : (1, M) torch.Tensor on GPU, L2 정규화됨
        """
        # 1. bbox crop
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        x1, y1, x2, y2 = bbox.astype(np.int32)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)

        roi = image_bgr[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return None

        # 2. 배경 제거 (contour 있을 때만)
        if contour is not None:
            if isinstance(contour, torch.Tensor):
                contour = contour.cpu().numpy()

            # contour 좌표를 crop 기준으로 변환
            contour_local = contour.astype(np.int32) - np.array([x1, y1])

            # 마스크 생성
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour_local], 255)

            # ImageNet mean으로 배경 채우기 (0이 아닌 중립값)
            imagenet_mean_bgr = np.array([0.406, 0.456, 0.485], dtype=np.float32) * 255
            bg = np.full_like(roi, imagenet_mean_bgr, dtype=np.float32)
            roi = np.where(mask[:, :, None] > 0, roi.astype(np.float32), bg)
            roi = roi.astype(np.uint8)

        # 3. BGR → RGB → PIL → transform
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 4. DINOv2 추론 (CLS token)
        features = self.model(input_tensor)  # (1, M)

        # 5. L2 정규화
        features = F.normalize(features, p=2, dim=1)

        return features  # GPU Tensor (1, M)

    @torch.no_grad()
    def encode_objects_batch(self, image_bgr, seg_objects):
        """
        모든 crop을 하나의 배치로 묶어 단일 forward pass
        """
        tensors = []
        valid_ids = []

        for i, obj in enumerate(seg_objects):
            # bbox crop
            bbox = obj['bbox']
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()
            x1, y1, x2, y2 = bbox.astype(np.int32)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_bgr.shape[1], x2), min(image_bgr.shape[0], y2)

            roi = image_bgr[y1:y2, x1:x2].copy()
            if roi.size == 0:
                continue

            # 배경 제거
            contour = obj['contour']
            if contour is not None:
                if isinstance(contour, torch.Tensor):
                    contour = contour.cpu().numpy()
                contour_local = contour.astype(np.int32) - np.array([x1, y1])
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour_local], 255)
                imagenet_mean_bgr = np.array([0.406, 0.456, 0.485]) * 255
                bg = np.full_like(roi, imagenet_mean_bgr, dtype=np.float32)
                roi = np.where(mask[:, :, None] > 0, roi.astype(np.float32), bg)
                roi = roi.astype(np.uint8)

            # transform만 적용 (model 호출 X)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(roi_rgb)
            tensors.append(self.transform(pil_img))  # (3, 224, 224)
            """
            roi_tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
            roi_tensor = TF.resize(roi_tensor, [224, 224],
                                   interpolation=TF.InterpolationMode.BICUBIC)
            roi_tensor = TF.normalize(roi_tensor,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            tensors.append(roi_tensor)
            """
            valid_ids.append(i)

        if not tensors:
            return None, []

        # ★ 핵심: 단일 forward pass
        batch = torch.stack(tensors).to(self.device)  # (N, 3, 224, 224)
        features = self.model(batch)  # (N, M) — 1회 추론
        features = F.normalize(features, p=2, dim=1)  # (N, M)

        return features, valid_ids

    @torch.no_grad()
    def encode_objects_batch2(self, image_bgr, seg_objects):
        """
        프레임 내 모든 객체를 배치로 인코딩 (효율적)
        image_bgr  : OpenCV BGR 이미지
        seg_objects: matcher.seg_storage[src][fid]['objects'] 리스트
        반환       : (N, M) torch.Tensor on GPU
        """
        tensors = []
        valid_ids = []

        for i, obj in enumerate(seg_objects):
            a = time.time()
            t = self.encode_object(
                image_bgr,
                bbox=obj['bbox'],
                contour=obj['contour']
            )
            b = time.time()
            if t is not None:
                tensors.append(t)
                valid_ids.append(i)
            c = time.time()
            print('batch', i, b-a, c-b, c-a)

        if not tensors:
            return None, []

        batch = torch.cat(tensors, dim=0)  # (N, M)
        print('dino res', batch.shape)
        return batch, valid_ids  # GPU Tensor + 유효 인덱스

    @torch.no_grad()
    def encode_mask(self, image, mask):
        """
        image: OpenCV BGR 이미지 (H, W, 3)
        mask: 해당 객체의 Binary Mask (H, W), 값은 0 또는 255 (혹은 True/False)
        """
        # 1. 마스크 영역 추출 (Bounding Box 크롭)
        y, x = np.where(mask > 0)
        if len(x) == 0 or len(y) == 0:
            return None

        x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
        roi = image[y1:y2 + 1, x1:x2 + 1]

        # 2. 배경 제거 (선택 사항: 객체 특징만 부각시키기 위해 마스크 적용)
        # roi_mask = mask[y1:y2+1, x1:x2+1]
        # roi = cv2.bitwise_and(roi, roi, mask=roi_mask.astype(np.uint8))

        # 3. PIL 변환 및 전처리
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(roi_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 4. 특징 추출 (CLS 토큰 사용)
        features = self.model(input_tensor)  # (1, embedding_dim)

        # 5. L2 정규화 (코사인 유사도 계산을 위해 필수)
        features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()  # (1, M) 형태의 numpy 배열 반환

    def fetch_out_indices(self, backbone_name, head_dataset, head_type, DINOV2_BASE_URL):

        config_url = (
            f"{DINOV2_BASE_URL}/{backbone_name}/"
            f"{backbone_name}_{head_dataset}_{head_type}_config.py"
        )
        print(f"Config  다운로드: {config_url}")
        resp = requests.get(config_url, timeout=30)
        resp.raise_for_status()
        ns = {}
        exec(resp.text, ns)
        out_indices = list(ns["model"]["backbone"]["out_indices"])
        print(f"out_indices: {out_indices}")
        return out_indices

    def load_head_checkpoint(self, head: segheader.BNHead, backbone_name, head_dataset, head_type, DINOV2_BASE_URL):
        """
        fbaipublicfiles에서 헤드 체크포인트 다운로드 후 BNHead에 로드.

        체크포인트 예시 키:
            decode_head.bn.weight        (SyncBN → BN2d, 키 이름 동일)
            decode_head.conv_seg.weight  shape=[num_classes, in_channels, 1, 1]
        """
        ckpt_url = (
            f"{DINOV2_BASE_URL}/{backbone_name}/"
            f"{backbone_name}_{head_dataset}_{head_type}_head.pth"
        )
        print(f"Checkpoint 다운로드: {ckpt_url}")
        resp = requests.get(ckpt_url, stream=True, timeout=120)
        resp.raise_for_status()

        ckpt = torch.load(BytesIO(resp.content), map_location="cpu", weights_only=False)
        raw_state = ckpt.get("state_dict", ckpt)

        # decode_head.* 키만 추출, 프리픽스 제거
        head_state = {}
        for k, v in raw_state.items():
            if k.startswith("decode_head."):
                head_state[k[len("decode_head."):]] = v

        if not head_state:
            raise RuntimeError(
                f"'decode_head.*' 키 없음. 발견된 키: {list(raw_state.keys())[:10]}"
            )

        missing, unexpected = head.load_state_dict(head_state, strict=True)
        if missing:    print(f"[경고] Missing   : {missing}")
        if unexpected: print(f"[경고] Unexpected: {unexpected}")
        print(f"헤드 로드 완료 ({len(head_state)} keys)")

    def masked_average_pool(self,
            feat_map: torch.Tensor,
            mask_np: np.ndarray,
    ) -> tuple:
        """
        Args:
            feat_map : [H_p, W_p, OUT_DIM]  float32, on DEVICE
            mask_np  : [H_orig, W_orig]     bool numpy array
                       RF-DETR detections.mask[i] 원본 해상도 (480, 640)

        Returns:
            vec       : [OUT_DIM]  float32  (마스크가 빈 경우 zero vector)
            n_patches : int        패치 그리드 상 마스크 면적 (품질 지표)

        Note:
            (480, 640) → (H_p, W_p) 단일 nearest 리사이즈.
            중간 DINOv2 입력 크기(476, 630)를 거칠 필요 없음 — nearest는
            단계를 합쳐도 결과가 동일하고 오히려 정확도 손실이 없음.
        """
        H_p, W_p, D = feat_map.shape

        # (480, 640) bool numpy → float tensor [1, 1, 480, 640]
        mask_t = torch.from_numpy(mask_np.astype(np.float32)).to(self.device)
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)

        # 패치 그리드(H_p, W_p)로 직접 다운샘플 (nearest: 경계 보존)
        mask_patch = F.interpolate(
            mask_t, size=(H_p, W_p), mode="nearest"
        ).squeeze()  # [H_p, W_p]

        mask_bool = mask_patch > 0.5  # [H_p, W_p] bool
        n_patches = int(mask_bool.sum().item())

        if n_patches == 0:
            return torch.zeros(D, device=self.device), 0

        feat_flat = feat_map.reshape(-1, D)  # [H_p*W_p, D]
        mask_flat = mask_bool.reshape(-1)  # [H_p*W_p]

        vec = feat_flat[mask_flat].mean(dim=0)  # [D]
        return vec, n_patches

    def batch_masked_average_pool(self,
                                  feat_map: torch.Tensor,
                                  masks: np.ndarray,  # [K, H_orig, W_orig]
                                  ) -> tuple:
        """
        Args:
            feat_map : [1, D, H_p, W_p]   float32
            masks    : [K, H_orig, W_orig] bool numpy or tensor
        """
        K = masks.shape[0]
        _, D, H_p, W_p = feat_map.shape

        # 1. K=0 예외 처리 (검출된 객체가 없을 때)
        if K == 0:
            return torch.empty((0, D), device=self.device), torch.zeros(0, device=self.device)

        # 2. 마스크를 Tensor로 변환 및 [K, 1, H, W] 형태로 준비
        if isinstance(masks, np.ndarray):
            masks_t = torch.from_numpy(masks).to(self.device).float()
        else:
            masks_t = masks.to(self.device).float()

        if masks_t.dim() == 3:
            masks_t = masks_t.unsqueeze(1)

        # 3. 보간법을 이용한 리사이즈 [K, 1, H_p, W_p]
        # nearest 모드는 bool 성질을 가장 잘 유지함
        masks_patch = F.interpolate(masks_t, size=(H_p, W_p), mode="nearest")
        masks_bool = (masks_patch.squeeze(1) > 0.5).float()  # [K, H_p, W_p]

        # 4. 각 마스크별 유효 패치 개수 계산 [K]
        n_patches = masks_bool.sum(dim=(1, 2))

        # 5. 행렬 곱셈을 이용한 고속 연산
        # feat_flat: [D, H_p*W_p]
        # masks_flat: [K, H_p*W_p]
        feat_flat = feat_map.view(D, -1)
        masks_flat = masks_bool.view(K, -1)

        # [K, H_p*W_p] @ [H_p*W_p, D] -> [K, D]
        sum_vecs = torch.mm(masks_flat, feat_flat.t())

        # 6. 평균 계산 (n_patches가 0인 마스크는 0 벡터 반환)
        denom = n_patches.unsqueeze(1)
        vecs = torch.where(denom > 0, sum_vecs / denom, torch.zeros_like(sum_vecs))

        return vecs, n_patches

    # ── 시각화 ────────────────────────────────────────────────────────────────────
    def visualize_segmentation(
            self,
            original_img: Image.Image,
            seg_map: np.ndarray,  # (H, W) int — 클래스 인덱스
            classes: list,
            palette: np.ndarray,  # (N, 3) uint8
            alpha: float = 0.55,
            figsize: tuple = (20, 7),
            output_path: str = None,
    ):
        """
        3열 시각화: 원본 / 세그멘테이션 맵 / 오버레이 + 레전드

        Args:
            alpha      : 오버레이에서 세그멘테이션 비율 (0=원본, 1=세그맵)
            output_path: 지정 시 파일 저장
        """
        colored = palette[seg_map]  # (H, W, 3)
        orig_np = np.array(original_img, dtype=np.float32)
        blended = ((1 - alpha) * orig_np + alpha * colored.astype(np.float32)) \
            .clip(0, 255).astype(np.uint8)

        present_ids = np.unique(seg_map)
        legend = [
            mpatches.Patch(
                facecolor=palette[i] / 255.0,
                edgecolor="white",
                linewidth=0.5,
                label=f"{i}: {classes[i]}" if i < len(classes) else f"class_{i}",
            )
            for i in present_ids if i < len(palette)
        ]

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, title, im in zip(
                axes,
                ["Original", "Segmentation Map", f"Overlay  α={alpha}"],
                [np.array(original_img), colored, blended],
        ):
            ax.imshow(im)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
            ax.axis("off")

        fig.legend(
            handles=legend, loc="lower center",
            ncol=min(len(legend), 7), fontsize=10,
            bbox_to_anchor=(0.5, -0.06), frameon=True,
            fancybox=True, framealpha=0.9,
        )
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"저장: {output_path}")

        plt.show()
        print(f"검출 클래스 ({len(present_ids)}개): "
              f"{[classes[i] for i in present_ids if i < len(classes)]}")

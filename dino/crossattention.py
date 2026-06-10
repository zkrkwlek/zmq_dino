import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameMatchedCrossAttention(nn.Module):
    # ViT-Small 기준: embed_dim=384, num_heads=6
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 384 // 6 = 64
        #self.scale = self.head_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)  # exp(2.6592) ≒ 14.2 (CLIP 기본값)
        self.gamma = nn.Parameter(torch.ones(1)*0.1) #zeros

        # Q, K, V 프로젝션 (384차원 유지)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, current_patches, retrieved_memory_patches, target_indices=None):
        """
        current_patches: [B, N_c, 384] (현재 프레임 DINO 패치)
        retrieved_memory_patches: [B, N_m, 384] (SALAD로 찾아낸 과거 프레임의 DINO 패치)
        """
        N_c, C = current_patches.shape
        N_m, _ = retrieved_memory_patches.shape

        # 1. Q, K, V 생성 및 멀티헤드 분할
        #Q = self.q_proj(current_patches).view(N_c, self.num_heads, self.head_dim).transpose(0,1)
        #K = self.k_proj(retrieved_memory_patches).view(N_m, self.num_heads, self.head_dim).transpose(0,1)
        #V = self.v_proj(retrieved_memory_patches).view(N_m, self.num_heads, self.head_dim).transpose(0,1)
        Q = current_patches.unsqueeze(0)  # [1530, 1, 384]
        K = retrieved_memory_patches.unsqueeze(0)  # [3060, 1, 384]
        V = retrieved_memory_patches.unsqueeze(0)  # [3060, 1, 384]

        # 2. 패치 단위 코사인 유사도 연산 (L2 정규화 후 내적)
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)

        # [B, 6, N_c, N_m] 크기의 유사도(Attention) 맵
        scale_factor = torch.exp(self.logit_scale)
        attn_weights = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) * scale_factor
        attn_probs = F.softmax(attn_weights, dim=-1)

        # 3. 정보 융합 및 출력
        fused = torch.matmul(attn_probs, V).transpose(0,1).contiguous().view(N_c, C)
        output = current_patches + self.gamma * fused#self.out_proj(fused)

        # attention
        last_head_attn = attn_probs[-1]  # [1530, 3060] (마지막 헤드 사용)
        #last_head_attn = attn_probs.mean(dim=0)
        if target_indices is not None and len(target_indices) > 0:
            target_attn = last_head_attn[:, target_indices]
            final_heatmap = target_attn.max(dim=1).values.unsqueeze(0)
            print(final_heatmap.shape)
            return output, final_heatmap
        else:
            return output, last_head_attn.unsqueeze(0)

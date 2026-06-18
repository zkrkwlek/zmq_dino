"""
global_object_bank.py
────────────────────────────────────────────────────────────────
GlobalObjectBank

객체별 avg_vec를 EMA로 관리하며 품질 메트릭으로 메모리 뱅크를 유지.

품질 기준 (filter_by_quality 와 동일):
    ① pure_area    : 독점 패치 없으면 불안정
    ② pure_vom_ratio: 혼합 신호 앵커 (여러 객체 동시 반응)
    ③ spatial_std  : vom 공간 발산 → 배경/바닥 앵커

EMA 트래킹:
    - stability      : EMA(pure_area > 0) → 1에 가까울수록 안정적 객체
    - pure_area_ema  : EMA(pure_area_cur)
    - spatial_std_ema: EMA(spatial_std_cur) → 높아지면 발산 중
    - centroid_history: vom 공간 중심 이력 → 분산 크면 split 후보

주요 API:
    register_or_update(avg_vecs, pure_areas, frame_id,
                       vom=None, grid_shape=None,
                       pure_vom_ratios=None, spatial_stds=None)
    remove_diverging() → 발산 entry 삭제
    get_quality_summary() → 전체 상태 출력
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────

@dataclass
class ObjectEntry:
    gid: int
    avg_vec: torch.Tensor              # [D] L2 정규화

    # 안정성
    stability: float = 0.0             # EMA(pure_area > 0)
    pure_area_ema: float = 0.0         # EMA(pure_area_cur)

    # 품질 메트릭 EMA
    pure_vom_ratio_ema: float = 1.0    # EMA(pure/vom). 낮아지면 혼합 신호
    spatial_std_ema: float = 0.0       # EMA(spatial_std). 높아지면 발산

    # 공간 이력
    centroid_history: deque = field(
        default_factory=lambda: deque(maxlen=10))   # (cx, cy) 패치 좌표
    split_candidate: bool = False      # centroid 분산 과다 → 분리 후보

    frame_count: int = 0
    last_frame_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# GlobalObjectBank
# ─────────────────────────────────────────────────────────────────

class GlobalObjectBank:
    """
    Args:
        reid_threshold       : 동일 객체 판정 코사인 유사도 하한 (기본 0.75)
        ema_alpha            : EMA 갱신 비율 (기본 0.3)
        stability_threshold  : 이 값 미만이면 발산 entry (기본 0.3)
        min_frames_to_judge  : 판단에 필요한 최소 프레임 수 (기본 5)
        centroid_var_threshold: centroid 분산 초과 시 split 후보 (기본 4.0 패치²)
        max_spatial_std      : spatial_std_ema 초과 시 발산 판정 (기본 10.0)
    """

    def __init__(self,
                 reid_threshold: float = 0.75,
                 ema_alpha: float = 0.3,
                 stability_threshold: float = 0.3,
                 min_frames_to_judge: int = 5,
                 centroid_var_threshold: float = 4.0,
                 max_spatial_std: float = 10.0):
        self.reid_threshold        = reid_threshold
        self.ema_alpha             = ema_alpha
        self.stability_threshold   = stability_threshold
        self.min_frames_to_judge   = min_frames_to_judge
        self.centroid_var_threshold = centroid_var_threshold
        self.max_spatial_std       = max_spatial_std

        self._entries: Dict[int, ObjectEntry] = {}
        self._next_gid = 0

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    def register_or_update(self,
                           avg_vecs: torch.Tensor,        # [K, D]
                           pure_areas: torch.Tensor,      # [K] float
                           frame_id: Optional[str] = None,
                           vom: Optional[torch.Tensor] = None,      # [K, N] bool
                           grid_shape: Optional[Tuple[int, int]] = None,
                           pure_vom_ratios: Optional[torch.Tensor] = None,  # [K]
                           spatial_stds: Optional[torch.Tensor] = None,     # [K]
                           ) -> List[int]:
        """
        K개 앵커를 기존 entry와 매칭 후 업데이트 또는 신규 등록.

        centroid는 vom + grid_shape로 자동 계산 (없으면 (-1,-1) 저장).
        pure_vom_ratios / spatial_stds 없으면 vom에서 직접 계산.

        Returns:
            gids : [K] int
        """
        K = avg_vecs.shape[0]
        if K == 0:
            return []

        avg_vecs_norm = F.normalize(avg_vecs, p=2, dim=1)

        # pure_vom_ratio, spatial_std 계산
        pvr = self._resolve_pure_vom_ratio(avg_vecs, vom, pure_areas, pure_vom_ratios)
        sstd = self._resolve_spatial_std(vom, grid_shape, spatial_stds, K)

        # centroid 계산 (vom + grid_shape)
        centroids_xy = self._compute_centroids_from_vom(vom, grid_shape, K)

        gids = []
        for k in range(K):
            vec       = avg_vecs_norm[k]
            pure_area = float(pure_areas[k].item())
            cx, cy    = centroids_xy[k]
            pv_ratio  = float(pvr[k])
            sp_std    = float(sstd[k])

            matched_gid = self._find_match(vec)
            if matched_gid is None:
                gid = self._register_new(vec, cx, cy, pure_area,
                                         pv_ratio, sp_std, frame_id)
            else:
                gid = matched_gid
                self._update(gid, vec, cx, cy, pure_area,
                             pv_ratio, sp_std, frame_id)
            gids.append(gid)

        # 이번 프레임에 보이지 않은 entry stability 패널티
        seen_gids = set(gids)
        for gid, entry in self._entries.items():
            if gid not in seen_gids and entry.last_frame_id != frame_id:
                entry.stability = entry.stability * (1 - self.ema_alpha)

        return gids

    def query(self, avg_vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """현재 avg_vec와 bank 전체를 비교. Returns (best_gids, best_sims)."""
        if self.size == 0 or avg_vecs.shape[0] == 0:
            K = avg_vecs.shape[0]
            return (torch.full((K,), -1, dtype=torch.long),
                    torch.zeros(K))

        bank_vecs, bank_gids = self._get_bank_matrix(avg_vecs.device)
        sims = torch.mm(F.normalize(avg_vecs, p=2, dim=1), bank_vecs.t())
        best_vals, best_idx = sims.max(dim=1)

        best_gids = torch.where(
            best_vals >= self.reid_threshold,
            bank_gids[best_idx],
            torch.full_like(bank_gids[best_idx], -1)
        )
        return best_gids, best_vals

    def get_stable_entries(self) -> List[ObjectEntry]:
        return [e for e in self._entries.values()
                if e.frame_count >= self.min_frames_to_judge
                and e.stability >= self.stability_threshold
                and e.spatial_std_ema < self.max_spatial_std]

    def get_diverging_entries(self) -> List[ObjectEntry]:
        """발산 entry: stability 낮거나 spatial_std 높음."""
        return [e for e in self._entries.values()
                if e.frame_count >= self.min_frames_to_judge
                and (e.stability < self.stability_threshold
                     or e.spatial_std_ema >= self.max_spatial_std)]

    def get_split_candidates(self) -> List[ObjectEntry]:
        return [e for e in self._entries.values() if e.split_candidate]

    def remove_diverging(self) -> List[int]:
        """발산 entry 삭제. 삭제된 gid 목록 반환."""
        to_remove = [e.gid for e in self.get_diverging_entries()]
        for gid in to_remove:
            del self._entries[gid]
        return to_remove

    def get_all_vecs(self, device) -> Tuple[torch.Tensor, List[int]]:
        if self.size == 0:
            return torch.zeros((0, 1), device=device), []
        vecs = torch.stack([e.avg_vec for e in self._entries.values()]).to(device)
        gids = list(self._entries.keys())
        return vecs, gids

    def get_quality_summary(self) -> dict:
        """전체 entry 품질 요약 반환."""
        if self.size == 0:
            return {'total': 0}
        stabilities   = [e.stability for e in self._entries.values()]
        pure_areas    = [e.pure_area_ema for e in self._entries.values()]
        pvr           = [e.pure_vom_ratio_ema for e in self._entries.values()]
        sstd          = [e.spatial_std_ema for e in self._entries.values()]
        return {
            'total'          : self.size,
            'stable'         : len(self.get_stable_entries()),
            'diverging'      : len(self.get_diverging_entries()),
            'split_candidate': len(self.get_split_candidates()),
            'stability_mean' : float(np.mean(stabilities)),
            'pure_area_mean' : float(np.mean(pure_areas)),
            'pvr_mean'       : float(np.mean(pvr)),
            'spatial_std_mean': float(np.mean(sstd)),
        }

    def print_summary(self):
        s = self.get_quality_summary()
        print(f"[ObjectBank] total={s['total']}  stable={s['stable']}  "
              f"diverging={s['diverging']}  split?={s['split_candidate']}")
        print(f"             stability={s['stability_mean']:.2f}  "
              f"pvr={s['pvr_mean']:.2f}  "
              f"spatial_std={s['spatial_std_mean']:.1f}")
        for e in self._entries.values():
            flags = ''
            if e.split_candidate:
                flags += ' [SPLIT?]'
            if (e.frame_count >= self.min_frames_to_judge
                    and (e.stability < self.stability_threshold
                         or e.spatial_std_ema >= self.max_spatial_std)):
                flags += ' [DIV]'
            print(f"  gid={e.gid:3d}  stab={e.stability:.2f}  "
                  f"pure_ema={e.pure_area_ema:.1f}  "
                  f"pvr={e.pure_vom_ratio_ema:.2f}  "
                  f"sstd={e.spatial_std_ema:.1f}  "
                  f"frames={e.frame_count}{flags}")

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _find_match(self, vec: torch.Tensor) -> Optional[int]:
        best_gid, best_sim = None, -1.0
        for gid, entry in self._entries.items():
            sim = float(torch.dot(vec, entry.avg_vec.to(vec.device)).item())
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        return best_gid if best_sim >= self.reid_threshold else None

    def _register_new(self, vec, cx, cy, pure_area,
                      pv_ratio, sp_std, frame_id) -> int:
        gid = self._next_gid
        self._next_gid += 1
        e = ObjectEntry(
            gid=gid,
            avg_vec=vec.cpu(),
            stability=1.0 if pure_area > 0 else 0.0,
            pure_area_ema=pure_area,
            pure_vom_ratio_ema=pv_ratio,
            spatial_std_ema=sp_std,
            frame_count=1,
            last_frame_id=frame_id,
        )
        e.centroid_history.append((cx, cy))
        self._entries[gid] = e
        return gid

    def _update(self, gid, vec, cx, cy, pure_area,
                pv_ratio, sp_std, frame_id):
        e = self._entries[gid]
        # 프레임당 1회만 갱신 — 같은 프레임 내 여러 앵커가 같은 gid에 매칭돼도
        # EMA/centroid/frame_count 가 중복 갱신되지 않도록 첫 매칭만 반영
        if frame_id == e.last_frame_id:
            return
        a = self.ema_alpha

        # avg_vec EMA
        new_vec = (1 - a) * e.avg_vec.to(vec.device) + a * vec
        e.avg_vec = F.normalize(new_vec, p=2, dim=0).cpu()

        # stability EMA
        e.stability        = (1 - a) * e.stability        + a * (1.0 if pure_area > 0 else 0.0)
        e.pure_area_ema    = (1 - a) * e.pure_area_ema    + a * pure_area
        e.pure_vom_ratio_ema = (1 - a) * e.pure_vom_ratio_ema + a * pv_ratio
        e.spatial_std_ema  = (1 - a) * e.spatial_std_ema  + a * sp_std

        # centroid 이력 + split 후보 판정
        e.centroid_history.append((cx, cy))
        if len(e.centroid_history) >= 4:
            xs = [p[0] for p in e.centroid_history if p[0] >= 0]
            ys = [p[1] for p in e.centroid_history if p[1] >= 0]
            if len(xs) >= 4:
                var = float(np.var(xs) + np.var(ys))
                e.split_candidate = var > self.centroid_var_threshold

        e.frame_count   += 1          # 위 early-return 으로 고유 프레임당 1회 보장
        e.last_frame_id  = frame_id

    def _compute_centroids_from_vom(self, vom, grid_shape, K):
        """vom [K, N] bool + grid_shape → [(cx, cy), ...] 현재 프레임 기준."""
        centroids = [(-1.0, -1.0)] * K
        if vom is None or grid_shape is None:
            return centroids
        H_p, W_p = grid_shape
        vom_np = vom.cpu().numpy()
        for k in range(K):
            idx = np.where(vom_np[k])[0]
            if len(idx) == 0:
                continue
            rows = (idx // W_p).astype(float)
            cols = (idx % W_p).astype(float)
            centroids[k] = (float(cols.mean()), float(rows.mean()))
        return centroids

    def _resolve_pure_vom_ratio(self, avg_vecs, vom, pure_areas, pvr_given):
        K = avg_vecs.shape[0]
        if pvr_given is not None:
            return pvr_given.cpu().float().tolist()
        if vom is not None:
            vom_sizes  = vom.sum(dim=1).float().cpu()
            pure_sizes = pure_areas.float().cpu()
            return (pure_sizes / vom_sizes.clamp(min=1.0)).tolist()
        return [1.0] * K

    def _resolve_spatial_std(self, vom, grid_shape, sstd_given, K):
        if sstd_given is not None:
            return sstd_given.cpu().float().tolist()
        if vom is not None and grid_shape is not None:
            H_p, W_p = grid_shape
            vom_np = vom.cpu().numpy()
            result = []
            for k in range(K):
                idx = np.where(vom_np[k])[0]
                if len(idx) < 2:
                    result.append(0.0)
                    continue
                rows = (idx // W_p).astype(float)
                cols = (idx % W_p).astype(float)
                result.append(float(np.std(rows) + np.std(cols)))
            return result
        return [0.0] * K

    def _get_bank_matrix(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        vecs = torch.stack([e.avg_vec for e in self._entries.values()]).to(device)
        gids = torch.tensor(list(self._entries.keys()), dtype=torch.long, device=device)
        return vecs, gids

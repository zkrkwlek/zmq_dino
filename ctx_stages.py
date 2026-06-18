"""
ctx_stages.py — 통합 data + ctx 파이프라인 (임시 참조 구현, STEP 1)
────────────────────────────────────────────────────────────────
기존 base 알고리즘(filter_by_quality / merge_anchors_heatmap /
_apply_group_and_recompute / compute_anchor_patch_context / ...)은
그대로 두고, data/ctx 사이 "필드 변환 + 호출"만 담당하는 어댑터 계층.

데이터 스키마
─────────────
data (frame / memory 공용, type 으로 구분):
    {
      "type":       "frame" | "memory",
      "x_cat":      tensor | None,      # memory: None
      "feat":       tensor,             # frame: [N,D] 패치피처 / memory: [K,D] 객체벡터
      "attn":       tensor | None,      # memory: None
      "grid_shape": (H_p, W_p) | None,  # memory: None
      "meta":       None | [K개 {gid, src, fid, idx}],  # frame: None / memory: 있음
    }

ctx (가공):
    {"sim"[K,N], "heat"[K,K], "vom"[K,N], "pure"[K,N],
     "oc"[N], "vec"[K,D], "centroid"[K], "keep"[K]}
    # 디퍼(재가공, transient): _group_labels, _n_comp, _H, meta

사용 예
───────
    from dino.patchcluster_v2 import DinoSemanticObjectExtractorV2
    class V2Ctx(CtxPipelineMixin, DinoSemanticObjectExtractorV2):
        pass
    objp = V2Ctx()
    # 단일 프레임:
    d1  = make_frame_data(x_cat, feat, attn, grid_shape)
    ctx = objp.ctx_generate_seeds(d1)
    ctx = objp.ctx_compute_context(d1, d1, ctx)        # 단일: data2 = data1
    ctx = objp.ctx_group_anchors(d1, ctx)
    ctx = objp.ctx_compute_anchor_response(d1, ctx, grid_shape=grid_shape)
    ctx = objp.ctx_object_vectors(d1, ctx, repr="avg")
    ctx = objp.ctx_filter_anchors(d1, ctx, min_pure_response=1,
                                  min_pure_vom_ratio=0.10, max_spatial_std=8.0)
    # 크로스 프레임:
    d2  = objp.build_memory_data(dino_mgr, neigh_keys, repr="avg")   # type="memory"
    ctxc = objp.ctx_compute_context(d1, d2)            # 메모리 벡터를 d1 에 투영
    ctxc = objp.ctx_filter_anchors(d1, ctxc, min_pure_response=3, ...)
"""
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# data 빌더 (frame / memory 공용)
# ──────────────────────────────────────────────────────────────
def make_frame_data(x_cat, feat, attn, grid_shape):
    return {"type": "frame", "x_cat": x_cat, "feat": feat,
            "attn": attn, "grid_shape": grid_shape, "meta": None}


def make_memory_data(feat, meta=None, grid_shape=None):
    return {"type": "memory", "x_cat": None, "feat": feat,
            "attn": None, "grid_shape": grid_shape, "meta": meta}


def is_memory(data):
    return data is not None and data.get("type") == "memory"


def new_ctx():
    return {k: None for k in
            ("sim", "heat", "vom", "pure", "oc", "vec", "centroid", "keep")}


class CtxPipelineMixin:
    """DinoSemanticObjectExtractorV2 에 믹스인. self.* 로 base 알고리즘 호출."""

    # ── 내부: 우리 ctx → base 알고리즘이 기대하는 레거시 ctx(dict) 변환 ──
    @staticmethod
    def _to_legacy_ctx(ctx, centroids, N):
        vom = ctx.get("vom")
        return {
            "centroids":   centroids,
            "sim_matrix":  ctx.get("sim"),
            "vom":         vom,
            "pure":        ctx.get("pure"),
            "oc":          ctx.get("oc"),
            "heatmap_sim": ctx.get("heat"),
            "H_matrix":    ctx.get("_H"),
            "K":           (0 if vom is None else vom.shape[0]),
            "N":           N,
        }

    # ── STAGE: 시드 생성 (frame 전용) → ctx['centroid'] ──
    def ctx_generate_seeds(self, data1, ctx=None, spatial_radius=3, sim_thresh=0.7):
        ctx = ctx if ctx is not None else new_ctx()
        mask, _ = self.generate_mask(data1["feat"], data1["grid_shape"],
                                     spatial_radius=spatial_radius, sim_thresh=sim_thresh)
        ctx["centroid"] = self.sample_patch(data1["attn"], mask)
        return ctx

    # ── STAGE: 컨텍스트 계산 (단일/크로스 공용) ──
    #   data1 = 타겟 프레임(feat=N×D 점수 대상)
    #   data2 = 쿼리원: frame 이면 ctx['centroid'](시드 인덱스), memory 이면 data2['feat'](K×D)
    def ctx_compute_context(self, data1, data2, ctx=None, **opts):
        ctx = ctx if ctx is not None else new_ctx()
        feat = data1["feat"]
        if is_memory(data2):
            query = data2["feat"]                                   # [K, D] 메모리 벡터
            ctx["centroid"] = torch.arange(query.shape[0], device=feat.device)
            ctx["meta"]     = data2.get("meta")                     # 행별 (gid,src,fid,idx)
        else:
            query = ctx["centroid"]                                 # 1D 시드 인덱스
        c = self.compute_anchor_patch_context(query, feat, **opts)
        ctx["sim"], ctx["heat"] = c["sim_matrix"], c["heatmap_sim"]
        ctx["vom"], ctx["pure"], ctx["oc"] = c["vom"], c["pure"], c["oc"]
        ctx["_H"] = c.get("H_matrix")
        if not is_memory(data2):
            ctx["centroid"] = c.get("centroids", ctx["centroid"])
        return ctx

    # ── STAGE: 객체 벡터 (avg/patch · weight · overlap 토글) → ctx['vec'] ──
    def ctx_object_vectors(self, data1, ctx, repr="avg",
                           weight="uniform", overlap="exclude"):
        # 현재 프레임 대표 패치: 단일=시드(centroid), 크로스=ctx['rep'](pure 응답 피크)
        sample = ctx.get("rep", ctx["centroid"])
        ctx["vec"] = self.compute_object_vectors(
            data1["feat"], sample, ctx["vom"], attn=data1.get("attn"),
            repr=repr, weight=weight, overlap=overlap)
        return ctx

    # ── STAGE: 히트맵 기반 그룹 병합 (frame 단일 경로) → 디퍼 필드 ──
    def ctx_group_anchors(self, data1, ctx, th_heatmap=0.85):
        legacy = self._to_legacy_ctx(ctx, ctx["centroid"], data1["feat"].shape[0])
        labels, n_comp = self.merge_anchors_heatmap(legacy, th_heatmap=th_heatmap)
        ctx["_group_labels"], ctx["_n_comp"] = labels, n_comp
        return ctx

    # ── STAGE: 그룹 후처리(대표 시드 + 재계산 + P3) → ctx 갱신 ──
    #   _apply_group_and_recompute 는 병합 후 앵커 집합(M)이 바뀌므로 ctx 를
    #   내부에서 재계산한다(merge_to_ctx). 그 결과로 ctx 를 덮어쓴다.
    def ctx_compute_anchor_response(self, data1, ctx, **opts):
        nc, gvom, gpure, avg, new_ctx = self._apply_group_and_recompute(
            ctx["centroid"], ctx["_group_labels"], ctx["_n_comp"], None,
            data1["feat"], data1["x_cat"],
            attn=data1.get("attn"), grid_shape=data1.get("grid_shape"), **opts)
        ctx["centroid"] = nc
        ctx["vom"], ctx["pure"] = gvom, gpure
        ctx["sim"]  = new_ctx.get("sim_matrix")
        ctx["heat"] = new_ctx.get("heatmap_sim")
        ctx["oc"]   = new_ctx.get("oc")
        ctx["_H"]   = new_ctx.get("H_matrix")
        ctx["vec"]  = avg          # 기본 avg. patch/weight/overlap 토글은 ctx_object_vectors 로 덮어쓰기
        return ctx

    # ── STAGE: 품질 필터 → ctx['keep'] (+ reduce 시 ctx 앵커 차원 축소) ──
    def ctx_filter_anchors(self, data1, ctx, reduce=True, **opts):
        keep = self.filter_by_quality(ctx["vom"], ctx["pure"],
                                      grid_shape=data1.get("grid_shape"), **opts)
        ctx["keep"] = keep
        if reduce and keep is not None and keep.any() and not keep.all():
            self._reduce_ctx_inplace(ctx, keep)
        return ctx

    @staticmethod
    def _reduce_ctx_inplace(ctx, keep):
        keep = keep.bool()
        for k in ("vom", "pure", "sim", "vec", "centroid", "_H"):
            v = ctx.get(k)
            if isinstance(v, torch.Tensor) and v.shape[:1] == keep.shape:
                ctx[k] = v[keep]
        h = ctx.get("heat")
        if isinstance(h, torch.Tensor) and h.shape[:1] == keep.shape:
            ctx["heat"] = h[keep][:, keep]
        if isinstance(ctx.get("vom"), torch.Tensor):
            ctx["oc"] = ctx["vom"].sum(dim=0)
        m = ctx.get("meta")
        if isinstance(m, list) and len(m) == keep.shape[0]:
            ctx["meta"] = [mm for mm, on in zip(m, keep.tolist()) if on]

    # ── STAGE(재가공): multiresponse → (I_n, overlap, clean_patch_mask) ──
    def ctx_detect_multiresponse(self, data1, ctx, th_sim=0.60, th_margin=0.12):
        sim = torch.mm(ctx["vec"], data1["feat"].t())
        I_n, overlap = self.detect_mixed_boundary_patches_by_counting(
            sim, th_sim=th_sim, th_margin=th_margin)
        return I_n, overlap, (I_n < 1)

    # ──────────────────────────────────────────────────────────
    # 메모리 data 빌더 (dino_mgr 저장분 → type="memory" + meta)
    #   dino_mgr.get(src,fid) 저장 슬롯 가정:
    #     (x_cat, sample, vom, avg_vec, bind_xfeat, gids)   ← gids[K] 슬롯 추가 필요
    # ──────────────────────────────────────────────────────────
    def build_memory_data(self, dino_mgr, neigh_keys, repr="avg", max_frames=5):
        vecs, meta, cnt = [], [], 0
        for (nsrc, nfid) in neigh_keys:
            if cnt >= max_frames:
                break
            try:
                rec = dino_mgr.get(nsrc, nfid)
            except Exception:
                continue
            avg  = rec[3]
            gids = rec[5] if len(rec) > 5 else None
            if avg is None or avg.shape[0] == 0:
                continue
            if repr == "patch":
                feat, _, _ = self._prepare_features(rec[0])
                vec = F.normalize(feat[rec[1]], p=2, dim=1)
            else:
                vec = avg
            for k in range(vec.shape[0]):
                g = int(gids[k]) if (gids is not None and k < len(gids)) else -1
                meta.append({"gid": g, "src": nsrc, "fid": nfid, "idx": k})
            vecs.append(vec.cpu())
            cnt += 1
        if not vecs:
            return None
        return make_memory_data(torch.cat(vecs, dim=0), meta=meta)

    # ── dedup 선택 (메모리 data 의 feat/meta 동시 축소) ──
    def select_memory(self, mem_data, sim_thresh=0.90, max_k=64):
        if mem_data is None or mem_data["feat"].shape[0] == 0:
            return mem_data
        sel, keep_idx = self.select_object_vectors(mem_data["feat"],
                                                   sim_thresh=sim_thresh, max_k=max_k)
        meta = mem_data.get("meta")
        new_meta = ([meta[i] for i in keep_idx.tolist()]
                    if isinstance(meta, list) else None)
        return make_memory_data(sel, meta=new_meta)

    # ── 디스패치 alias (set_stage_impl 로 갈아끼우려면 이 이름으로 호출) ──
    #   compute_anchor_response/filter_anchors/group_anchors/generate_seeds/
    #   detect_multiresponse 는 patchcluster_v2 에 이미 alias 존재.
    #   compute_context/object_vectors 는 ctx 파이프라인 전용이라 여기서 추가.
    def compute_context(self, *args, method=None, **kwargs):
        return self._run_stage("compute_context", *args, method=method, **kwargs)

    def object_vectors(self, *args, method=None, **kwargs):
        return self._run_stage("object_vectors", *args, method=method, **kwargs)

    # ── 디스패치를 ctx 어댑터로 매핑. 서버는 alias 이름으로 호출 →
    #    set_stage_impl 로 각 단계 구현을 갈아끼울 수 있다. ──
    def use_ctx_stages(self):
        self.stage_impl.update({
            "generate_seeds":          "ctx_generate_seeds",
            "compute_context":         "ctx_compute_context",
            "group_anchors":           "ctx_group_anchors",
            "compute_anchor_response": "ctx_compute_anchor_response",
            "object_vectors":          "ctx_object_vectors",
            "filter_anchors":          "ctx_filter_anchors",
            "detect_multiresponse":    "ctx_detect_multiresponse",
        })

"""
pipeline_debug_visualizer.py  -- OpenCV only
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
import torch

try:
    from anchor_debug_visualizer import (make_distinct_colors, patch_to_pixel, draw_patch_mask)
except ImportError:
    from dino.anchor_debug_visualizer import (make_distinct_colors, patch_to_pixel, draw_patch_mask)

# ── helpers ──────────────────────────────────────────────────────

def _pad_h(img, target_h):
    h, w = img.shape[:2]
    if h >= target_h: return img
    return np.vstack([img, np.zeros((target_h-h, w, 3), dtype=np.uint8)])

def _pad_w(img, target_w):
    h, w = img.shape[:2]
    if w >= target_w: return img
    return np.hstack([img, np.zeros((h, target_w-w, 3), dtype=np.uint8)])

def _hstack(*imgs):
    if not imgs: return np.zeros((100,100,3),dtype=np.uint8)
    mh = max(im.shape[0] for im in imgs)
    return np.hstack([_pad_h(im, mh) for im in imgs])

def _vstack(*imgs):
    if not imgs: return np.zeros((100,100,3),dtype=np.uint8)
    mw = max(im.shape[1] for im in imgs)
    return np.vstack([_pad_w(im, mw) for im in imgs])

def _label(img, text, bh=22):
    bar = np.full((bh, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text[:80], (4, bh-5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210,210,210), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

def _tb(w, text, h=26):
    bar = np.full((h, w, 3), 20, dtype=np.uint8)
    cv2.putText(bar, text[:100], (6, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,60), 1, cv2.LINE_AA)
    return bar

def _save(img, save_path):
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"[viz] saved: {save_path}")

def _rh(img, h):
    oh, ow = img.shape[:2]
    return cv2.resize(img, (max(1, int(ow*h/oh)), h), interpolation=cv2.INTER_LINEAR)

def _bar(values, colors_bgr, title="", w=400, h=220):
    canvas = np.full((h, w, 3), 35, dtype=np.uint8)
    n = len(values)
    if n == 0: return canvas
    ml, mb, mt = 8, 24, 20
    pw, ph = w-ml-4, h-mb-mt
    vmax = float(max(np.max(np.abs(values)), 1e-9))
    bw = max(2, pw//n - 2)
    for i,(v,c) in enumerate(zip(values, colors_bgr)):
        bh2 = int(abs(float(v))/vmax * ph)
        x0 = ml + i*(pw//n) + 1; x1 = x0+bw
        y1 = h-mb; y0 = y1-bh2
        cv2.rectangle(canvas, (x0, max(y0,mt)), (x1,y1), c, -1)
        cv2.putText(canvas, str(i), (x0, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (170,170,170), 1)
    if title:
        cv2.putText(canvas, title[:50], (ml, mt-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
    return canvas

def _scatter(coords, pt_colors, markers=None, title="", w=360, h=360):
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    if len(coords) == 0: return canvas
    mg = 20
    x,y = coords[:,0].astype(float), coords[:,1].astype(float)
    xr = x.max()-x.min()+1e-9; yr = y.max()-y.min()+1e-9
    pw, ph = w-2*mg, h-2*mg-20
    for i,(xi,yi) in enumerate(zip(x,y)):
        px = int((xi-x.min())/xr*pw)+mg
        py = int((yi-y.min())/yr*ph)+mg+18
        is_star = markers is not None and bool(markers[i])
        c = pt_colors[i] if i < len(pt_colors) else (180,180,180)
        cv2.circle(canvas, (px,py), 5 if is_star else 3, c, -1, cv2.LINE_AA)
        if is_star:
            cv2.circle(canvas, (px,py), 7, (255,255,255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title[:40], (mg,14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
    return canvas

def _centroid(vom_row, grid_shape, patch_size):
    H_p, W_p = grid_shape
    idx = np.where(vom_row)[0]
    if len(idx) == 0: return W_p*patch_size//2, H_p*patch_size//2
    cy = int((( idx//W_p).mean()+0.5)*patch_size)
    cx = int(((idx % W_p).mean()+0.5)*patch_size)
    return cx, cy

# ── Phase 1 ──────────────────────────────────────────────────────

def phase1_attention(img_bgr, attn, grid_shape, patch_size=14, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    an = attn.cpu().float().numpy().reshape(H_p, W_p)
    an = (an-an.min())/(an.max()-an.min()+1e-8)
    heat = cv2.applyColorMap((cv2.resize(an,(W,H))*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(img_bgr, 0.45, heat, 0.55, 0)
    grid_big = cv2.resize(cv2.applyColorMap((an*255).astype(np.uint8), cv2.COLORMAP_INFERNO), (W,H), interpolation=cv2.INTER_NEAREST)
    panels = _hstack(_label(img_bgr.copy(),"Original"), _label(blended,"CLS Attn"), _label(grid_big,f"Grid [{H_p}x{W_p}]"))
    _save(_vstack(_tb(panels.shape[1],"PHASE 1 -- CLS Attention"), panels), save_path)

def phase1_xfeat_patches(img_bgr, kp, bind_mat, grid_shape, patch_size=14, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    kp_np = kp.cpu().numpy()
    bind_np = bind_mat.cpu().numpy()
    kpp = bind_np.sum(axis=1)
    canvas = img_bgr.copy().astype(float)
    mx = max(kpp.max(), 1)
    for n in range(H_p*W_p):
        r,c = n//W_p, n%W_p
        y0,x0 = r*patch_size, c*patch_size
        y1,x1 = min(y0+patch_size,H), min(x0+patch_size,W)
        t = kpp[n]/mx
        canvas[y0:y1,x0:x1] = canvas[y0:y1,x0:x1]*0.5 + np.array([0,t*180,t*80])*0.5
    canvas = canvas.clip(0,255).astype(np.uint8)
    for i in range(len(kp_np)):
        cv2.circle(canvas, (int(kp_np[i,0]),int(kp_np[i,1])), 2, (0,200,255), -1)
    N = H_p*W_p; step = max(1,N//60)
    bar_img = _bar(kpp[::step], [(0,int(v/mx*180),int(v/mx*80)) for v in kpp[::step]],
                   title=f"kp/patch N={N} M={len(kp_np)}", w=W, h=H//2)
    out = _hstack(_label(canvas,f"XFeat kp={len(kp_np)}"), _label(_pad_h(bar_img,H),"density"))
    _save(_vstack(_tb(out.shape[1],"PHASE 1 -- XFeat Binding"), out), save_path)

# ── Phase 2 ──────────────────────────────────────────────────────

def phase2_memory_pool(pool_vecs, source_labels=None, selected_mask=None, save_path=None):
    import torch.nn.functional as F
    M = pool_vecs.shape[0]
    if M == 0: return
    vecs_np = F.normalize(pool_vecs, p=2, dim=1).cpu().numpy()
    sel_np = selected_mask.cpu().numpy() if selected_mask is not None else None
    n_src = len(set(source_labels)) if source_labels else 1
    palette = make_distinct_colors(max(n_src,1))
    W_sc, H_sc = 380, 380
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(vecs_np) if M >= 3 else np.zeros((M,2))
        pt_cols = [(int(palette[source_labels[i]%len(palette)][2]),
                    int(palette[source_labels[i]%len(palette)][1]),
                    int(palette[source_labels[i]%len(palette)][0])) if source_labels else (180,180,180)
                   for i in range(M)]
        sc = _scatter(coords, pt_cols, markers=sel_np, title="Pool PCA (*=selected)", w=W_sc, h=H_sc)
    except ImportError:
        sc = np.full((H_sc,W_sc,3), 40, dtype=np.uint8)
        cv2.putText(sc,"sklearn unavailable",(10,H_sc//2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    info = np.full((H_sc,200,3),35,dtype=np.uint8)
    for i,txt in enumerate([f"Pool: {M}", f"Selected: {int(sel_np.sum()) if sel_np is not None else M}",
                             f"Sources: {n_src}", f"Dim: {pool_vecs.shape[1]}"]):
        cv2.putText(info, txt, (8,30+i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,210,220), 1)
    panels = _hstack(sc, info)
    _save(_vstack(_tb(panels.shape[1],"PHASE 2 -- Memory Pool PCA"), panels), save_path)

def phase2_neighbor_frames(imgs_bgr, sims, selected_idx=None, save_path=None):
    if not imgs_bgr: return
    panels = []
    for i,(im,sim) in enumerate(zip(imgs_bgr, sims)):
        p = _rh(im, 180)
        is_sel = selected_idx is not None and i in selected_idx
        p = _label(p, f"{'[SEL] ' if is_sel else ''}sim={sim:.2f}")
        if is_sel: cv2.rectangle(p,(0,0),(p.shape[1]-1,p.shape[0]-1),(0,255,80),2)
        panels.append(p)
    out = _hstack(*panels)
    _save(_vstack(_tb(out.shape[1],"PHASE 2 -- Neighbor Frames"), out), save_path)

# ── Phase 3 ──────────────────────────────────────────────────────

def phase3_anchors(img_bgr, sample, vom, pure, grid_shape, patch_size=14, title_suffix="", save_path=None):
    K = vom.shape[0]
    vom_np = vom.cpu().bool().numpy()
    pure_np = pure.cpu().bool().numpy()
    palette = make_distinct_colors(max(K,1))
    cv_vom = img_bgr.copy(); cv_pure = img_bgr.copy()
    for k in range(K):
        cv_vom  = draw_patch_mask(cv_vom,  np.where(vom_np[k])[0],  grid_shape, patch_size, color=palette[k], alpha=0.40)
        cv_pure = draw_patch_mask(cv_pure, np.where(pure_np[k])[0], grid_shape, patch_size, color=palette[k], alpha=0.50)
        cx,cy = _centroid(vom_np[k], grid_shape, patch_size)
        cv2.circle(cv_vom,(cx,cy),5,palette[k],-1)
        cv2.putText(cv_vom,str(k),(cx+4,cy-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    panels = _hstack(_label(cv_vom,f"VOM K={K}"), _label(cv_pure,"Pure (oc==1)"))
    _save(_vstack(_tb(panels.shape[1],f"PHASE 3 -- Anchors {title_suffix}"), panels), save_path)

def phase3_cross_frame(img_bgr, vom, pure, valid_mask, grid_shape, patch_size=14, save_path=None):
    K = vom.shape[0]
    vom_np = vom.cpu().bool().numpy()
    valid_np = valid_mask.cpu().bool().numpy()
    palette = make_distinct_colors(max(K,1))
    cv_ok = img_bgr.copy(); cv_bad = img_bgr.copy()
    for k in range(K):
        pts = np.where(vom_np[k])[0]
        cx,cy = _centroid(vom_np[k], grid_shape, patch_size)
        if valid_np[k]:
            cv_ok = draw_patch_mask(cv_ok, pts, grid_shape, patch_size, color=palette[k], alpha=0.40)
            cv2.circle(cv_ok,(cx,cy),5,palette[k],-1)
            cv2.putText(cv_ok,str(k),(cx+4,cy-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        else:
            cv_bad = draw_patch_mask(cv_bad, pts, grid_shape, patch_size, color=(200,60,60), alpha=0.40)
            cv2.circle(cv_bad,(cx,cy),5,(200,60,60),-1)
            cv2.putText(cv_bad,str(k),(cx+4,cy-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,100),1)
    panels = _hstack(_label(cv_ok,f"Valid {int(valid_np.sum())}/{K}"), _label(cv_bad,f"Invalid {int((~valid_np).sum())}/{K}"))
    _save(_vstack(_tb(panels.shape[1],"PHASE 3 -- Cross-Frame Projection"), panels), save_path)

# ── Phase 4 ──────────────────────────────────────────────────────

def phase4_quality_filter(img_bgr, vom, pure, keep, grid_shape, patch_size=14, quality_metrics=None, save_path=None):
    K = vom.shape[0]
    vom_np = vom.cpu().bool().numpy()
    keep_np = keep.cpu().bool().numpy()
    palette = make_distinct_colors(max(K,1))
    cv_k = img_bgr.copy(); cv_r = img_bgr.copy()
    for k in range(K):
        pts = np.where(vom_np[k])[0]
        cx,cy = _centroid(vom_np[k], grid_shape, patch_size)
        if keep_np[k]:
            cv_k = draw_patch_mask(cv_k, pts, grid_shape, patch_size, color=palette[k], alpha=0.40)
            cv2.circle(cv_k,(cx,cy),5,palette[k],-1)
            cv2.putText(cv_k,str(k),(cx+4,cy-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        else:
            cv_r = draw_patch_mask(cv_r, pts, grid_shape, patch_size, color=(220,50,50), alpha=0.40)
            cv2.circle(cv_r,(cx,cy),5,(220,50,50),-1)
            cv2.putText(cv_r,str(k),(cx+4,cy-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,100),1)
    H = img_bgr.shape[0]
    if quality_metrics:
        key = next(iter(quality_metrics))
        vals = np.asarray(quality_metrics[key], dtype=float)
        bar_colors = [(0,180,60) if keep_np[k] else (50,50,200) for k in range(K)]
        b = _bar(vals, bar_colors, title=key, w=H, h=H//2)
    else:
        pc = pure.sum(dim=1).cpu().numpy().astype(float)
        bar_colors = [(0,180,60) if keep_np[k] else (50,50,200) for k in range(K)]
        b = _bar(pc, bar_colors, title="pure count", w=H, h=H//2)
    panels = _hstack(_label(cv_k,f"Keep {int(keep_np.sum())}/{K}"), _label(cv_r,f"Remove {int((~keep_np).sum())}/{K}"), _label(_pad_h(b,H),"metric"))
    _save(_vstack(_tb(panels.shape[1],"PHASE 4 -- Quality Filter"), panels), save_path)

def phase4_multiresponse(img_bgr, I_n, grid_shape, patch_size=14, clean_patch_mask=None, save_path=None):
    H_p, W_p = grid_shape
    H, W = img_bgr.shape[:2]
    I_np = I_n.cpu().numpy()
    mx = int(I_np.max()) if len(I_np) > 0 else 0
    canvas = img_bgr.copy().astype(float)
    for n in range(H_p*W_p):
        r,c = n//W_p, n%W_p
        y0,x0 = r*patch_size, c*patch_size
        y1,x1 = min(y0+patch_size,H), min(x0+patch_size,W)
        ov = int(I_np[n]) if n < len(I_np) else 0
        if ov==0:   col = np.array([80,80,80])
        elif ov==1: col = np.array([40,210,60])
        else:
            t = min((ov-1)/max(mx-1,1),1.0)
            col = np.array([40,int(210*(1-t)),int(60+160*t)])
        canvas[y0:y1,x0:x1] = canvas[y0:y1,x0:x1]*0.4 + col*0.6
    canvas = canvas.clip(0,255).astype(np.uint8)
    leg = np.full((50,W,3),35,dtype=np.uint8)
    for i,(col,txt) in enumerate([((80,80,80),"0=none"),((40,210,60),"1=clean"),((40,60,200),">=2=multi")]):
        x = 10+i*(W//3)
        cv2.rectangle(leg,(x,8),(x+16,26),col,-1)
        cv2.putText(leg,txt,(x+20,22),cv2.FONT_HERSHEY_SIMPLEX,0.38,(210,210,210),1)
    _save(_vstack(_tb(W,f"PHASE 4 -- Multi-Response  max={mx}"), canvas, leg), save_path)

def phase4_patch_ids(img_bgr, patch_gid_map, grid_shape, patch_size=14, save_path=None):
    H_p, W_p = grid_shape; H,W = img_bgr.shape[:2]
    gid_np = patch_gid_map.cpu().numpy()
    unique = np.unique(gid_np[gid_np>=0])
    palette = make_distinct_colors(max(len(unique),1))
    g2c = {int(g): palette[i%len(palette)] for i,g in enumerate(unique)}
    canvas = img_bgr.copy().astype(float)
    for n in range(H_p*W_p):
        r,c = n//W_p, n%W_p
        y0,x0 = r*patch_size, c*patch_size
        y1,x1 = min(y0+patch_size,H), min(x0+patch_size,W)
        g = int(gid_np[n]) if n < len(gid_np) else -1
        if g >= 0:
            col = np.array(g2c.get(g,(128,128,128)),dtype=float)
            canvas[y0:y1,x0:x1] = canvas[y0:y1,x0:x1]*0.4 + col*0.6
    canvas = canvas.clip(0,255).astype(np.uint8)
    _save(_vstack(_tb(W,f"PHASE 4 -- Patch GID ({len(unique)} groups)"), canvas), save_path)

def phase4_anchor_xfeat(img_bgr, mat_xfeat, kp, grid_shape, patch_size=14, max_anchors=8, save_path=None):
    K = min(mat_xfeat.shape[0], max_anchors)
    mat_np = mat_xfeat.cpu().numpy(); kp_np = kp.cpu().numpy()
    palette = make_distinct_colors(max(K,1))
    canvas = img_bgr.copy()
    for k in range(K):
        for j in np.where(mat_np[k] > 0.01)[0]:
            if j < len(kp_np):
                cv2.circle(canvas,(int(kp_np[j,0]),int(kp_np[j,1])),3,palette[k],-1)
    _save(_vstack(_tb(canvas.shape[1],f"PHASE 4 -- Anchor XFeat K={K}"), canvas), save_path)

# ── Phase 5 ──────────────────────────────────────────────────────

def phase5_bank_state(obj_bank, highlight_gids=None, save_path=None):
    try: entries = list(obj_bank.bank.values())
    except: entries = []
    if not entries:
        img = np.full((200,400,3),35,dtype=np.uint8)
        cv2.putText(img,"bank empty",(10,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
        _save(img, save_path); return
    gids  = [getattr(e,'gid',i) for i,e in enumerate(entries)]
    stabs = np.array([getattr(e,'stability',0.0) for e in entries],dtype=float)
    pemas = np.array([getattr(e,'pure_area_ema',0.0) for e in entries],dtype=float)
    sstds = np.array([getattr(e,'spatial_std_ema',0.0) for e in entries],dtype=float)
    palette = make_distinct_colors(max(len(entries),1))
    hl = set(highlight_gids or [])
    bc = [(0,255,255) if g in hl else (int(palette[i][2]),int(palette[i][1]),int(palette[i][0]))
          for i,g in enumerate(gids)]
    b1 = _bar(stabs, bc, title="stability",      w=300, h=220)
    b2 = _bar(pemas, bc, title="pure_area_ema",  w=300, h=220)
    b3 = _bar(sstds, bc, title="spatial_std_ema",w=300, h=220)
    bars = _hstack(b1,b2,b3)
    _save(_vstack(_tb(bars.shape[1],f"PHASE 5 -- Bank State  size={len(entries)}"), bars), save_path)

def phase5_bank_register(img_bgr, gids, vom, grid_shape, patch_size=14, is_new=None, save_path=None):
    K = vom.shape[0]; vom_np = vom.cpu().bool().numpy()
    palette = make_distinct_colors(max(K,1))
    canvas = img_bgr.copy()
    for k in range(K):
        canvas = draw_patch_mask(canvas, np.where(vom_np[k])[0], grid_shape, patch_size, color=palette[k], alpha=0.40)
        cx,cy = _centroid(vom_np[k], grid_shape, patch_size)
        new_flag = is_new[k] if is_new else False
        cv2.circle(canvas,(cx,cy),6,(0,255,80) if new_flag else (0,180,255),-1)
        cv2.putText(canvas,f"g{gids[k]}{'*' if new_flag else ''}",(cx+5,cy-5),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
    _save(_vstack(_tb(canvas.shape[1],f"PHASE 5 -- Bank Register K={K} (*=new)"), canvas), save_path)

# ── Phase 6 ──────────────────────────────────────────────────────

def phase6_cross_match(img1_bgr, img2_bgr, match_mask, match12, kp1, kp2,
                        anchor_labels1=None, max_lines=200, save_path=None):
    H1,W1 = img1_bgr.shape[:2]; H2,W2 = img2_bgr.shape[:2]
    th = max(H1,H2)
    i1 = _rh(img1_bgr, th); i2 = _rh(img2_bgr, th)
    W1r, W2r = i1.shape[1], i2.shape[1]
    canvas = np.hstack([i1,i2])
    mask_np = match_mask.cpu().bool().numpy()
    m12_np  = match12.cpu().numpy().flatten()
    kp1_np  = kp1.cpu().numpy(); kp2_np = kp2.cpu().numpy()
    al_np   = anchor_labels1.cpu().numpy() if anchor_labels1 is not None else None
    K_anc   = int(al_np.max())+1 if al_np is not None else 1
    palette = make_distinct_colors(max(K_anc,1))
    vidx = np.where(mask_np)[0]
    if len(vidx) > max_lines:
        vidx = vidx[np.linspace(0,len(vidx)-1,max_lines,dtype=int)]
    s1x,s1y = W1r/W1, th/H1; s2x,s2y = W2r/W2, th/H2
    for i in vidx:
        if i >= len(kp1_np): continue
        j = int(m12_np[i])
        if j >= len(kp2_np): continue
        x1b,y1b = int(kp1_np[i,0]*s1x), int(kp1_np[i,1]*s1y)
        x2b,y2b = int(kp2_np[j,0]*s2x)+W1r, int(kp2_np[j,1]*s2y)
        k_idx = int(al_np[i]) if al_np is not None and i < len(al_np) else 0
        col = palette[k_idx%len(palette)]
        cv2.line(canvas,(x1b,y1b),(x2b,y2b),col,1,cv2.LINE_AA)
        cv2.circle(canvas,(x1b,y1b),3,col,-1); cv2.circle(canvas,(x2b,y2b),3,col,-1)
    cv2.line(canvas,(W1r,0),(W1r,th),(200,200,200),1)
    _save(_vstack(_tb(canvas.shape[1],f"PHASE 6 -- Cross-View Match  matches={int(mask_np.sum())}"), canvas), save_path)

# ── Controller ───────────────────────────────────────────────────

class PipelineDebugVisualizer:
    def __init__(self, output_dir='./debug', enable_phases=None, patch_size=14, frame_id=None):
        self.output_dir    = output_dir
        self.enable_phases = enable_phases
        self.patch_size    = patch_size
        self.frame_id      = frame_id
        self._counter      = 0
        os.makedirs(output_dir, exist_ok=True)

    def set_frame_id(self, frame_id):
        self.frame_id = frame_id
        self._counter += 1

    def _enabled(self, phase):
        return self.enable_phases is None or phase in self.enable_phases

    def _path(self, name):
        fid = self.frame_id or "noframe"
        return os.path.join(self.output_dir, f"{self._counter:06d}_{fid}_{name}.png")

    def phase1_attention(self, img, attn, grid_shape):
        if self._enabled(1): phase1_attention(img, attn, grid_shape, self.patch_size, self._path("p1_attention"))
    def phase1_xfeat_patches(self, img, kp, bind_mat, grid_shape):
        if self._enabled(1): phase1_xfeat_patches(img, kp, bind_mat, grid_shape, self.patch_size, self._path("p1_xfeat"))
    def phase2_memory_pool(self, pool_vecs, source_labels=None, selected_mask=None):
        if self._enabled(2): phase2_memory_pool(pool_vecs, source_labels, selected_mask, self._path("p2_pool"))
    def phase2_neighbor_frames(self, imgs_bgr, sims, selected_idx=None):
        if self._enabled(2): phase2_neighbor_frames(imgs_bgr, sims, selected_idx, self._path("p2_neighbors"))
    def phase3_anchors(self, img, sample, vom, pure, grid_shape, suffix=""):
        if self._enabled(3): phase3_anchors(img, sample, vom, pure, grid_shape, self.patch_size, suffix, self._path(f"p3_anchors_{suffix}" if suffix else "p3_anchors"))
    def phase3_cross_frame(self, img, vom, pure, valid_mask, grid_shape):
        if self._enabled(3): phase3_cross_frame(img, vom, pure, valid_mask, grid_shape, self.patch_size, self._path("p3_cross"))
    def phase4_quality_filter(self, img, vom, pure, keep, grid_shape, quality_metrics=None):
        if self._enabled(4): phase4_quality_filter(img, vom, pure, keep, grid_shape, self.patch_size, quality_metrics, self._path("p4_quality"))
    def phase4_multiresponse(self, img, I_n, grid_shape, clean_patch_mask=None):
        if self._enabled(4): phase4_multiresponse(img, I_n, grid_shape, self.patch_size, clean_patch_mask, self._path("p4_multiresponse"))
    def phase4_patch_ids(self, img, patch_gid_map, grid_shape):
        if self._enabled(4): phase4_patch_ids(img, patch_gid_map, grid_shape, self.patch_size, self._path("p4_patchids"))
    def phase4_anchor_xfeat(self, img, mat_xfeat, kp, grid_shape, max_anchors=8):
        if self._enabled(4): phase4_anchor_xfeat(img, mat_xfeat, kp, grid_shape, self.patch_size, max_anchors, self._path("p4_xfeat"))
    def phase5_bank_state(self, obj_bank, highlight_gids=None):
        if self._enabled(5): phase5_bank_state(obj_bank, highlight_gids, self._path("p5_bank"))
    def phase5_bank_register(self, img, gids, vom, grid_shape, is_new=None):
        if self._enabled(5): phase5_bank_register(img, gids, vom, grid_shape, self.patch_size, is_new, self._path("p5_register"))
    def phase6_cross_match(self, img1, img2, match_mask, match12, kp1, kp2, anchor_labels1=None, max_lines=200):
        if self._enabled(6): phase6_cross_match(img1, img2, match_mask, match12, kp1, kp2, anchor_labels1, max_lines, self._path("p6_match"))

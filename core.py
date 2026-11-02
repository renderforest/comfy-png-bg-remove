"""My bg remove - core (numpy / scipy / PIL), testable without ComfyUI."""
import numpy as np
from PIL import Image

try:
    from scipy import ndimage as _ndi
except Exception:  # scipy ships with ComfyUI; fallbacks keep it working anyway
    _ndi = None


# ---------------------------------------------------------------- helpers
def _grow(a, r):
    r = int(round(r))
    if r <= 0:
        return a
    a = np.ascontiguousarray(a, dtype=np.float32)
    if _ndi is not None:
        return _ndi.maximum_filter(a, size=2 * r + 1, mode="nearest")
    import torch
    t = torch.from_numpy(a)[None, None]
    t = torch.nn.functional.pad(t, (r, r, r, r), mode="replicate")
    return torch.nn.functional.max_pool2d(t, 2 * r + 1, stride=1)[0, 0].numpy()


def _shrink(a, r):
    return 1.0 - _grow(1.0 - np.asarray(a, np.float32), r)


def _blur(a, sigma):
    a = np.asarray(a, np.float32)
    if sigma <= 0:
        return a
    if _ndi is not None:
        if a.ndim == 3:
            return np.stack([_ndi.gaussian_filter(a[..., c], sigma, mode="nearest", truncate=3.0)
                             for c in range(a.shape[2])], -1)
        return _ndi.gaussian_filter(a, sigma, mode="nearest", truncate=3.0)
    from PIL import ImageFilter
    def one(x):
        lo, hi = float(x.min()), float(x.max())
        sc = (hi - lo) or 1.0
        im = Image.fromarray(((x - lo) / sc * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(sigma))
        return np.asarray(im).astype(np.float32) / 255 * sc + lo
    return np.stack([one(a[..., c]) for c in range(a.shape[2])], -1) if a.ndim == 3 else one(a)


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _resize(a, size_wh, nearest=False):
    im = Image.fromarray(np.asarray(a, np.float32), "F")
    return np.asarray(im.resize(size_wh, Image.NEAREST if nearest else Image.BOX if size_wh[0] < a.shape[1]
                                else Image.BILINEAR), np.float32)


# ---------------------------------------------------------------- background
def detect_background(img, border_pct=1.5, uniform_tol=30.0):
    """img HxWx3 float 0-255 -> (bg color, fraction of border that matches, noise level)."""
    h, w, _ = img.shape
    b = max(2, int(round(min(h, w) * border_pct / 100)))
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                             img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    if border.shape[0] > 400000:
        border = border[:: border.shape[0] // 400000 + 1]
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(border - bg, axis=1)
    frac = float((dist <= uniform_tol).mean())
    close = dist[dist <= uniform_tol]
    noise = float(np.percentile(close, 99)) if close.size else 10.0
    return bg.astype(np.float32), frac, noise


def bg_distance(img, bg, remove_shadows, shadow_depth, sc=1.0):
    d = np.sqrt(((img - bg) ** 2).sum(-1))
    if remove_shadows:
        lum = img.mean(-1)
        bl = max(float(bg.mean()), 1.0)
        if bl > 40:  # shadows only make sense on a background that is not almost black
            ratio = lum / bl
            sm = _blur(img, 1.5 * sc)            # judge the tint on a smoothed copy (no noise)
            cd = np.sqrt(((sm / (sm.mean(-1)[..., None] + 1e-3) - bg / bl) ** 2).sum(-1))
            smooth = np.sqrt(((img - sm) ** 2).sum(-1)) < np.maximum(14.0, 0.25 * d)   # no fine detail (lines, text)
            shadow = ((ratio >= 1.0 - shadow_depth) & (ratio <= 1.08) & (cd < 0.08) & smooth).astype(np.float32)
            # a floor shadow is a big soft area; thin lines with a similar tint are NOT shadows
            r = 4 * sc
            shadow = _grow(_shrink(shadow, r), r) > 0.5
            d = np.where(shadow, 0.0, d)
    return d.astype(np.float32)


def matte(img, bg, d, tolerance, softness, sc):
    """Alpha + local foreground color.
    Inside the logo and in the open background a soft color key is used.
    Along edges the alpha is measured against the real foreground color next to the edge
    (projection on the line background -> foreground). This removes the dark/light halos
    that upscalers (RTX, ESRGAN, Lanczos sharpening) draw around shapes, and gives clean
    semi-transparent edges."""
    a0 = _smoothstep(tolerance, tolerance + softness, d)
    core_thr = max(3.0 * tolerance, 60.0)
    # core = clearly logo AND not a blended edge pixel (an edge pixel is weaker than its neighbours)
    dmax = _grow(d, round(2 * sc))
    core = ((d > core_thr) & (d >= 0.8 * dmax)).astype(np.float32)
    s = 1.5 * sc
    wc = _blur(core, s)
    F = _blur(img * core[..., None], s) / np.maximum(wc, 1e-4)[..., None]
    v = F - bg
    vv = (v * v).sum(-1)
    proj = ((img - bg) * v).sum(-1) / np.maximum(vv, 1e-3)
    # only pixels next to the background are edge pixels; color borders inside the logo stay untouched
    near_bg = _grow((d < tolerance).astype(np.float32), round(3 * sc)) > 0.5
    use = near_bg & (wc > 0.01) & (vv > core_thr * core_thr)
    alpha = np.where(use, np.minimum(a0, np.clip(proj, 0, 1)), a0).astype(np.float32)
    # Halos are thin. A big area that the edge rule wanted to delete is a real design part
    # (e.g. a black window on a dark green background) -> give it back.
    killed = ((a0 - alpha) > 0.5).astype(np.float32)
    r = 2.5 * sc
    thick = _grow(_shrink(killed, r), r + 1) > 0.5
    alpha = np.where(thick, a0, alpha).astype(np.float32)
    has_f = wc > 0.01
    return alpha, F, has_f


# ---------------------------------------------------------------- glass / one-tone logos
def one_tone_check(img, bg, d, tolerance):
    """Is the logo basically ONE color tone on the background (white logo on green, cream logo on navy,
    black logo on white)? Then every pixel is "logo color mixed with background" and the cleanest matte
    is a straight unmix: perfect anti-aliasing, glows and glass stay see-through, upscaler halos vanish."""
    H, W = d.shape
    step = max(1, int(np.ceil(max(H, W) / 800)))
    im, ds = img[::step, ::step], d[::step, ::step]
    strong = ds > max(3.0 * tolerance, 60.0)
    if strong.sum() < 50:
        return None, 0.0
    top = im[ds >= np.percentile(ds[strong], 80)]
    fg = np.median(top, axis=0).astype(np.float32)
    v = fg - bg
    vv = float((v * v).sum())
    if vv < 1e-3:
        return None, 0.0
    diff = im[strong] - bg
    p = diff @ v / vv
    perp = np.sqrt(((diff - p[:, None] * v) ** 2).sum(-1))
    online = perp <= max(12.0, 0.12 * np.sqrt(vv))
    return fg, float(online.mean())


def glass_matte(img, bg, fg, tolerance, base_alpha):
    v = fg - bg
    vv = float((v * v).sum())
    diff = img - bg
    p = (diff @ v) / vv
    perp = np.sqrt(((diff - p[..., None] * v) ** 2).sum(-1))
    p0 = tolerance / np.sqrt(vv)                     # background noise level in "alpha units"
    a = np.clip(p, 0, 1) * _smoothstep(0.5 * p0, 1.5 * p0, p)
    off = perp > max(12.0, 0.12 * np.sqrt(vv))       # a pixel of another color: keep its normal key alpha
    return np.where(off, np.maximum(a, base_alpha), a).astype(np.float32)


# ---------------------------------------------------------------- inside areas
def handle_inside(alpha, inside, hole_size_pct, d=None, tol=20.0, work=1600):
    """Background-colored areas that are ENCLOSED by the logo (not touching the image border).
    auto: removes letter holes / gaps between lines; keeps highlights and small light details
          that sit deep inside a solid shape.
    outside only: keeps every enclosed area.  inside too: removes every enclosed area."""
    if inside == "inside too" or _ndi is None:
        return alpha, 0, 0
    H, W = alpha.shape
    f = max(1, int(np.ceil(max(H, W) / work)))
    small = _resize(alpha, (max(1, W // f), max(1, H // f))) if f > 1 else alpha
    d_s = None
    if d is not None:
        d_s = _resize(d, (small.shape[1], small.shape[0])) if f > 1 else d
    h, w = small.shape
    bgm = small < 0.5
    lab, n = _ndi.label(bgm)
    if n == 0:
        return alpha, 0, 0
    edge_ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge_ids = edge_ids[edge_ids > 0]
    is_edge = np.zeros(n + 1, bool)
    is_edge[edge_ids] = True
    enclosed = ~is_edge
    enclosed[0] = False
    if not enclosed[1:].any():
        return alpha, 0, 0

    keep = np.zeros(n + 1, bool)
    if inside == "outside only":
        keep = enclosed.copy()
    else:
        md = min(h, w)
        r_big = max(2.0, md * hole_size_pct / 100.0)
        r_min = max(1.5, md * 0.003)
        ring_r = max(3.0, md * 0.015)
        ids = np.arange(n + 1)
        edt = _ndi.distance_transform_edt(np.pad(bgm, 1))[1:-1, 1:-1]
        r_in = np.asarray(_ndi.maximum(edt, lab, ids), np.float32)
        area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float32)
        elong = area / (np.pi * r_in * r_in + 1e-6)
        # ring around every enclosed area: how much of it is background (gap between lines)
        enc_mask = enclosed[lab]
        dist, (iy, ix) = _ndi.distance_transform_edt(~enc_mask, return_indices=True)
        ring = (dist > 0) & (dist <= ring_r)
        owner = lab[iy, ix]
        tot = np.bincount(owner[ring], minlength=n + 1).astype(np.float32)
        bgc = np.bincount(owner[ring & bgm], minlength=n + 1).astype(np.float32)
        ring_bg = bgc / np.maximum(tot, 1)
        # how strong is the color around it? a highlight sits in a light / low-contrast fill,
        # a letter hole or window sits in a strong, high-contrast shape
        if d_s is not None:
            # average strength of the LOGO pixels around it (background / halo pixels ignored)
            rl = ring & (d_s > max(2.0 * tol, 30.0))
            dsum = np.bincount(owner[rl], weights=d_s[rl], minlength=n + 1)
            ring_d = dsum / np.maximum(np.bincount(owner[rl], minlength=n + 1), 1)
        else:
            ring_d = np.full(n + 1, 999.0)
        is_hole = ((ring_bg > 0.25)          # gap between lines that opens to other background
                   | (r_in >= r_big)         # big hole (inside of O, R, window)
                   | (ring_d >= 240.0)       # sits in a strong, high-contrast shape (letter hole)
                   | (r_in < max(2.5, md * 0.0018))  # tiny pocket / speck of paper: not a real highlight
                   | (elong > 40.0))         # very long thin slit (gap between frame lines)
        keep = enclosed & ~is_hole

    kept, removed = int(keep.sum()), int(enclosed.sum() - keep.sum())
    if kept:
        keep_s = keep[lab].astype(np.float32)
        keep_f = (_resize(keep_s, (W, H), nearest=True) if f > 1 else keep_s) > 0.5
        # map the decision back onto the exact full-resolution shapes (no blocky edges)
        bg_f = alpha < 0.5
        lab_f, n_f = _ndi.label(bg_f)
        tot_f = np.bincount(lab_f.ravel(), minlength=n_f + 1)
        hit_f = np.bincount(lab_f[keep_f & bg_f], minlength=n_f + 1)
        e = np.unique(np.concatenate([lab_f[0], lab_f[-1], lab_f[:, 0], lab_f[:, -1]]))
        sel = hit_f > 0.5 * np.maximum(tot_f, 1)
        sel[e] = False
        sel[0] = False
        if sel.any():
            region = sel[lab_f].astype(np.float32)
            rad = max(1, int(round(1.5 * max(H, W) / 1216)))
            outside = np.isin(lab_f, e[e > 0])
            fill = (_grow(region, rad) > 0.5) & ~outside   # include its soft edge
            alpha = np.where(fill, 1.0, alpha).astype(np.float32)
    return alpha, kept, removed


def remove_specks(alpha, min_area_pct):
    if _ndi is None or min_area_pct <= 0:
        return alpha
    solid = alpha > 0.15   # faint parts (light lines) count too, only real dust is removed
    lab, n = _ndi.label(solid)
    if n == 0:
        return alpha
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= max(4, alpha.size * min_area_pct / 100.0)
    keep[0] = False
    r = max(2, int(round(2 * max(alpha.shape) / 1024)))
    return alpha * _grow(keep[lab].astype(np.float32), r)


# ---------------------------------------------------------------- main
def process(img, ai_mask=None, mode="auto", tolerance=0.0, softness=0.0,
            remove_shadows=True, shadow_depth=0.35, choke=0, feather=0.5,
            despeckle=0.0005, decontaminate_edges=True, inside="auto", hole_size=2.5,
            transparency="auto"):
    """img HxWx3 float 0-255; ai_mask HxW 0-1 (white = keep) or None.
    Returns rgba HxWx4 (0-1), alpha HxW, info string."""
    img = np.asarray(img, np.float32)
    H, W = img.shape[:2]
    sc = max(1.0, max(H, W) / 1216.0)          # 1.0 for a normal 1024-1216 image, 4.0 after a 4x upscale
    bg, frac, noise = detect_background(img)
    if tolerance <= 0:
        tolerance = float(np.clip(1.5 * noise + 6, 12, 60))
    if softness <= 0:
        softness = tolerance

    use_key = mode == "color key" or (mode == "auto" and (frac >= 0.6 or ai_mask is None))
    if mode == "ai mask" and ai_mask is None:
        use_key = True

    F, has_f = None, None
    if use_key:
        d = bg_distance(img, bg, remove_shadows, shadow_depth, sc)
        alpha, F, has_f = matte(img, bg, d, tolerance, softness, sc)
        how = "color key"
        if transparency != "solid":
            fg, tone = one_tone_check(img, bg, d, tolerance)
            if fg is not None and (transparency == "glass" or tone >= 0.96):
                alpha = glass_matte(img, bg, fg, tolerance, alpha)
                F = np.broadcast_to(fg, img.shape).astype(np.float32)
                has_f = np.ones(alpha.shape, bool)
                how = f"glass / one-tone matte (tone match {tone*100:.0f}%)"
            else:
                how += f" (solid, tone match {tone*100:.0f}%)"
        if ai_mask is not None and mode == "auto" and not how.startswith("glass"):
            # where logo color ~ background color (dark navy on black) the key is unsure -> trust AI there
            partial = ((alpha > 0.03) & (alpha < 0.97)).astype(np.float32)
            unsure = _grow((_blur(partial, 3 * sc) > 0.5).astype(np.float32), round(4 * sc)) > 0.5
            ai = np.clip(np.asarray(ai_mask, np.float32), 0, 1)
            rescue = unsure & (ai > 0.5)
            if rescue.any():
                alpha = np.where(rescue, np.maximum(alpha, ai), alpha)
                how += " + AI rescue"
        # in a one-tone logo every background-colored pocket is negative space
        inside_eff = "inside too" if (how.startswith("glass") and inside == "auto") else inside
        alpha, kept, removed = handle_inside(alpha, inside_eff, hole_size, d, tolerance)
        how += f" | inside {inside}: kept {kept}, removed {removed}"
    else:
        alpha = np.clip(np.asarray(ai_mask, np.float32), 0, 1)
        how = "AI mask"

    alpha = remove_specks(alpha, despeckle)
    base_alpha = alpha.copy()
    if choke > 0:
        alpha = _shrink(alpha, choke)
    elif choke < 0:
        alpha = _grow(alpha, -choke)
    if feather > 0:
        alpha = _blur(alpha, feather)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    if use_key and decontaminate_edges:
        a = base_alpha[..., None]
        unmix = np.clip(bg + (img - bg) / np.maximum(a, 1e-3), 0, 255)
        rgb = np.where(a > 0.97, img, unmix)
        weak = (base_alpha < 0.35) & has_f            # nearly transparent edge pixel: use the logo color next to it
        rgb = np.where(weak[..., None], np.clip(F, 0, 255), rgb)
        empty = (base_alpha <= 0.0) & ~has_f
        rgb = np.where(empty[..., None], img, rgb)
    else:
        rgb = img
    rgba = np.concatenate([rgb / 255.0, alpha[..., None]], -1).astype(np.float32)
    info = (f"{how} | bg RGB {tuple(int(round(v)) for v in bg)} | border uniform {frac*100:.0f}% | "
            f"tolerance {tolerance:.0f} softness {softness:.0f} | {W}x{H}")
    if use_key and frac < 0.6:
        info += " | WARNING: background is not one solid color - connect an AI mask for photos"
    return rgba, alpha, info

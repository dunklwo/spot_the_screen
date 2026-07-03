"""
Feature extraction for "real photo" vs "photo of a screen" (recapture) detection.

All features are classic CV / signal-processing based (no deep net), chosen to
capture the physical artifacts a screen recapture introduces:

  1. Moire / periodic high-frequency energy from photographing a pixel grid
     (FFT ring-band energy + peak-to-median ratio).
  2. Local sharpness / blur statistics (recaptures are often slightly softer
     or have unnatural double-blur from lens + screen diffusion).
  3. Color-cast / gamut compression (screens can't reproduce full real-world
     dynamic range and often have a slight blue/green tint and clipped highlights).
  4. Specular highlight / glare stats (screens reflect ambient light in a
     telltale way - large, low-saturation bright blobs).
  5. Edge/border rectangularity (a bezel or screen edge inside the frame).
  6. Noise texture statistics (sensor noise from the original capture gets
     re-filtered when recaptured, changing local noise structure).

Returns a fixed-length numpy feature vector per image so a tiny classifier
(logistic regression) can be trained on ~150 photos and still generalize.
"""

import cv2
import numpy as np


def _load_gray_and_color(path, max_side=2000):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _fft_patch_stats(patch):
    s = patch.shape[0]
    patch = patch.astype(np.float32)
    patch = patch - patch.mean()
    win = np.hanning(s)
    win2d = np.outer(win, win)
    f = np.fft.fftshift(np.fft.fft2(patch * win2d))
    mag = np.log1p(np.abs(f))

    cy, cx = s // 2, s // 2
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()

    low_mask = r_norm < 0.15
    mid_mask = (r_norm >= 0.15) & (r_norm < 0.45)
    high_mask = r_norm >= 0.45

    low_e = mag[low_mask].mean()
    mid_e = mag[mid_mask].mean()
    high_e = mag[high_mask].mean()

    ring_energy = []
    n_rings = 16
    for i in range(n_rings):
        m = (r_norm >= i / n_rings) & (r_norm < (i + 1) / n_rings)
        if m.sum() > 0:
            ring_energy.append(mag[m].mean())
        else:
            ring_energy.append(0.0)
    ring_energy = np.array(ring_energy)
    ring_std = ring_energy.std()
    ring_peak_ratio = (ring_energy.max() + 1e-6) / (ring_energy.mean() + 1e-6)

    axis_band = 3
    off_axis_mask = (r_norm > 0.2) & (np.abs(yy - cy) > axis_band) & (np.abs(xx - cx) > axis_band)
    off_axis_peak = mag[off_axis_mask].max() if off_axis_mask.any() else 0.0
    off_axis_ratio = (off_axis_peak + 1e-6) / (mag[mid_mask].mean() + 1e-6)

    return [low_e, mid_e, high_e, mid_e - high_e, ring_std, ring_peak_ratio, off_axis_ratio]


def _fft_features(gray):
    h, w = gray.shape
    s = min(384, h, w)
    centers = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (h // 4, 3 * w // 4),
        (3 * h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
    ]
    all_stats = []
    for cy0, cx0 in centers:
        y0 = min(max(cy0 - s // 2, 0), h - s)
        x0 = min(max(cx0 - s // 2, 0), w - s)
        patch = gray[y0:y0 + s, x0:x0 + s]
        all_stats.append(_fft_patch_stats(patch))
    all_stats = np.array(all_stats)
    mean_stats = all_stats.mean(axis=0).tolist()
    max_off_axis = all_stats[:, -1].max()
    max_ring_peak = all_stats[:, 5].max()
    return mean_stats + [max_off_axis, max_ring_peak]


def _sharpness_features(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = lap.var()

    # Local sharpness variance across tiles - real scenes usually have more
    # variation (near/far objects); recaptures of a flat screen are more uniform.
    h, w = gray.shape
    tiles = []
    ty, tx = 4, 4
    for i in range(ty):
        for j in range(tx):
            y0, y1 = i * h // ty, (i + 1) * h // ty
            x0, x1 = j * w // tx, (j + 1) * w // tx
            tile = gray[y0:y1, x0:x1]
            if tile.size > 0:
                tiles.append(cv2.Laplacian(tile, cv2.CV_64F).var())
    tiles = np.array(tiles) if tiles else np.array([0.0])
    tile_sharp_std = tiles.std()
    tile_sharp_mean = tiles.mean()

    return [lap_var, tile_sharp_std, tile_sharp_mean]


def _color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    sat_mean = s.mean()
    sat_std = s.std()
    val_mean = v.mean()

    # Highlight clipping: fraction of near-white / near-black pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clip_high = (gray > 250).mean()
    clip_low = (gray < 5).mean()

    # Blue/green channel bias relative to red (screens often skew cool)
    b, g, r = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    bg_bias = (b.mean() + g.mean()) / 2 - r.mean()

    # Color histogram "peakiness" - screens often use a narrower set of
    # discrete colors due to compression/panel quantization
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist / (hist.sum() + 1e-9)
    hist_entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))

    return [sat_mean, sat_std, val_mean, clip_high, clip_low, bg_bias, hist_entropy]


def _glare_features(img, gray):
    # Large, low-saturation bright regions = glare/reflection typical of screens
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    glare_mask = (v > 220) & (s < 40)
    glare_frac = glare_mask.mean()

    # Count connected bright blobs (specular reflections often form a few
    # smooth blobs vs many tiny sensor-noise highlights)
    mask_u8 = (glare_mask.astype(np.uint8)) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    big_blobs = 0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big_blobs = int((areas > (gray.size * 0.001)).sum())

    return [glare_frac, big_blobs]


def _border_features(gray):
    # Look for a strong rectangular edge near the image border (screen bezel)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    border = 0.04
    by, bx = int(h * border), int(w * border)
    border_mask = np.zeros_like(edges, dtype=bool)
    border_mask[:by, :] = True
    border_mask[-by:, :] = True
    border_mask[:, :bx] = True
    border_mask[:, -bx:] = True
    border_edge_density = edges[border_mask].mean() / 255.0

    # Long straight lines via Hough - bezels create long axis-aligned lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min(h, w) * 0.35, maxLineGap=10)
    n_long_lines = 0 if lines is None else len(lines)

    return [border_edge_density, n_long_lines]


def _chroma_moire_features(img):
    # Moire from photographing a screen's sub-pixel (RGB) grid shows up as
    # *color* fringing / high-frequency chroma noise that is largely
    # uncorrelated between channels - natural textures vary in luminance but
    # the R/G/B channels stay highly correlated locally. This is often a much
    # stronger discriminator than luminance-only FFT features.
    img = img.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]

    def highpass(ch):
        blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=1.5)
        return ch - blur

    hr, hg, hb = highpass(r), highpass(g), highpass(b)

    # Standard deviation of high-freq color-difference channels
    rg_diff_std = (hr - hg).std()
    gb_diff_std = (hg - hb).std()
    rb_diff_std = (hr - hb).std()

    # Correlation between channel high-pass residuals (lower = more chroma
    # noise / color moire, higher = coherent natural-texture edges)
    def corr(a, b_):
        a_f, b_f = a.ravel(), b_.ravel()
        a_std, b_std = a_f.std(), b_f.std()
        if a_std < 1e-6 or b_std < 1e-6:
            return 1.0
        cov = np.mean((a_f - a_f.mean()) * (b_f - b_f.mean()))
        return float(cov / (a_std * b_std))

    rg_corr = corr(hr, hg)
    gb_corr = corr(hg, hb)
    rb_corr = corr(hr, hb)

    chroma_noise_energy = (rg_diff_std + gb_diff_std + rb_diff_std) / 3.0
    channel_corr_mean = (rg_corr + gb_corr + rb_corr) / 3.0

    return [chroma_noise_energy, channel_corr_mean, rg_diff_std, rb_diff_std]


def _noise_features(gray):
    # Residual noise after denoising: sensor noise from an ORIGINAL scene
    # differs in structure from noise re-photographed off a screen (which is
    # smoothed then re-introduces panel/pixel-grid structure instead).
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)
    noise_std = residual.std()
    noise_mean_abs = np.abs(residual).mean()
    # High-freq noise autocorrelation peakiness (screens -> more periodic
    # residual). FFT cost grows fast with size, so cap the patch used here.
    h, w = residual.shape
    s = min(1024, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    res_crop = residual[y0:y0 + s, x0:x0 + s]
    fft_res = np.fft.fftshift(np.fft.fft2(res_crop))
    mag = np.abs(fft_res)
    peak_ratio = (mag.max() + 1e-6) / (mag.mean() + 1e-6)
    return [noise_std, noise_mean_abs, peak_ratio]


FEATURE_NAMES = (
    ["fft_low", "fft_mid", "fft_high", "fft_mid_minus_high", "fft_ring_std", "fft_ring_peak_ratio",
     "fft_off_axis_ratio", "fft_max_off_axis", "fft_max_ring_peak"]
    + ["lap_var", "tile_sharp_std", "tile_sharp_mean"]
    + ["sat_mean", "sat_std", "val_mean", "clip_high", "clip_low", "bg_bias", "hist_entropy"]
    + ["glare_frac", "glare_blobs"]
    + ["border_edge_density", "n_long_lines"]
    + ["noise_std", "noise_mean_abs", "noise_peak_ratio"]
    + ["chroma_noise_energy", "channel_corr_mean", "rg_diff_std", "rb_diff_std"]
)


def extract_features(path):
    img_hi, gray_hi = _load_gray_and_color(path, max_side=2000)
    # FFT (moire) features need the higher-resolution image to preserve
    # pixel-grid aliasing; everything else runs on a smaller, faster copy.
    fft_feats = _fft_features(gray_hi)

    h, w = img_hi.shape[:2]
    small_side = 2000
    scale = small_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img_hi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = img_hi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = list(fft_feats)
    feats += _sharpness_features(gray)
    feats += _color_features(img)
    feats += _glare_features(img, gray)
    feats += _border_features(gray)
    feats += _noise_features(gray)
    feats += _chroma_moire_features(img)
    return np.array(feats, dtype=np.float32)

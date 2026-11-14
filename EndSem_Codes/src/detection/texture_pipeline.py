"""
Classical texture-descriptor detector (LBP + GLCM + Gabor) at full-roll level.

ECE501 Digital Image Processing -- texture analysis (statistical / structural /
spectral descriptors).

This is the reference "normal-texture model" detector that the course topic
calls for, evaluated under exactly the same protocol as the dual-scale
morphological pipeline in `src/pipeline.py`:

  1. FIT  (clean TRAIN rolls only)
       - Per fabric, the mean/std of every LBP, GLCM and Gabor feature map are
         pooled over the active-fabric pixels of the clean training rolls
         (`fit_normal_texture_model`).
       - The detection threshold of each descriptor configuration is the
         99.99th percentile of the training deviation values of that fabric
         (the same calibration percentile as the morphological pipeline).
  2. DETECT (any roll)
       - Feature maps -> per-descriptor mean |Z-score| deviation map
         (`per_descriptor_deviation`) -> fused map (mean of the active
         descriptor deviations) -> global threshold -> small-component cleanup.
  3. SCREEN
       - Roll flagged if the cleaned prediction has >= `screening_min_px` pixels
         (one uniform threshold for every fabric).

No labels or test images are used for fitting or for choosing thresholds.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data.active_fabric_mask import detect_active_fabric_mask, get_support_valid_mask
from src.detection.deviation_map import per_descriptor_deviation
from src.detection.feature_fusion import flatten_multiscale
from src.detection.normal_texture import NormalTextureModel
from src.pipeline import apply_cleanup
from src.texture.texture_scale import ScaleParams, extract_multiscale_features
from src.utils.io import load_image_gray, load_mask_binary
from src.utils.parallel import limit_thread_oversubscription

DESCRIPTORS = ("lbp", "glcm", "gabor")
ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# Detector configurations evaluated: each descriptor alone, and all three fused.
CONFIGS: Dict[str, Tuple[str, ...]] = {
    "lbp": ("lbp",),
    "glcm": ("glcm",),
    "gabor": ("gabor",),
    "fused": ("lbp", "glcm", "gabor"),
}

DEFAULT_SCALE = ScaleParams(
    name="medium", lbp_radius=2, lbp_points=16, glcm_window=17,
    glcm_distance=2, gabor_frequency=0.15, gabor_sigma=4.0,
)


def _descriptor_keys(flat: Dict[str, np.ndarray]) -> List[str]:
    return [k for k in flat if k.split("__", 1)[1].split("_", 1)[0] in DESCRIPTORS]


def _roll_features(img: np.ndarray, scale: ScaleParams, glcm_levels: int) -> Dict[str, np.ndarray]:
    ms = extract_multiscale_features(img, [scale], ANGLES, glcm_levels, ANGLES)
    flat = flatten_multiscale(ms)
    return {k: flat[k] for k in _descriptor_keys(flat)}


def _valid_mask(img: np.ndarray, fabric_id: str, scale: ScaleParams) -> np.ndarray:
    act = detect_active_fabric_mask(img, fabric_id=fabric_id)
    k = scale.glcm_window
    return get_support_valid_mask(act, (k, k), extra_margin=15)


def _fabric_task(args):
    """Worker: per-feature (sum, sum of squares, count) over valid pixels."""
    sample, scale, glcm_levels = args
    img = load_image_gray(sample["image_path"])
    valid = _valid_mask(img, sample["fabric_id"], scale)
    flat = _roll_features(img, scale, glcm_levels)
    stats = {}
    for key, fmap in flat.items():
        v = fmap[valid].astype(np.float64)
        v = v[np.isfinite(v)]
        stats[key] = (float(v.sum()), float((v * v).sum()), int(v.size))
    return sample["fabric_id"], stats


def _deviation_maps(flat: Dict[str, np.ndarray], model: NormalTextureModel) -> Dict[str, np.ndarray]:
    """Per-configuration deviation maps (mean |Z| per descriptor; fused = mean of the three)."""
    per = {d: per_descriptor_deviation(flat, model, d) for d in DESCRIPTORS}
    out = {name: np.mean([per[d] for d in descs], axis=0) for name, descs in CONFIGS.items()}
    return out


def _train_dev_task(args):
    """Worker: subsample of the training deviation values of every configuration."""
    sample, scale, glcm_levels, model, n_sub, seed = args
    img = load_image_gray(sample["image_path"])
    valid = _valid_mask(img, sample["fabric_id"], scale)
    devs = _deviation_maps(_roll_features(img, scale, glcm_levels), model)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid.ravel())
    if idx.size > n_sub:
        idx = rng.choice(idx, size=n_sub, replace=False)
    return sample["fabric_id"], {n: d.ravel()[idx].astype(np.float32) for n, d in devs.items()}


def _detect_task(args):
    sample, scale, glcm_levels, model, thresholds, min_area = args
    img = load_image_gray(sample["image_path"])
    valid = _valid_mask(img, sample["fabric_id"], scale)
    devs = _deviation_maps(_roll_features(img, scale, glcm_levels), model)
    is_def = bool(sample["is_defective"])
    gt = (load_mask_binary(sample["mask_path"]).astype(bool)
          if is_def and sample.get("mask_path") else np.zeros(img.shape, dtype=bool))
    rec = {"sample_id": sample["sample_id"], "fabric_id": sample["fabric_id"],
           "is_defective": is_def, "gt_px": int(gt.sum()), "cfg": {}}
    for name, dev in devs.items():
        pred = apply_cleanup((dev > thresholds[name]) & valid, min_area=min_area)
        rec["cfg"][name] = {"pred_px": int(pred.sum()), "tp": int((pred & gt).sum())}
    return rec


class TextureDescriptorPipeline:
    """LBP / GLCM / Gabor normal-texture detector (see module docstring)."""

    def __init__(self, scale: ScaleParams = DEFAULT_SCALE, glcm_levels: int = 32,
                 percentile: float = 99.99, min_area: int = 5,
                 screening_min_px: int = 15, n_workers: int = 8, n_sub: int = 40000):
        self.scale = scale
        self.glcm_levels = glcm_levels
        self.percentile = percentile
        self.min_area = min_area
        self.screening_min_px = screening_min_px
        self.n_workers = n_workers
        self.n_sub = n_sub
        self.models: Dict[str, NormalTextureModel] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}

    def _pool(self):
        return ProcessPoolExecutor(max_workers=self.n_workers, initializer=limit_thread_oversubscription)

    def fit(self, train_samples: Sequence[dict]) -> "TextureDescriptorPipeline":
        clean = [s for s in train_samples if not s["is_defective"]]
        with self._pool() as ex:
            stats_list = list(ex.map(_fabric_task, [(s, self.scale, self.glcm_levels) for s in clean]))
        agg: Dict[str, Dict[str, List[float]]] = {}
        counts: Dict[str, int] = {}
        for fid, stats in stats_list:
            counts[fid] = counts.get(fid, 0) + 1
            a = agg.setdefault(fid, {})
            for key, (s1, s2, n) in stats.items():
                cur = a.setdefault(key, [0.0, 0.0, 0])
                cur[0] += s1; cur[1] += s2; cur[2] += n
        for fid, a in agg.items():
            means, stds = {}, {}
            for key, (s1, s2, n) in a.items():
                m = s1 / max(n, 1)
                means[key] = m
                stds[key] = max(float(np.sqrt(max(s2 / max(n, 1) - m * m, 0.0))), 1e-8)
            self.models[fid] = NormalTextureModel(fabric_id=fid, means=means, stds=stds, n_images=counts[fid])
        tasks = [(s, self.scale, self.glcm_levels, self.models[s["fabric_id"]], self.n_sub, i)
                 for i, s in enumerate(clean)]
        pooled: Dict[str, Dict[str, List[np.ndarray]]] = {}
        with self._pool() as ex:
            for fid, devs in ex.map(_train_dev_task, tasks):
                p = pooled.setdefault(fid, {})
                for name, v in devs.items():
                    p.setdefault(name, []).append(v)
        for fid, p in pooled.items():
            self.thresholds[fid] = {n: float(np.percentile(np.concatenate(v), self.percentile))
                                    for n, v in p.items()}
        return self

    def evaluate(self, samples: Sequence[dict]) -> Dict[str, dict]:
        """Roll-level confusion + pixel Dice for every configuration."""
        known = [s for s in samples if s["fabric_id"] in self.models]
        skipped = len(samples) - len(known)
        tasks = [(s, self.scale, self.glcm_levels, self.models[s["fabric_id"]],
                  self.thresholds[s["fabric_id"]], self.min_area) for s in known]
        with self._pool() as ex:
            recs = list(ex.map(_detect_task, tasks))
        out: Dict[str, dict] = {}
        for name in CONFIGS:
            tp = fn = fp = tn = 0
            tp_px = fp_px = fn_px = 0
            dices: Dict[str, List[float]] = {}
            for r in recs:
                c = r["cfg"][name]
                flagged = c["pred_px"] >= self.screening_min_px
                if r["is_defective"]:
                    tp += flagged; fn += (not flagged)
                    tp_px += c["tp"]; fp_px += c["pred_px"] - c["tp"]; fn_px += r["gt_px"] - c["tp"]
                    den = c["pred_px"] + r["gt_px"]
                    dices.setdefault(r["fabric_id"], []).append(2.0 * c["tp"] / den if den else 0.0)
                else:
                    fp += flagged; tn += (not flagged)
            n = tp + fn + fp + tn
            pooled = 2.0 * tp_px / max(2.0 * tp_px + fp_px + fn_px, 1)
            macro = float(np.mean([np.mean(v) for v in dices.values()])) if dices else 0.0
            out[name] = {
                "accuracy_pct": 100.0 * (tp + tn) / max(n, 1),
                "recall_pct": 100.0 * tp / max(tp + fn, 1),
                "precision_pct": 100.0 * tp / max(tp + fp, 1),
                "specificity_pct": 100.0 * tn / max(tn + fp, 1),
                "pooled_dice_pct": 100.0 * pooled,
                "macro_dice_pct": 100.0 * macro,
                "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                "skipped_unknown_fabric": int(skipped),
            }
        return out

"""
compare_depth_methods.py

OFFLINE comparison of depth estimation methods on a recorded session
(sesija_dubina.npz from record_depth_session.py). No camera is required.

METHODS:
  realsense  - measured sensor depth                  (REFERENCE)
  yolo       - YOLO26-depth, monocular, metric       (pip install ultralytics)
  depthany   - Depth Anything V2                     (optional, HuggingFace)
  mp_size    - apparent hand size in the image        (baseline)

All methods estimate depth at the SAME PIXEL -- the hand wrist pixel
(landmark 0), which is already detected and saved during recording.
This compares exactly what the pipeline uses, rather than average scene depth.

NOTE on mp_size baseline:
  MediaPipe hand_world_landmarks describe the hand shape in metric units,
  but are relative to the wrist and do not provide absolute camera distance.
  Therefore, the apparent hand size in pixels is used as a baseline:
  the farther the hand is, the smaller it appears in the image.
  The relation is 1/d, so one calibration parameter (k) is determined
  at the closest recorded distance, followed by d = k / size.

Usage:
    python compare_depth_methods.py
    python compare_depth_methods.py --methods realsense yolo mp_size
    python compare_depth_methods.py --yolo-model yolo26s-depth.pt
"""

import argparse
import json
import os

import numpy as np

# Loading the recorded session
def load_session(path="sesija_dubina.npz", depth_scale_override=None):
    """
    depth_scale_override: if the session was recorded with an incorrect
    depth scale, it can be corrected without recording again because
    the raw depth values are stored. D405 typically uses 0.0001 m/unit.
    """
    d = np.load(path, allow_pickle=True)
    return {
        "color_jpg": d["color_jpg"],
        "depth": d["depth"],
        "landmarks": d["landmarks"],
        "label_cm": d["label_cm"],
        "intrinsics": d["intrinsics"][0],
        "depth_scale": float(depth_scale_override if depth_scale_override is not None
                             else d["depth_scale"]),
    }


def decode_color(session, i):
    import cv2
    return cv2.imdecode(np.frombuffer(session["color_jpg"][i], np.uint8), cv2.IMREAD_COLOR)


def wrist_pixel(session, i):
    """Returns the wrist pixel (landmark 0) in frame i."""
    h, w = session["depth"][i].shape
    lm = session["landmarks"][i][0]
    px = int(np.clip(lm[0] * w, 0, w - 1))
    py = int(np.clip(lm[1] * h, 0, h - 1))
    return px, py


def _to_numpy(x):
    """
    Converts a model output to a NumPy array regardless of its original type.

    Handles GPU tensors (.cpu()), tensors with gradients (.detach()),
    lists/tuples of tensors, and regular NumPy arrays.
    """
    if isinstance(x, (list, tuple)):
        if not x:
            raise ValueError("Model returned an empty list.")
        x = x[0]
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32).squeeze()


def _sample_patch(depth_map, px, py, k=2):
    """
    Returns the median value in a small window around the pixel, ignoring zeros.
    A single pixel on a thin structure or hand edge may return an invalid value,
    so the local median is used instead.
    """
    h, w = depth_map.shape
    y0, y1 = max(0, py - k), min(h, py + k + 1)
    x0, x1 = max(0, px - k), min(w, px + k + 1)
    patch = depth_map[y0:y1, x0:x1].astype(np.float64)
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else np.nan


# Depth estimation methods
def depth_realsense(session):
    """REFERENCE: measured sensor depth at the wrist pixel."""
    out = []
    for i in range(len(session["depth"])):
        px, py = wrist_pixel(session, i)
        raw = _sample_patch(session["depth"][i], px, py)
        out.append(raw * session["depth_scale"] if not np.isnan(raw) else np.nan)
    return np.array(out)


def depth_yolo(session, model_name="yolo26n-depth.pt"):
    """YOLO26-depth: dense metric depth map from a single RGB image."""
    from ultralytics import YOLO
    model = YOLO(model_name)

    out = []
    for i in range(len(session["depth"])):
        img = decode_color(session, i)
        res = model.predict(img, verbose=False)[0]
        dmap = _to_numpy(res.depth.data)      
        px, py = wrist_pixel(session, i)
        h_in, w_in = img.shape[:2]
        h_d, w_d = dmap.shape
        pxd = int(np.clip(px * w_d / w_in, 0, w_d - 1))
        pyd = int(np.clip(py * h_d / h_in, 0, h_d - 1))
        out.append(_sample_patch(dmap, pxd, pyd))
    return np.array(out)


def depth_depthanything(session, model_name="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
    """
    Depth Anything V2 metric model for indoor scenes.

    Available sizes: Small-hf, Base-hf, Large-hf. Outdoor variants also exist,
    but Indoor is the appropriate choice for a tabletop setup.

    If a relative model is used instead, its output is not in meters and
    scale fitting is required; see fit_scale_to_reference.
    """
    import torch
    from transformers import pipeline as hf_pipeline
    from PIL import Image
    import cv2

    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline("depth-estimation", model=model_name, device=device)

    out = []
    for i in range(len(session["depth"])):
        img = decode_color(session, i)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dmap = _to_numpy(pipe(pil)["predicted_depth"])
        px, py = wrist_pixel(session, i)
        h_in, w_in = img.shape[:2]
        h_d, w_d = dmap.shape
        pxd = int(np.clip(px * w_d / w_in, 0, w_d - 1))
        pyd = int(np.clip(py * h_d / h_in, 0, h_d - 1))
        out.append(_sample_patch(dmap, pxd, pyd))
    return np.array(out)


def hand_apparent_size(session, i):
    """
    Apparent hand size in pixels, measured as the diagonal of the bounding
    box containing all 21 landmarks. This is more robust than the distance
    between two landmarks because it does not depend on finger configuration.
    """
    h, w = session["depth"][i].shape
    lm = session["landmarks"][i]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def depth_mp_size(session, calib_distance_m=None, reference=None):
    """
    Baseline: depth from apparent hand size, d = k / size.
    The calibration parameter k is determined at the closest recorded
    distance and then applied to all other frames.
    """
    sizes = np.array([hand_apparent_size(session, i) for i in range(len(session["depth"]))])
    labels = session["label_cm"]

    calib_cm = labels.min()
    mask = labels == calib_cm
    if calib_distance_m is None:
        calib_distance_m = calib_cm / 100.0
    k = calib_distance_m * np.median(sizes[mask])
    return k / sizes


def fit_scale_to_reference(pred, ref):
    """
    For methods with relative output, fits the best linear transformation
    (a*pred + b) to the reference using least squares.

    This is the most favorable possible case for a relative method, so it
    should be reported transparently in the evaluation.
    """
    m = np.isfinite(pred) & np.isfinite(ref)
    if m.sum() < 2:
        return pred
    a, b = np.polyfit(pred[m], ref[m], 1)
    return a * pred + b


# Metrics
def compute_metrics(pred, ref, labels_cm):
    """Computes metrics per distance and overall. pred/ref are in meters."""
    m = np.isfinite(pred) & np.isfinite(ref)
    err = pred - ref

    per_distance = {}
    for d in np.unique(labels_cm):
        sel = (labels_cm == d) & m
        if sel.sum() == 0:
            continue
        e = err[sel]
        per_distance[int(d)] = {
            "bias_mm": float(np.mean(e) * 1000),       
            "std_mm": float(np.std(e) * 1000),         
            "mae_mm": float(np.mean(np.abs(e)) * 1000),
            "n": int(sel.sum()),
            "mean_pred_m": float(np.mean(pred[sel])),
            "mean_ref_m": float(np.mean(ref[sel])),
        }

    return {
        "per_distance": per_distance,
        "overall": {
            "bias_mm": float(np.mean(err[m]) * 1000),
            "std_mm": float(np.std(err[m]) * 1000),
            "mae_mm": float(np.mean(np.abs(err[m])) * 1000),
            "rmse_mm": float(np.sqrt(np.mean(err[m] ** 2)) * 1000),
            "valid_frames": int(m.sum()),
            "total_frames": int(len(pred)),
        },
    }


def metrics_vs_ruler(pred, labels_cm):
    """Computes error relative to the ruler, independently of the sensor."""
    ruler_m = labels_cm / 100.0
    return compute_metrics(pred, ruler_m, labels_cm)


# Main pipeline
def run(session_path="sesija_dubina.npz", methods=("realsense", "yolo", "depthany", "mp_size"),
        yolo_model="yolo26n-depth.pt", out_json="results_depth.json",
        depth_scale_override=None):
    import time

    session = load_session(session_path, depth_scale_override)
    labels = session["label_cm"]
    n = len(labels)
    print(f"Loaded {n} frames, distances: {sorted(set(labels.tolist()))} cm\n")

    preds, timings = {}, {}

    for method in methods:
        print(f"  {method:12s} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            if method == "realsense":
                preds[method] = depth_realsense(session)
            elif method == "yolo":
                preds[method] = depth_yolo(session, yolo_model)
            elif method == "depthany":
                preds[method] = depth_depthanything(session)
            elif method == "mp_size":
                preds[method] = depth_mp_size(session)
            else:
                print("unknown method, skipping")
                continue
            elapsed = time.perf_counter() - t0
            timings[method] = elapsed / n * 1000
            print(f"done ({timings[method]:.1f} ms/frame)")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

    if "realsense" not in preds:
        raise RuntimeError("No reference (realsense) -- comparison is not possible.")

    ref_bias_mm = abs(np.nanmean(preds["realsense"] - labels / 100.0)) * 1000
    if ref_bias_mm > 100:
        factor = np.nanmean(preds["realsense"]) / np.mean(labels / 100.0)
        print("\n" + "!" * 68)
        print(f"WARNING: RealSense reference differs from the ruler by {ref_bias_mm:.0f} mm.")
        print(f"Measured-to-true distance ratio is approximately {factor:.1f}x.")

        if 5 < factor < 20:
            print("This indicates an incorrect depth scale. Run again with:")
            print(f"    --depth-scale {float(session['depth_scale'])/10:g}")
        print("!" * 68 + "\n")

    ref = preds["realsense"]

    # Scale alignment for monocular methods.
    # Both raw and aligned results are reported because the reference would not be available during real deployment.
    for m in list(preds.keys()):
        if m in ("realsense",):
            continue
        aligned = fit_scale_to_reference(preds[m], ref)
        preds[m + "_aligned"] = aligned
        if m in timings:
            timings[m + "_aligned"] = timings[m]

    results = {"n_frames": n, "distances_cm": sorted(set(labels.tolist())),
               "timings_ms_per_frame": timings, "methods": {}}

    for method, pred in preds.items():
        results["methods"][method] = {
            "vs_sensor": compute_metrics(pred, ref, labels),
            "vs_ruler": metrics_vs_ruler(pred, labels),
            "pred": pred.tolist(),
        }
    results["labels_cm"] = labels.tolist()

    with open(out_json, "w") as f:
        json.dump(results, f)
    print(f"\nSaved to {out_json}")

    print_detailed_table(results)
    return results


def print_detailed_table(results):
    """Detailed output: estimated vs. actual distance for each method."""
    labels = np.array(results["labels_cm"])
    distances = sorted(set(labels.tolist()))

    # ---- Table 1: estimated distance by actual distance ----
    print("\n" + "=" * 78)
    print("ESTIMATED DISTANCE [cm]  (mean +- std, by actual distance)")
    print("=" * 78)
    header = f"{'Stvarno':>9s} |" + "".join(f"{m:>21s}" for m in results["methods"])
    print(header)
    print("-" * len(header))
    for d in distances:
        row = f"{d:>7d}cm |"
        for m in results["methods"]:
            pred = np.array(results["methods"][m]["pred"])
            sel = (labels == d) & np.isfinite(pred)
            if sel.sum():
                vals = pred[sel] * 100  
                row += f"{vals.mean():>13.1f} +-{vals.std():>5.1f}"
            else:
                row += f"{'--':>21s}"
        print(row)

    # ---- Table 2: error relative to the ruler ----
    print("\n" + "=" * 78)
    print("ERROR RELATIVE TO RULER [mm]  (bias / standard deviation)")
    print("=" * 78)
    print(header)
    print("-" * len(header))
    for d in distances:
        row = f"{d:>7d}cm |"
        for m in results["methods"]:
            pd_ = results["methods"][m]["vs_ruler"]["per_distance"]
            key = str(d) if str(d) in pd_ else (d if d in pd_ else None)
            if key is not None:
                e = pd_[key]
                row += f"{e['bias_mm']:>13.1f} /{e['std_mm']:>6.1f}"
            else:
                row += f"{'--':>21s}"
        print(row)

    # ---- Table 3: overall results ----
    print("\n" + "=" * 78)
    print("OVERALL")
    print("=" * 78)
    print(f"{'Method':22s} {'Bias':>13s} {'Std':>12s} {'MAE':>10s} "
          f"{'Rel.error':>12s} {'ms/frame':>10s}")
    print("-" * 78)
    for m in results["methods"]:
        o = results["methods"][m]["vs_ruler"]["overall"]
        t = results.get("timings_ms_per_frame", {}).get(m, float("nan"))
        rel = 100 * (o["mae_mm"] / 10.0) / np.mean(labels)
        print(f"{m:22s} {o['bias_mm']:>10.1f}mm {o['std_mm']:>10.1f}mm "
              f"{o['mae_mm']:>8.1f}mm {rel:>10.1f}% {t:>9.1f}")
    print("=" * 78)
    print("Bias       = systematic deviation")
    print("Std        = spread around the mean error")
    print("Rel. error = MAE relative to the mean actual distance")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sesija_dubina.npz")
    ap.add_argument("--methods", nargs="+", default=["realsense", "yolo", "depthany", "mp_size"])
    ap.add_argument("--yolo-model", default="yolo26n-depth.pt")
    ap.add_argument("--out", default="results_depth.json")
    ap.add_argument("--depth-scale", type=float, default=None,
                    help="correct depth scale without recording again (e.g. 0.0001 for D405)")
    args = ap.parse_args()

    run(args.session, tuple(args.methods), args.yolo_model, args.out, args.depth_scale)
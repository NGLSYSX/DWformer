import os
import time
from datetime import datetime
from pathlib import Path
from glob import glob
from natsort import natsorted
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib

CONFIG = {
    "weight_path": "",
    "yaml_path": "options/all.yml",
    "data_root": "",
    "datasets": [
        {"name": "rain", "label": "Rain", "lq_path": ""},
        {"name": "snow", "label": "Snow", "lq_path": ""},
        {"name": "haze", "label": "Haze", "lq_path": ""},
        {"name": "clean", "label": "Clean", "lq_path": ""},
    ],
    "output_base": None,
    "recursive": False,
    "factor": 8,
    "use_amp": True,
    "use_dp": True,
}

try:
    from basicsr.models.archs.restormer_arch import EnhancedRestormer as Restormer
except Exception:
    from basicsr.models.archs.restormer_arch import Restormer


def load_yaml(yaml_path):
    import yaml
    try:
        from yaml import CLoader as Loader
    except Exception:
        from yaml import Loader
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=Loader)


def build_model_from_yaml(yaml_path):
    cfg = load_yaml(yaml_path)
    net_cfg = dict(cfg.get("network_g", {}))
    net_cfg.pop("type", None)
    return Restormer(**net_cfg)


def smart_load_state_dict(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        sd = obj.get("params_ema") or obj.get("params") or obj.get("state_dict") or obj
    else:
        sd = obj
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    model_keys = set(model.state_dict().keys())
    filtered_sd = {k: v for k, v in new_sd.items() if k in model_keys}
    dropped = sorted(set(new_sd.keys()) - model_keys)
    missing = sorted(k for k in model_keys if k not in filtered_sd)
    print(f"load matched={len(filtered_sd)} dropped={len(dropped)} missing={len(missing)}")
    if dropped:
        print("dropped sample:", dropped[:10])
    if missing:
        print("missing sample:", missing[:10])
    model.load_state_dict(filtered_sd, strict=False)


def pad_to_factor(x, factor=8):
    b, c, h, w = x.shape
    H = (h + factor - 1) // factor * factor
    W = (w + factor - 1) // factor * factor
    if H != h or W != w:
        x = F.pad(x, (0, W - w, 0, H - h), mode="reflect")
    return x, h, w


def collect_images(input_dir, recursive=False):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    if not recursive:
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(input_dir, ext)))
        return natsorted(files)
    out = []
    for root, _, _ in os.walk(input_dir):
        for ext in exts:
            out.extend(glob(os.path.join(root, ext)))
    return natsorted(out)


def ensure_output_base():
    if CONFIG["output_base"]:
        os.makedirs(CONFIG["output_base"], exist_ok=True)
        return CONFIG["output_base"]
    ts = datetime.now().strftime("%m%d_%H%M%S")
    wname = Path(CONFIG["weight_path"]).stem or "net_g"
    base = Path(__file__).resolve().parent / f"weights_stat_{wname}_{ts}"
    os.makedirs(base, exist_ok=True)
    CONFIG["output_base"] = str(base)
    return CONFIG["output_base"]


def to_scalar(x):
    return float(x.detach().cpu().mean().item())


def safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        return plt, np
    except Exception:
        return None, None


def plot_radar(save_path, labels, values):
    plt, np = safe_import_matplotlib()
    if plt is None:
        return False
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    for name, vals, color in values:
        v = vals + vals[:1]
        ax.plot(angles, v, color=color, linewidth=2, label=name)
        ax.fill(angles, v, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_lines(save_path, xs, series):
    plt, np = safe_import_matplotlib()
    if plt is None:
        return False
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    for name, ys, color in series:
        ax.plot(xs, ys, color=color, linewidth=1.5, label=name)
    ax.set_xlabel("index")
    ax.set_ylabel("weight")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def main():
    out_base = ensure_output_base()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_yaml(CONFIG["yaml_path"])
    smart_load_state_dict(model, CONFIG["weight_path"]) 
    model = model.to(device)
    if torch.cuda.is_available() and CONFIG["use_dp"]:
        model = nn.DataParallel(model)
    model.eval()

    use_amp = CONFIG["use_amp"] and torch.cuda.is_available()
    amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

    summary_rows = []
    radar_values = []
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    color_idx = 0

    for di, dataset in enumerate(CONFIG["datasets"], 1):
        lq_rel = dataset.get("lq_path", "")
        if not lq_rel or not lq_rel.strip():
            continue
        lq_dir = os.path.join(CONFIG["data_root"], lq_rel)
        if not os.path.exists(lq_dir):
            continue
        files = collect_images(lq_dir, recursive=CONFIG["recursive"])
        if not files:
            continue
        color = colors[color_idx % len(colors)]
        color_idx += 1

        rec_w_lka = []
        rec_w_freq = []
        rec_w_rcp = []
        cond_means = []
        cond_stds = []

        m = model.module if isinstance(model, nn.DataParallel) else model
        if not hasattr(m, "weight_predictor") or not hasattr(m, "weather_encoder"):
            print(out_base)
            return
        if not getattr(m, "use_weather_adaptive", False):
            print(out_base)
            return

        with torch.no_grad():
            for fp in files:
                img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = np.float32(img_rgb) / 255.0
                ten = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
                cond_feat = m.weather_encoder(ten, return_logits=False)
                wl, wf, wr = m.weight_predictor(cond_feat)
                rec_w_lka.append(to_scalar(wl))
                rec_w_freq.append(to_scalar(wf))
                rec_w_rcp.append(to_scalar(wr))
                cond_means.append(float(cond_feat.detach().cpu().mean().item()))
                cond_stds.append(float(cond_feat.detach().cpu().std().item()))

        

        ds_out = Path(out_base) / dataset["name"]
        os.makedirs(ds_out, exist_ok=True)

        xs = list(range(1, len(rec_w_lka) + 1))
        plot_lines(str(ds_out / "lines.png"), xs, [("w_lka", rec_w_lka, color), ("w_freq", rec_w_freq, "#ff7f0e"), ("w_rcp", rec_w_rcp, "#17becf")])

        with open(str(ds_out / "weights.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "w_lka", "w_freq", "w_rcp"]) 
            for i, (a, b, c) in enumerate(zip(rec_w_lka, rec_w_freq, rec_w_rcp), start=1):
                w.writerow([i, a, b, c])
        with open(str(ds_out / "cond_stats.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "cond_mean", "cond_std"]) 
            for i, (cm, cs) in enumerate(zip(cond_means, cond_stds), start=1):
                w.writerow([i, cm, cs])

        m_lka = float(np.mean(rec_w_lka)) if len(rec_w_lka) else 0.0
        m_freq = float(np.mean(rec_w_freq)) if len(rec_w_freq) else 0.0
        m_rcp = float(np.mean(rec_w_rcp)) if len(rec_w_rcp) else 0.0
        s_lka = float(np.std(rec_w_lka)) if len(rec_w_lka) else 0.0
        s_freq = float(np.std(rec_w_freq)) if len(rec_w_freq) else 0.0
        s_rcp = float(np.std(rec_w_rcp)) if len(rec_w_rcp) else 0.0

        summary_rows.append([dataset["name"], dataset.get("label", dataset["name"]), len(rec_w_lka), m_lka, m_freq, m_rcp, s_lka, s_freq, s_rcp])
        radar_values.append((dataset.get("label", dataset["name"]), [m_lka, m_freq, m_rcp], color))

    with open(os.path.join(out_base, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label", "count", "mean_w_lka", "mean_w_freq", "mean_w_rcp", "std_w_lka", "std_w_freq", "std_w_rcp"]) 
        for row in summary_rows:
            w.writerow(row)

    plot_radar(os.path.join(out_base, "radar.png"), ["w_lka", "w_freq", "w_rcp"], radar_values)
    print(out_base)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))

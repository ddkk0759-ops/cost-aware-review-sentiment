"""Shared utilities: reproducibility + device selection."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from . import config


def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device (mps > cuda > cpu)."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_compute_summary() -> None:
    import torch
    dev = get_device()
    print(f"[device] using: {dev}")
    print(f"[device] mps_built={torch.backends.mps.is_built()} "
          f"mps_avail={torch.backends.mps.is_available()} "
          f"cuda_avail={torch.cuda.is_available()}")
    try:
        use_xgb = config.classic_tree_use_cuda()
        use_lgb = config.lightgbm_use_cuda()
        print(f"[classic trees] CLASSIC_TREE_DEVICE={config.CLASSIC_TREE_DEVICE}  "
              f"XGBoost={'cuda' if use_xgb else 'cpu'}  "
              f"LightGBM={'cuda' if use_lgb else 'cpu'}  "
              f"(set LIGHTGBM_USE_CUDA=1 for pip+GPU LGBM build)")
        print(f"[seq dataloader] BATCH_SIZE={config.BATCH_SIZE}  "
              f"num_workers={config.TORCH_NUM_WORKERS}  "
              f"pin_memory={config.TORCH_PIN_MEMORY}  AMP={config.SEQ_USE_AMP}")
    except Exception:
        pass


def configure_torch_training_backends() -> None:
    """Call once before heavy PyTorch training (CUDA throughput)."""
    try:
        import torch
        if torch.cuda.is_available() and config.TORCH_CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    except ImportError:
        pass


_MPL_ZH_FONT_CACHE = Path.home() / ".cache" / "sentiment_analysis_mpl_fonts"
# 单文件 OTF，便于无系统 CJK 字体时下载后 addfont（Source Han / Noto 系列）
_NOTO_SC_OTF_URL = (
    "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/"
    "Sans/SubsetOTF/SC/NotoSansSC-Regular.otf"
)


def _fc_match_file(pattern: str) -> Path | None:
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}\n", pattern],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        p = Path(out.stdout.strip().split("\n")[0])
        return p if p.is_file() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _find_local_cjk_ttc() -> Path | None:
    """Linux 常见路径：NotoSansCJK TTC 内含 SC/JP/TC 等多字族。"""
    globs = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    )
    for g in globs:
        p = Path(g)
        if p.is_file():
            return p
    fonts_root = Path("/usr/share/fonts")
    if fonts_root.is_dir():
        for p in fonts_root.rglob("NotoSansCJK-Regular.ttc"):
            return p
    p = _fc_match_file("Noto Sans CJK SC")
    return p


def _ensure_noto_sc_otf() -> Path | None:
    """无系统 Noto TTC 时，拉取 NotoSansSC Subset OTF 到用户缓存（约 8MB，仅首次）。"""
    _MPL_ZH_FONT_CACHE.mkdir(parents=True, exist_ok=True)
    target = _MPL_ZH_FONT_CACHE / "NotoSansSC-Regular.otf"
    if target.is_file() and target.stat().st_size > 500_000:
        return target
    try:
        req = urllib.request.Request(
            _NOTO_SC_OTF_URL,
            headers={"User-Agent": "sentiment-analysis/1.0"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp, target.open("wb") as f:
            shutil.copyfileobj(resp, f)
    except OSError:
        if target.is_file():
            target.unlink(missing_ok=True)
        return None
    return target if target.is_file() and target.stat().st_size > 500_000 else None


def setup_matplotlib_chinese() -> None:
    """
    让 matplotlib / seaborn 图中的中文正常显示。

    - 优先用系统 Noto Sans CJK（TTC 需 addfont，否则 rc 里写 SC 会回退 DejaVu）。
    - 否则尝试下载 NotoSansSC OTF 到 ~/.cache。
    - 最后保留 macOS / Windows 常见字体名作为备选。
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False

    ttc = _find_local_cjk_ttc()
    if ttc is not None:
        try:
            font_manager.fontManager.addfont(str(ttc))
        except (ValueError, OSError):
            pass
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "Noto Sans CJK TC",
            "Noto Sans CJK JP",
            "Noto Sans CJK KR",
            "DejaVu Sans",
        ]
        return

    otf = _ensure_noto_sc_otf()
    if otf is not None:
        try:
            font_manager.fontManager.addfont(str(otf))
        except (ValueError, OSError):
            otf = None
    if otf is not None:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]

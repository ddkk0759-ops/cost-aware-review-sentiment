"""Training utilities for sequence models with cost-aware CrossEntropyLoss."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from . import config
from .cost import torch_class_weight_tensor
from .models_seq import SEQ_BUILDERS
from .utils import configure_torch_training_backends, get_device


class _SentDataset(Dataset):
    def __init__(self, indices_arr: np.ndarray, labels: List[int]):
        self.x = indices_arr.astype(np.int64)
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i], dtype=torch.long)


def make_loader(indices_arr, labels, batch_size: int = config.BATCH_SIZE,
                shuffle: bool = False) -> DataLoader:
    pin = bool(config.TORCH_PIN_MEMORY and torch.cuda.is_available())
    nw = max(0, config.TORCH_NUM_WORKERS)
    kw: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": nw,
        "pin_memory": pin,
    }
    if nw > 0:
        kw["persistent_workers"] = True
    return DataLoader(_SentDataset(indices_arr, labels), **kw)


def _train_epoch(model, loader, opt, criterion, device, use_amp: bool, scaler: GradScaler | None):
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total


def _eval_loader(model, loader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = total_correct = total = 0
    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                probas = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probas.append(probas.cpu().numpy())
    return (total_loss / total, total_correct / total,
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probas, axis=0))


def train_one(
    name: str,
    embedding_matrix,
    pad_idx: int,
    train_loader,
    val_loader,
    test_loader,
    epochs: int = config.EPOCHS,
    lr: float = 1e-3,
    cost_aware: bool = True,
    device=None,
    verbose: bool = True,
):
    """Train a single sequence model and return predictions on val/test."""
    if device is None:
        device = get_device()
    configure_torch_training_backends()
    use_amp = bool(config.SEQ_USE_AMP and device.type == "cuda")
    scaler: GradScaler | None = GradScaler() if use_amp else None

    model = SEQ_BUILDERS[name](embedding_matrix, pad_idx=pad_idx).to(device)

    ls = float(config.SEQ_LABEL_SMOOTHING)
    ce_kw: dict = {}
    if ls > 0.0:
        ce_kw["label_smoothing"] = ls
    if cost_aware:
        weight = torch_class_weight_tensor().to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, **ce_kw)
    else:
        criterion = nn.CrossEntropyLoss(**ce_kw)
    wd = float(config.SEQ_WEIGHT_DECAY)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    patience = int(config.SEQ_EARLY_STOPPING_PATIENCE)
    best_val = float("inf")
    best_state = None
    no_improve = 0
    if verbose:
        print(f"\n{'='*55}\n  training {name}  (cost_aware={cost_aware}  amp={use_amp}  "
              f"wd={wd:g}  label_smoothing={ls:g}  early_stop_patience={patience})\n{'='*55}")
    for epoch in range(1, epochs + 1):
        tl, ta = _train_epoch(model, train_loader, opt, criterion, device, use_amp, scaler)
        vl, va, *_ = _eval_loader(model, val_loader, criterion, device, use_amp)
        sched.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs}  "
                  f"train_loss={tl:.4f} train_acc={ta:.4f}  "
                  f"val_loss={vl:.4f} val_acc={va:.4f}")
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        elif patience > 0:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stopping (val_loss 连续 {patience} 个 epoch 未下降)")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    _, _, val_pred, val_labels, val_proba = _eval_loader(
        model, val_loader, criterion, device, use_amp)
    _, _, test_pred, test_labels, test_proba = _eval_loader(
        model, test_loader, criterion, device, use_amp)

    return {
        "name": name,
        "model": model,
        "history": history,
        "val_pred": val_pred,
        "val_proba": val_proba,
        "val_labels": val_labels,
        "test_pred": test_pred,
        "test_proba": test_proba,
        "test_labels": test_labels,
    }


SEQ_LR = {
    "MeanPooling": 1e-3,
    "LSTM": 5e-4,
    "Transformer": 5e-4,
    "TextCNN": 1e-3,
}


def train_all_seq(
    embedding_matrix,
    pad_idx: int,
    train_loader,
    val_loader,
    test_loader,
    cost_aware: bool = True,
    epochs: int = config.EPOCHS,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name in config.SEQ_MODEL_NAMES:
        out[name] = train_one(
            name, embedding_matrix, pad_idx,
            train_loader, val_loader, test_loader,
            epochs=epochs, lr=SEQ_LR[name],
            cost_aware=cost_aware,
        )
    return out

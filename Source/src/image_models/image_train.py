"""
train.py — Training pipeline for CLIP ViT-B/32 image-only binary classifier.

Supports three conditions:
    nCF       — 6k originals, no adversarial head
    CF        — 18k augmented, WITH gradient-reversal adversarial head
    CF-no-adv — 18k augmented, WITHOUT adversarial head

Training details:
    • Differential learning rates (backbone 1e-4, heads 1e-3)
    • AdamW with weight decay 1e-4
    • CosineAnnealingLR schedule
    • Early stopping on validation F1
    • Gradient clipping (max norm 1.0)
    • Label smoothing via soft BCE targets
"""

import os
import sys
import time
import copy
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_model, grl_lambda_schedule
from data_prep import N_GROUPS, MODELS_DIR, CHECKPOINT_DIR, OUTPUT_DIR, RANDOM_STATE
from config import ADV_WEIGHT

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ─── Device auto-detection ────────────────────────────────────────────────
def get_device():
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Apple MPS detected")
    else:
        device = torch.device("cpu")
        print("  Using CPU")
    return device


# ─── Default hyper-parameters ────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'epochs':          20,
    'batch_size':      64,
    'lr_backbone':     1e-4,
    'lr_heads':        1e-3,
    'weight_decay':    1e-4,
    'label_smoothing': 0.05,
    'adv_weight':      ADV_WEIGHT,  # GRL adversarial-loss weight (see image_models/config.py)
    'patience':        5,
    'min_delta':       1e-4,
    'freeze_blocks':   0,
    'architecture':    'clip_vit_b32',
    'freeze_encoder':  True,
    'dropout':         0.3,
    'num_workers':     2,
    'grad_clip':       1.0,
}


def _checkpoint_basename(condition: str, architecture: str) -> str:
    if architecture == 'clip_vit_b32':
        return f"clip_vitb32_{condition}.pth"
    return f"{architecture}_{condition}.pth"


# ─── Early stopping ─────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_score = None
        self.should_stop = False
        self.best_state  = None

    def __call__(self, score: float, model: nn.Module):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ─── One training epoch ─────────────────────────────────────────────────────
def _smooth_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Apply label smoothing to binary targets: y' = y*(1-α) + 0.5*α."""
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(
    model, loader, optimizer,
    criterion_task, criterion_adv,
    use_adversarial, adv_weight,
    epoch, total_epochs,
    label_smoothing=0.0,
    grad_clip=1.0,
    device='cpu',
    scaler=None,
    use_amp=False,
):
    model.train()
    running_loss = 0.0
    running_task = 0.0
    running_adv  = 0.0
    correct = 0
    total   = 0

    if use_adversarial:
        lam = grl_lambda_schedule(epoch, total_epochs)
        model.set_grl_lambda(lam)

    pbar = tqdm(loader, desc=f"    Epoch {epoch+1}/{total_epochs}",
                  leave=False, ncols=100)
    for batch_idx, (images, labels, groups) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        groups = groups.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            task_logits, adv_logits = model(images)

            # Task loss (BCE with optional label smoothing)
            smooth = _smooth_labels(labels, label_smoothing) if label_smoothing > 0 else labels
            t_loss = criterion_task(task_logits, smooth)

            # Adversarial loss
            if use_adversarial and adv_logits is not None:
                a_loss = criterion_adv(adv_logits, groups)
                loss = t_loss + adv_weight * a_loss
                running_adv += a_loss.item() * images.size(0)
            else:
                loss = t_loss

        # Mixed precision backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_task += t_loss.item() * images.size(0)
        preds = (torch.sigmoid(task_logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{correct/total:.4f}")

    n = max(total, 1)
    return {
        'loss':      running_loss / n,
        'task_loss': running_task / n,
        'adv_loss':  running_adv / n if use_adversarial else 0.0,
        'accuracy':  correct / n,
    }


# ─── Validation / test pass ─────────────────────────────────────────────────
def validate(model, loader, criterion_task, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels, _groups in loader:
            images = images.to(device)
            labels = labels.to(device)

            task_logits, _ = model(images)
            loss = criterion_task(task_logits, labels)

            probs = torch.sigmoid(task_logits)
            preds = (probs >= 0.5).long()

            running_loss += loss.item() * images.size(0)
            correct += (preds == labels.long()).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n = max(total, 1)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return {
        'loss':         running_loss / n,
        'accuracy':     correct / n,
        'f1':           f1,
        'predictions':  np.array(all_preds),
        'labels':       np.array(all_labels),
        'probabilities': np.array(all_probs),
    }


# ─── Full training loop for one condition ────────────────────────────────────
def train_model(
    condition: str,
    loaders: dict,
    config: dict = None,
    device=None,
) -> dict:
    """
    Train CLIP ViT-B/32 image classifier for one condition and return results dict.

    Parameters
    ----------
    condition : str   'ncf', 'cf', or 'cf_no_adv'
    loaders   : dict  with 'train', 'val', 'test' DataLoaders
    config    : dict  hyper-parameters (defaults to DEFAULT_CONFIG)
    device    : torch.device or None  (auto-detect if None)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    use_adversarial = (condition == 'cf')

    print(f"\n{'='*60}")
    print(f"TRAINING  — {condition.upper()}")
    print(f"  Architecture     : {config['architecture']}")
    print(f"  Adversarial head : {'ON' if use_adversarial else 'OFF'}")
    print(f"  Epochs / patience: {config['epochs']} / {config['patience']}")
    print(f"  LR backbone/heads: {config['lr_backbone']} / {config['lr_heads']}")
    print(f"  Device           : {device}")
    print(f"{'='*60}")

    # ── Mixed precision setup ────────────────────────────────────────────
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled")

    # ── Model ────────────────────────────────────────────────────────────
    model = create_model(
        use_adversarial=use_adversarial,
        n_groups=N_GROUPS,
        dropout=config['dropout'],
        freeze_blocks=config['freeze_blocks'],
        architecture=config['architecture'],
        freeze_encoder=config['freeze_encoder'],
    )
    model = model.to(device)

    # ── Optimizer with differential LR ───────────────────────────────────
    backbone_params = [p for p in model.get_backbone_parameters() if p.requires_grad]
    head_params = list(model.task_head.parameters())
    if use_adversarial:
        head_params += list(model.adv_head.parameters())

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': config['lr_backbone']})
    param_groups.append({'params': head_params, 'lr': config['lr_heads']})

    optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6,
    )

    criterion_task = nn.BCEWithLogitsLoss()
    criterion_adv  = nn.CrossEntropyLoss() if use_adversarial else None

    early_stop = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta'],
    )

    # ── Training loop ────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [],  'val_acc': [],
        'val_f1': [],
    }
    t0 = time.time()

    for epoch in range(config['epochs']):
        ep_start = time.time()
        print(f"\n  Epoch {epoch+1}/{config['epochs']}")

        train_m = train_one_epoch(
            model, loaders['train'], optimizer,
            criterion_task, criterion_adv,
            use_adversarial, config['adv_weight'],
            epoch, config['epochs'],
            label_smoothing=config['label_smoothing'],
            grad_clip=config['grad_clip'],
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_m = validate(model, loaders['val'], criterion_task, device)

        scheduler.step()

        ep_time = time.time() - ep_start
        print(f"  → train_loss {train_m['loss']:.4f} | "
              f"val_loss {val_m['loss']:.4f} | "
              f"val_F1 {val_m['f1']:.4f} | "
              f"val_acc {val_m['accuracy']:.4f} | "
              f"{ep_time:.0f}s")

        history['train_loss'].append(train_m['loss'])
        history['val_loss'].append(val_m['loss'])
        history['train_acc'].append(train_m['accuracy'])
        history['val_acc'].append(val_m['accuracy'])
        history['val_f1'].append(val_m['f1'])

        early_stop(val_m['f1'], model)
        if early_stop.should_stop:
            print(f"  ✓ Early stopping at epoch {epoch+1} "
                  f"(best val F1 = {early_stop.best_score:.4f})")
            break

    total_time = time.time() - t0

    # ── Restore best weights ─────────────────────────────────────────────
    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)

    # ── Test evaluation ──────────────────────────────────────────────────
    test_m = validate(model, loaders['test'], criterion_task, device)
    print(f"\n  TEST  — F1 {test_m['f1']:.4f} | "
          f"Acc {test_m['accuracy']:.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    ckpt = {
        'model_state_dict': model.state_dict(),
        'condition':        condition,
        'use_adversarial':  use_adversarial,
        'architecture':     config['architecture'],
        'config':           config,
        'best_val_f1':      early_stop.best_score,
        'total_time':       total_time,
    }
    ckpt_path = os.path.join(MODELS_DIR, _checkpoint_basename(condition, config['architecture']))
    torch.save(ckpt, ckpt_path)
    # Compatibility alias for downstream scripts that still expect old filenames.
    legacy_ckpt_path = os.path.join(MODELS_DIR, f"efficientnet_{condition}.pth")
    torch.save(ckpt, legacy_ckpt_path)
    print(f"  Model saved → {ckpt_path}")
    print(f"  Legacy alias → {legacy_ckpt_path}")

    return {
        'model':           model,
        'condition':       condition,
        'use_adversarial': use_adversarial,
        'history':         history,
        'test_metrics':    test_m,
        'total_time':      total_time,
        'best_val_f1':     early_stop.best_score,
        'config':          config,
        'architecture':    config['architecture'],
    }


# ─── Load model from saved checkpoint (eval-only mode) ──────────────────────
def load_model_from_checkpoint(
    condition: str,
    loaders: dict,
    device=None,
) -> dict:
    """
    Load a previously trained model from disk and return a results dict
    with the same structure as train_model(), so run_full_evaluation() works
    without re-training.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    clip_ckpt = os.path.join(MODELS_DIR, _checkpoint_basename(condition, 'clip_vit_b32'))
    legacy_ckpt = os.path.join(MODELS_DIR, f"efficientnet_{condition}.pth")
    ckpt_path = clip_ckpt if os.path.exists(clip_ckpt) else legacy_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {clip_ckpt} or {legacy_ckpt}\n"
            f"Run without --eval-only to train first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    use_adversarial = ckpt.get('use_adversarial', condition == 'cf')
    cfg = ckpt.get('config', DEFAULT_CONFIG.copy())

    model = create_model(
        use_adversarial=use_adversarial,
        n_groups=N_GROUPS,
        dropout=cfg.get('dropout', DEFAULT_CONFIG['dropout']),
        freeze_blocks=cfg.get('freeze_blocks', DEFAULT_CONFIG['freeze_blocks']),
        architecture=ckpt.get('architecture', cfg.get('architecture', DEFAULT_CONFIG['architecture'])),
        freeze_encoder=cfg.get('freeze_encoder', DEFAULT_CONFIG['freeze_encoder']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded checkpoint: {ckpt_path}  "
          f"(val F1={ckpt.get('best_val_f1', '?'):.4f}  "
          f"adv={'ON' if use_adversarial else 'OFF'})")

    # Dummy history (no training data available from saved checkpoint)
    dummy_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'val_f1':     [],
    }

    return {
        'model':           model,
        'condition':       condition,
        'use_adversarial': use_adversarial,
        'history':         dummy_history,
        'test_metrics':    {},
        'total_time':      ckpt.get('total_time', 0.0),
        'best_val_f1':     ckpt.get('best_val_f1', None),
        'config':          cfg,
        'architecture':    ckpt.get('architecture', cfg.get('architecture', DEFAULT_CONFIG['architecture'])),
    }


# ─── CLI quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Use run_all.py to execute the full pipeline.")

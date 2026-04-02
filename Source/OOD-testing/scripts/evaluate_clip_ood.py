import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OOD_ROOT = PROJECT_ROOT / 'OOD-testing'
DATA_DIR = OOD_ROOT / 'data'
RESULTS_DIR = OOD_ROOT / 'results'

class LocalImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.image_paths = self.df['resolved_image_path'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': row['label_binary'],
            'id': row['sample_id']
        }


def resolve_image_path(row):
    """Resolve image path robustly, preserving subdirectories like img/16395.png."""
    candidates = []

    if 'image_path' in row and pd.notna(row['image_path']):
        candidates.append(Path(row['image_path']))

    if 'hf_img_relpath' in row and pd.notna(row['hf_img_relpath']):
        rel = Path(str(row['hf_img_relpath']))
        candidates.append(DATA_DIR / 'hateful_memes_images' / rel)
        candidates.append(DATA_DIR / 'hateful_memes_images' / rel.name)

    for cand in candidates:
        if cand.exists():
            return str(cand)

    return None


def build_eval_transform():
    """Match training-time CLIP evaluation preprocessing exactly."""
    clip_mean = [0.48145466, 0.45782750, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    nh = (y_true == 0)
    h = (y_true == 1)
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'f1': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'auc_roc': round(float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan'), 4),
        'fpr': round(float((y_pred[nh] == 1).sum() / max(int(nh.sum()), 1)), 4),
        'fnr': round(float((y_pred[h] == 0).sum() / max(int(h.sum()), 1)), 4),
    }

def main():
    from image_models.model import create_model

    eval_transforms = build_eval_transform()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading dataset...')

    df = pd.read_csv(DATA_DIR / 'hateful_memes_test_ood.csv')
    df['resolved_image_path'] = df.apply(resolve_image_path, axis=1)
    missing_mask = df['resolved_image_path'].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0:
        print(f'WARNING: {n_missing}/{len(df)} images could not be resolved. Dropping missing rows.')
        df = df.loc[~missing_mask].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError('No resolvable images found in hateful_memes_test_ood.csv.')

    dataset = LocalImageDataset(df=df, transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    y_true = df['label_binary'].to_numpy(dtype=int)

    print(f'Using {len(df)} samples for evaluation.')
    
    conditions = {
        'nCF': 'clip_vitb32_ncf.pth',
        'CF-no-adv': 'clip_vitb32_cf_no_adv.pth',
        'CF+GRL': 'clip_vitb32_cf.pth'
    }

    out = {}
    for cond, name in conditions.items():
        print(f'Evaluating {cond}...')
        ckpt_path = PROJECT_ROOT / 'image_models' / 'models' / name
        if not ckpt_path.exists():
            print(f'Skipping {cond}, {ckpt_path} not found.')
            continue
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            cfg = ckpt.get('config', {})
            use_adv = bool(ckpt.get('use_adversarial', cond == 'CF+GRL'))
            state_dict = ckpt['model_state_dict']
        else:
            use_adv = (cond == 'CF+GRL')
            state_dict = ckpt
            
        model = create_model(
            use_adversarial=use_adv,
            n_groups=8,
            architecture='clip_vit_b32',
            device=device
        )
        
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
            
        model.eval()

        probs = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                logits, _ = model(batch['image'].to(device))
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p.tolist())

        probs = np.array(probs)
        mets = compute_binary_metrics(y_true, probs)
        out[cond] = mets
        print(f'Results for {cond}: {mets}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'clip_ood_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()

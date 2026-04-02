import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

root = Path('OOD-testing/results')
out = {}
for cond in ['ncf','cf']:
    df = pd.read_csv(root / f'text_ood_predictions_{cond}.csv')
    y_true = df['label_binary'].astype(int)
    y_prob = df['pred_prob'].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    nh = (y_true == 0)
    h = (y_true == 1)
    out[cond] = {
        'n_samples': int(len(df)),
        'metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else float('nan'),
            'fpr': float((y_pred[nh] == 1).sum() / max(int(nh.sum()),1)),
            'fnr': float((y_pred[h] == 0).sum() / max(int(h.sum()),1)),
            'threshold': 0.5,
        }
    }

(root / 'text_ood_metrics.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))

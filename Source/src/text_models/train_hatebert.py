import json
import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

from text_models.enhanced_analysis import load_condition

def _safe_div(n, d):
    return float(n / d) if d > 0 else float("nan")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    
    mac_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    fpr_calc = 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = _safe_div(fp, fp + tn)
    
    return {
        "macro_f1": float(mac_f1),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "fpr": float(fpr),
        "auc": float(roc_auc_score(labels, probs[:, 1]))
    }

def finetune_hatebert(cond: str):
    print(f"\n========== Fine-Tuning HateBERT on [{cond.upper()}] ==========")
    
    # 1. Load Data
    splits = load_condition(cond)
    
    tokenizer = AutoTokenizer.from_pretrained('GroNLP/hateBERT')
    model = AutoModelForSequenceClassification.from_pretrained('GroNLP/hateBERT', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 2. Convert to HF Datasets
    def to_hg_ds(df):
        texts = df["text"].fillna("").tolist()
        labels = df["binary_label"].tolist()
        return Dataset.from_dict({"text": texts, "labels": labels})
    
    ds_train = to_hg_ds(splits["train"])
    ds_val = to_hg_ds(splits["val"])
    ds_test = to_hg_ds(splits["test"])
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=128)
    
    tk_train = ds_train.map(tokenize_function, batched=True)
    tk_val = ds_val.map(tokenize_function, batched=True)
    tk_test = ds_test.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 3. Setup Trainer
    out_dir = f"text_models/models/hatebert_finetune_{cond}"
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tk_train,
        eval_dataset=tk_val,
        processing_class=tokenizer, # For newer transformers use processing_class
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 4. Train
    print(f"Starting Trainer.train() for {cond.upper()}... GPU: {torch.cuda.is_available()}")
    trainer.train()
    
    # 5. Get Probs for Val, Test for thresholding
    val_preds = trainer.predict(tk_val)
    test_preds = trainer.predict(tk_test)
    
    y_val = val_preds.label_ids
    y_prob_val = torch.softmax(torch.tensor(val_preds.predictions), dim=-1).numpy()[:, 1]
    
    y_test = test_preds.label_ids
    y_prob_test = torch.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()[:, 1]
    
    # Thresholding
    thresholds = np.linspace(0.1, 0.9, 161)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred_val = (y_prob_val >= t).astype(int)
        f = f1_score(y_val, pred_val, average="macro", zero_division=0)
        if f > best_score:
            best_score, best_t = f, t
            
    # Apply to Test
    y_pred_def = (y_prob_test >= 0.5).astype(int)
    y_pred_opt = (y_prob_test >= best_t).astype(int)
    
    def _get_cm_metrics(yt, yp, probs):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "fpr": float(_safe_div(fp, fp + tn)),
            "fnr": float(_safe_div(fn, fn + tp)),
            "auc": float(roc_auc_score(yt, probs)),
        }
    
    res_def = _get_cm_metrics(y_test, y_pred_def, y_prob_test)
    res_opt = _get_cm_metrics(y_test, y_pred_opt, y_prob_test)
    
    print(f"--- {cond.upper()} Test Results ---")
    print(f"Default (t=0.5): F1={res_def['f1']:.4f}, FPR={res_def['fpr']:.4f}, AUC={res_def['auc']:.4f}")
    print(f"Optimal (t={best_t:.3f}): F1={res_opt['f1']:.4f}, FPR={res_opt['fpr']:.4f}")
    
    return {
        "name": f"HateBERT_Finetuned_{cond}",
        "condition": cond,
        "default_f1": res_def["f1"],
        "default_fpr": res_def["fpr"],
        "default_fnr": res_def["fnr"],
        "opt_f1": res_opt["f1"],
        "opt_fpr": res_opt["fpr"],
        "opt_fnr": res_opt["fnr"],
        "roc_auc": res_def["auc"],
        "opt_threshold": float(best_t),
        "test_df": splits["test"]
    }

def main():
    print("BEGIN HATEBERT FINETUNING (PyTorch End-to-End)")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # 1. Finetune NCF
    res_ncf = finetune_hatebert("ncf")
    
    # 2. Finetune CF
    res_cf = finetune_hatebert("cf")
    
    # 3. Print Comparison
    print("\n\n==========================================")
    print("      FINAL COMPARISON (End-to-End)")
    print("==========================================")
    print(f"HateBERT NCF (Opt): F1 = {res_ncf['opt_f1']:.4f}  | FPR = {res_ncf['opt_fpr']:.4f}")
    print(f"HateBERT CF  (Opt): F1 = {res_cf['opt_f1']:.4f}  | FPR = {res_cf['opt_fpr']:.4f}")
    print("==========================================")

if __name__ == "__main__":
    main()

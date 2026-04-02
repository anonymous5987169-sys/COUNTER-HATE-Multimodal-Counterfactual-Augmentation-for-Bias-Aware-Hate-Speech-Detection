# EXECUTION ROADMAP: Get CLIP Training Started Now

**Objective**: Start CLIP ViT-B/32 training and get results ready for comparison  
**Timeline**: 8-12 hours of training + 1 hour for analysis and reporting  
**Effort**: Choose one path below, execute command, monitor

---

## 🎯 CHOOSE YOUR EXECUTION PATH

### ✅ RECOMMENDED: Path A — Background Job (Most Reliable)
**Why**: Immune to terminal timeout, completely hands-off, auto-monitoring

**Command** (copy-paste into PowerShell):
```powershell
.\run_clip_background.ps1 -epochs 20 -patience 5 -device cpu
```

**What it does**:
1. Starts CLIP training in background PowerShell job
2. Trains all 3 conditions sequentially (nCF → CF-no-adv → CF)
3. Auto-monitors progress every 5 minutes
4. Shows completion status and checkpoint verification
5. You can close terminal and come back later

**Timeline**: ~12 hours (starts immediately, runs overnight)

**Status check** (if you close terminal):
```powershell
Get-Job | Select-Object Id, Name, State, PSEndTime
Get-Job -Id <ID> | Receive-Job  # see output
```

**Best for**: Users who can leave machine running 8-12 hours

---

### 🔄 ALTERNATIVE: Path B — Sequential Control (More Flexible)
**Why**: Structured, explicit control over each condition, better error isolation

**Command 1** (run first):
```powershell
python train_clip_sequential.py --condition nCF --epochs 20 --patience 5
```
→ Wait ~3 hours for completion

**Command 2** (run after Command 1 finishes):
```powershell
python train_clip_sequential.py --condition CF_no_adv --epochs 20 --patience 5
```
→ Wait ~4 hours for completion

**Command 3** (run after Command 2 finishes):
```powershell
python train_clip_sequential.py --condition CF --epochs 20 --patience 5
```
→ Wait ~4 hours for completion

**What it does**:
- Trains one condition at a time with error checking
- Shows checkpoint verification after each condition
- Can pause/resume between runs if needed

**Timeline**: Same as Path A (~12 hours) but split into 3 separate runs

**Best for**: Users with specific preferences for condition ordering or ability to check progress between runs

---

### ⚡ QUICK TEST: Path C — Validation Only (Testing)
**Why**: Verify code works before committing to full 20-epoch run

**Command**:
```powershell
python image_models/run_all.py --epochs 10 --patience 5 --device cpu
```

**Timeline**: ~5 hours

**⚠️ Important**: This produces incomplete metrics (only 10 epochs). Use for:
- Testing that CLIP forward pass works
- Quick sanity check that no code errors exist
- Then restart with Path A/B for full 20-epoch training

**Best for**: Users unsure if code will work, want quick validation before full run

---

## 📋 STEP-BY-STEP EXECUTION

### BEFORE YOU START (5 minutes)
- [ ] Read this entire document
- [ ] Verify Python 3.12: `python --version` (should be Python 3.12.x)
- [ ] Check disk space: ~5 GB free
- [ ] Choose one of the 3 paths above
- [ ] Close other applications to free CPU

### STEP 1: START TRAINING (1 minute)
Execute your chosen command:

**Path A**:
```powershell
.\run_clip_background.ps1 -epochs 20 -patience 5 -device cpu
```

**Path B**:
```powershell
python train_clip_sequential.py --condition nCF --epochs 20 --patience 5
```

**Path C**:
```powershell
python image_models/run_all.py --epochs 10 --patience 5 --device cpu
```

**Expected output** (first 30 seconds):
```
Downloading openai/clip-vit-base-patch32 from Hugging Face Hub...
Loading data with canonical splits...
Creating CLIP ViT-B/32 classifier (frozen encoder)...
Starting training (max 20 epochs)...
Epoch 1/20: 100%|████████████| 130/130 [02:15<00:00, 1.04s/batch]
...
```

✓ **If you see this**: Training started successfully

✗ **If you see errors**: Check troubleshooting section below

### STEP 2: MONITOR PROGRESS (Periodic checks)

**For Path A** (background job):
```powershell
# Check status every 1-2 hours
Get-Job | Select-Object State
# When done, should show "Completed"

# See output anytime
Get-Job | Receive-Job
```

**For Path B** (sequential):
- Watch terminal for progress bars
- Wait for "✓ Training complete" message
- Then run next condition command

**For Path C** (quick test):
- Watch terminal for progress
- Takes ~5 hours

**What to expect**:
- Progress bar for each batch (130 batches per epoch for ~3-4 sec each)
- Full epoch takes 5-10 minutes on CPU
- 11-20 epochs until early stopping triggers
- Total: 2-4 hours per condition

### STEP 3: VERIFY COMPLETION (5 minutes)

**Check checkpoints exist**:
```powershell
ls image_models/models/clip_vitb32_*.pth
```

**Expected output**:
```
clip_vitb32_nCF.pth        30 MB
clip_vitb32_CF_no_adv.pth  30 MB
clip_vitb32_CF.pth         30 MB
```

✓ **If all 3 exist**: Training successful

✗ **If missing**: Check error log (see troubleshooting)

### STEP 4: GENERATE METRICS (30 minutes)

```powershell
python image_models/evaluate.py
```

**Expected output**:
```
Evaluating nCF condition...
Evaluating CF_no_adv condition...
Evaluating CF condition...
✓ Metrics saved to image_models/results/evaluation_results.json
```

### STEP 5: COMPARE ARCHITECTURES (5 minutes)

```powershell
python scripts/compare_architectures.py
```

**Expected output**:
```
✓ Comparison written to CLIP_vs_EFFICIENTNET_<timestamp>.md
✓ CLIP nCF F1: 0.78 (vs EfficientNet 0.7809) → Δ= -0.3%
✓ CLIP CF-no-adv F1: 0.81 (vs EfficientNet 0.8080) → Δ= +0.8%
✓ CLIP CF+GRL F1: 0.79 (vs EfficientNet 0.7885) → Δ= +0.2%
```

### STEP 6: ASSESS RESULTS (Quick decision)

Check if all F1 scores are in acceptable range [0.74–0.85]:

**✓ All within range?**
→ CLIP is working! Proceed to Step 7 (report updates)

**✗ Any F1 < 0.74?**
→ Apply tuning from `EFFICIENTNET_vs_CLIP_COMPARISON.md`
→ Estimated additional time: 4-12 hours

### STEP 7: UPDATE REPORTS (1 hour)

Edit these files and add CLIP results next to EfficientNet metrics:

**File 1: prof-report.md**
- Find: Architecture results table
- Add: CLIP F1, AUC, FPR for each condition

**File 2: Architectures.md**
- Find: Architecture specifications section
- Add: CLIP ViT-B/32 specs (dims, pretraining, freezing info)

**File 3: results.md**
- Find: Per-condition analysis sections
- Add: CLIP vs EfficientNet deltas and fairness comparison

---

## 🛠️ QUICK TROUBLESHOOTING

| Problem | Cause | Solution |
|---|---|---|
| **"No such file or directory: run_clip_background.ps1"** | Script not in current directory | `cd c:\Users\gritw\Downloads\major-project\major-project` then retry |
| **"Python not found"** | Python not in PATH | Use full path: `C:/Users/gritw/AppData/Local/Programs/Python/Python312/python.exe train_clip_sequential.py` |
| **KeyboardInterrupt after 5% of epoch 1** | Terminal timeout | Use Path A (background job) instead of Path B/C |
| **Training starts but slows down rapidly** | CPU under load | Close other applications, reduce system tasks |
| **"CUDA out of memory"** (you're not using CUDA) | Not applicable to this setup | Ignore, CPU training is default |
| **Training says "complete" but no checkpoint file** | Crashed silently | Check terminal output for error messages (often at end) |

**For more troubleshooting**: See `CLIP_TRAINING_RESTART_GUIDE.md` → "Troubleshooting"

---

## ⏱️ TIME BREAKDOWN

### Training Phase
| Condition | Time | Start | End |
|---|---|---|---|
| nCF | 2-3 hrs | T+0 | T+3 |
| CF-no-adv | 3-4 hrs | T+3 | T+7 |
| CF+GRL | 3-4 hrs | T+7 | T+11 |
| **Total training** | **8-12 hrs** | Now | In 12 hrs |

### Analysis Phase (After training)
| Task | Time | Command |
|---|---|---|
| Evaluation metrics | 30 min | `python image_models/evaluate.py` |
| Comparison analysis | 5 min | `python scripts/compare_architectures.py` |
| Review results | 10 min | Read comparison markdown |
| Report updates | 1 hr | Edit prof-report.md, Architectures.md, results.md |
| **Total analysis** | **~2 hrs** | After training |

**Total time needed**: ~14 hours (12 hrs training + 2 hrs analysis)

---

## 📊 SUCCESS CRITERIA

### Minimum (Must Have)
- ✓ Training completes without interruption for all 3 conditions
- ✓ 3 checkpoint files created: `clip_vitb32_{nCF,CF_no_adv,CF}.pth`
- ✓ Evaluation metrics generated: `evaluation_results.json` updated
- ✓ Comparison table created showing CLIP vs EfficientNet

### Target (Ideal)
- ✓ All above
- ✓ F1 scores within [0.74–0.85] (no tuning needed)
- ✓ EO-diff improved vs baseline (fairness better)
- ✓ Reports updated with final metrics and analysis

### If Underperforming
- ✓ Apply tuning from `EFFICIENTNET_vs_CLIP_COMPARISON.md`
- ✓ Rerun training with tuned parameters
- ✓ Verify improvement
- ✓ Update reports

---

## 🚀 THE COMMAND (Just Copy-Paste)

### If you're ready to start RIGHT NOW:

```powershell
# Path A - Recommended (recommended for first-time)
.\run_clip_background.ps1 -epochs 20 -patience 5 -device cpu
```

**What happens next**:
1. PowerShell background job starts
2. CLIP training initializes (shows "Downloading openai/clip-vit-base-patch32...")
3. Training progresses (shows epoch 1/20, 2/20, etc.)
4. Auto-monitoring shows status every 5 minutes
5. Completes in ~12 hours
6. Shows checkpoint verification at end

**Then when it's done**:
```powershell
python image_models/evaluate.py
python scripts/compare_architectures.py
```

---

## 📞 HELP & REFERENCES

- **Execution guide**: `CLIP_TRAINING_RESTART_GUIDE.md`
- **Performance expectations**: `EFFICIENTNET_vs_CLIP_COMPARISON.md`
- **Tuning playbook**: `EFFICIENTNET_vs_CLIP_COMPARISON.md` → "Tuning Playbook"
- **Code reference**: Model in `image_models/model.py`, training in `image_models/train.py`

---

## ✨ SUMMARY

You have:
- ✓ Complete CLIP ViT-B/32 implementation (integrated into all pipeline stages)
- ✓ 3 execution paths to choose from
- ✓ Full documentation and troubleshooting guides
- ✓ Tuning playbook if needed
- ✓ Comparison framework ready for metric population

**Next action**: Choose one of the 3 paths above and execute the command.

**Estimated time to final results**: 14 hours (12 hrs training + 2 hrs analysis/reporting)

---

**Ready? Execute this now**:
```powershell
.\run_clip_background.ps1 -epochs 20 -patience 5 -device cpu
```

Check back in 12 hours. 🚀

#!/usr/bin/env bash
set -euo pipefail

cd /home/vslinux/Documents/research/major-project
PY=/home/vslinux/.pyenv/versions/3.12.0/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "Post-image rerun pipeline started"

if [[ -f image_models.pid ]]; then
  IMG_PID="$(cat image_models.pid)"
  if ps -p "$IMG_PID" > /dev/null 2>&1; then
    log "Waiting for image pipeline PID $IMG_PID to complete"
    while ps -p "$IMG_PID" > /dev/null 2>&1; do
      sleep 30
    done
    log "Image pipeline completed"
  else
    log "Image PID file exists but process is not running; continuing"
  fi
else
  log "No image_models.pid found; continuing"
fi

log "Running Step 4: late fusion ensemble"
"$PY" cross_modal/late_fusion_ensemble.py

log "Running Step 4: ablation calibration study"
"$PY" cross_modal/ablation_calibration_study.py

log "Running Step 4: cross-modal orchestrator"
"$PY" cross_modal/run_all.py

log "Running Step 4 sanity checks"
"$PY" - <<'PY'
import json
from pathlib import Path

root = Path('/home/vslinux/Documents/research/major-project')
req = [
    root/'cross_modal/results/late_fusion_results.json',
    root/'cross_modal/results/stacking_ensemble_results.json',
    root/'cross_modal/results/learned_fusion_results.json',
    root/'cross_modal/results/cross_attention_fusion_results.json',
]
for p in req:
    if not p.exists():
        raise SystemExit(f'MISSING: {p}')

required_labels = {'nCF', 'CF-no-adv', 'CF+GRL'}

# late fusion
late = json.loads((root/'cross_modal/results/late_fusion_results.json').read_text())
conds = {row.get('condition') for row in late.get('table6_consolidated', [])}
if not required_labels.issubset(conds):
    raise SystemExit(f'late_fusion missing conditions: {required_labels - conds}')

# stacking
stack = json.loads((root/'cross_modal/results/stacking_ensemble_results.json').read_text())
conds = {row.get('condition') for row in stack.get('summary_table', [])}
if not required_labels.issubset(conds):
    raise SystemExit(f'stacking missing conditions: {required_labels - conds}')

# learned
learned = json.loads((root/'cross_modal/results/learned_fusion_results.json').read_text())
conds = {row.get('condition') for row in learned.get('summary_table', [])}
if not required_labels.issubset(conds):
    raise SystemExit(f'learned missing conditions: {required_labels - conds}')

# cross-attn
xattn = json.loads((root/'cross_modal/results/cross_attention_fusion_results.json').read_text())
conds = {row.get('condition') for row in xattn.get('summary_table', [])}
if not required_labels.issubset(conds):
    raise SystemExit(f'cross_attention missing conditions: {required_labels - conds}')


def has_bad(value):
    if isinstance(value, dict):
        return any(has_bad(v) for v in value.values())
    if isinstance(value, list):
        return any(has_bad(v) for v in value)
    if isinstance(value, float):
        return value != value
    return False

for p in req:
    obj = json.loads(p.read_text())
    if has_bad(obj):
        raise SystemExit(f'NaN detected in: {p}')

print('sanity-check: OK')
PY

log "Running Step 5: analysis (skip text/image/cross-modal retrain)"
"$PY" analysis/run_all.py --skip-text --skip-image --skip-cross-modal

log "Running Step 7: generate all plots"
"$PY" scripts/generate_all_plots.py

log "Running Step 8: tests"
"$PY" -m pytest tests/ -v --tb=short -x

log "Post-image rerun pipeline completed successfully"

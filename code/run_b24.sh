#!/usr/bin/env bash
set -euo pipefail

RUN="${B24_RUN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PARENT="${B24_PARENT_ROOT:?Set B24_PARENT_ROOT to the external parent-model directory}"
CODE="${RUN}"
PYTHON="${B24_PYTHON:-python}"
CHECKPOINTS=${RUN}/checkpoints
VALIDATION=${RUN}/validation
OUTPUT=${RUN}/results
LOGS=${RUN}/logs
TEACHER="${B24_TEACHER_CKPT:?Set B24_TEACHER_CKPT to the external teacher checkpoint}"

mkdir -p "${CHECKPOINTS}" "${VALIDATION}" "${OUTPUT}" "${LOGS}"
cd "${RUN}"
trap 'date --iso-8601=seconds > pipeline_failed.txt' ERR

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${RUN}:${CODE}
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

BASELINE=${PARENT}/checkpoints/v12/iterativepfn_best.pkl
BASELINE_ARGS=${PARENT}/checkpoints/v12/args.json
V29=${PARENT}/checkpoints/v29/v29_best.pkl
V33=${PARENT}/checkpoints/v33/v33_best.pkl
V41=${PARENT}/checkpoints/v41/v41_best_active.pkl
V45=${PARENT}/checkpoints/v45/v45_best_active.pkl
V65=${PARENT}/checkpoints/v65/v65_best_active.pkl
ELIGIBLE=${RUN}/eligible_checkpoints.json
MANIFEST=${RUN}/validation_manifest.json

common=(
  --baseline_ckpt "${BASELINE}" --baseline_args "${BASELINE_ARGS}"
  --v29_ckpt "${V29}" --v33_ckpt "${V33}"
  --v41_ckpt "${V41}" --v45_ckpt "${V45}" --v65_ckpt "${V65}"
  --teacher_ckpt "${TEACHER}"
  --parent_strength 1.25 --v45_strength 1.25 --v65_strength 1.50
  --seed 8242401
)

EXPECTED_TEACHER_SHA256=${B24_EXPECTED_TEACHER_SHA256:-511ccd594c81cccb1ab769fd52c3049e6c316375ddce91b89204f86fce0ab8ef}
if [[ "$(sha256sum "${TEACHER}" | awk '{print $1}')" != "${EXPECTED_TEACHER_SHA256}" ]]; then
  echo "B24 immutable B20 C4 teacher hash mismatch" >&2
  exit 1
fi

rm -f pipeline_failed.txt pipeline_complete.txt training_finished_before_test_access
rm -f selection_locked.json test_access_allowed.json result.zip result.zip.sha256
rm -f selected_batch.txt batch_selection.json batch_probe_*.json
rm -rf "${CHECKPOINTS}" "${VALIDATION}" "${OUTPUT}"
mkdir -p "${CHECKPOINTS}" "${VALIDATION}" "${OUTPUT}" "${LOGS}"
date --iso-8601=seconds > pipeline_started.txt
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > gpu_start.txt

probe() {
  local batch=$1
  timeout --signal=TERM --kill-after=5m 75m "${PYTHON}" "${RUN}/benchmark_b24.py" \
    --output "${RUN}/batch_probe_${batch}.json" \
    --data_root /root --train_list /root/datalist/train_b.txt \
    --batch_size "${batch}" --num_workers 8 --steps 50 \
    "${common[@]}" > "${LOGS}/batch_probe_${batch}.log" 2>&1
}

if probe 6; then
  MEMORY6=$("${PYTHON}" -c "import json; print(json.load(open('${RUN}/batch_probe_6.json'))['gpu_memory_mib'])")
  if [[ "${MEMORY6}" -lt 20000 ]]; then
    probe 8 || rm -f batch_probe_8.json
  fi
else
  rm -f batch_probe_6.json
  if ! probe 4; then
    rm -f batch_probe_4.json
    probe 3
  fi
fi
"${PYTHON}" "${RUN}/select_batch_b24.py" | tee "${LOGS}/batch_selection.log"
BATCH=$(cat selected_batch.txt)

timeout --signal=TERM --kill-after=10m 360m "${PYTHON}" "${RUN}/preflight_b24.py" \
  --run_root "${RUN}" --data_root /root \
  --train_list /root/datalist/train_b.txt --val_list /root/datalist/validate_b.txt \
  --batch_size "${BATCH}" --num_workers 8 --patch_batch 6 \
  --full_cloud_audits 4 --reproduction_only 1 --conflict_steps 0 --smoke_steps 0 \
  "${common[@]}" 2>&1 | tee "${LOGS}/reproduction.log"

# Run gradient checks in a fresh process so full-cloud Jittor caches have been
# returned to the driver before the 200-batch optimizer smoke.
timeout --signal=TERM --kill-after=10m 360m "${PYTHON}" "${RUN}/preflight_b24.py" \
  --run_root "${RUN}" --data_root /root \
  --train_list /root/datalist/train_b.txt --val_list /root/datalist/validate_b.txt \
  --batch_size "${BATCH}" --num_workers 8 --patch_batch 6 \
  --full_cloud_audits 0 --reproduction_only 0 --conflict_steps 50 --smoke_steps 200 \
  "${common[@]}" 2>&1 | tee "${LOGS}/preflight.log"

# This watchdog covers only Stage0 + pure optimization. The configured pure
# Stage A/B/C target is 13.5 h and the summed stage hard limit is 15.0 h.
timeout --signal=TERM --kill-after=10m 930m "${PYTHON}" "${RUN}/train_b24.py" \
  --run_root "${RUN}" --save_dir "${CHECKPOINTS}" --eligible_json "${ELIGIBLE}" \
  --data_root /root --train_list /root/datalist/train_b.txt \
  --batch_size "${BATCH}" --num_workers 8 --num_points 32768 \
  --patch_size 1000 --patch_ratio 1.2 --alignment_k 32 \
  --stage0_batches 256 \
  --stage_a_hours 2.5 --stage_a_hard 2.8 \
  --stage_b_hours 8.5 --stage_b_hard 9.4 --stage_b_midpoint 4.3 \
  --stage_c_hours 2.5 --stage_c_hard 2.8 \
  --stage_a_lr 4e-5 --stage_a_min_lr 5e-6 \
  --stage_b_lr 2.5e-5 --stage_b_min_lr 5e-7 \
  --stage_c_lr 7e-6 --stage_c_min_lr 2e-7 \
  --weight_decay_a 2e-6 --weight_decay_bc 3e-6 \
  --log_interval 100 "${common[@]}" 2>&1 | tee "${LOGS}/train.log"

date --iso-8601=seconds > training_finished_before_test_access
"${PYTHON}" "${RUN}/validate_b24.py" \
  --run_root "${RUN}" --eligible_json "${ELIGIBLE}" --manifest "${MANIFEST}" \
  --data_root /root --val_list /root/datalist/validate_b.txt \
  --patch_size 1000 --seed_k 6 --beta 12 --patch_batch 6 \
  --surface_count 250000 "${common[@]}" 2>&1 | tee "${LOGS}/validation.log"

"${PYTHON}" "${RUN}/infer_b24.py" \
  --run_root "${RUN}" --data_root /root --test_list /root/datalist/test_b.txt \
  --output_root "${OUTPUT}" --patch_size 1000 --seed_k 6 --beta 12 \
  --patch_batch 6 --resume 1 "${common[@]}" 2>&1 | tee "${LOGS}/predict.log"

"${PYTHON}" "${RUN}/validate_submission.py" \
  --data_root /root --test_list /root/datalist/test_b.txt --output_root "${OUTPUT}" \
  2>&1 | tee "${LOGS}/validate_submission.log"
"${PYTHON}" -c "import shutil; shutil.make_archive('${RUN}/result', 'zip', '${OUTPUT}', 'dataset_test_noisy')"
"${PYTHON}" "${RUN}/validate_submission.py" \
  --data_root /root --test_list /root/datalist/test_b.txt --output_root "${OUTPUT}" \
  --zip_path "${RUN}/result.zip" 2>&1 | tee -a "${LOGS}/validate_submission.log"
sha256sum result.zip > result.zip.sha256

tar --exclude='./parents' --exclude='./checkpoints' --exclude='./validation' \
  --exclude='./results' --exclude='./logs' --exclude='./source_snapshot.tar.gz' \
  -czf source_snapshot.tar.gz ./*.py ./*.sh ./*.json ./*.md 2>/dev/null || true
date --iso-8601=seconds > pipeline_complete.txt
printf 'B24_PIPELINE_COMPLETE active_result=READY\n'

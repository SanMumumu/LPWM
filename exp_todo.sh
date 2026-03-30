#!/usr/bin/env bash
set -euo pipefail

# Flow-LPWM experiment runner
# Included experiments:
# - Sketchy-U
# - Sketchy-A
# - BAIR-U
# - Mario-U
# - Bridge-L
# - PandaPush
# - OGBench-Scene
#
# Epoch counts follow LPWM Table 5:
# - Sketchy: 18
# - BAIR: 20
# - Mario: 200
# - Bridge: 42
# - PandaPush: 84
# - OGBench: 80

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"
ACCEL_CFG="${ACCEL_CFG:-./accel_conf.yml}"

RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_EVAL_AND_VIS="${RUN_EVAL_AND_VIS:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-0}"
USE_MULTI_GPU="${USE_MULTI_GPU:-1}"
LPWM_ACCELERATE_GPUS="${LPWM_ACCELERATE_GPUS:-0,1,2,3,4,5,6,7}"
ALLOW_FVD_DOWNLOAD="${ALLOW_FVD_DOWNLOAD:-1}"

export LPWM_ACCELERATE_GPUS

CFG_SKETCHY_U="./configs/sketchy_u_flow_exp.json"
CFG_SKETCHY_A="./configs/sketchy_a_flow_exp.json"
CFG_BAIR_U="./configs/bair_u_flow_exp.json"
CFG_MARIO_U="./configs/mario_u_flow_exp.json"
CFG_BRIDGE_L="./configs/bridge_l_flow_exp.json"
CFG_PANDAPUSH="./configs/pandapush_flow_exp.json"
CFG_OGBENCH_SCENE="./configs/ogbench_scene_flow_exp.json"

SUMMARY_CSV="${SUMMARY_CSV:-./flow_exp_summary.csv}"
SUMMARY_MD="${SUMMARY_MD:-./flow_exp_summary.md}"
BASELINE_TEMPLATE_CSV="${BASELINE_TEMPLATE_CSV:-./baseline_comparison_template.csv}"
BASELINE_TEMPLATE_MD="${BASELINE_TEMPLATE_MD:-./baseline_comparison_template.md}"
FVD_WEIGHTS="${FVD_WEIGHTS:-./eval/fvd/fvd/styleganv/i3d_torchscript.pt}"

latest_run_dir() {
  local pattern="$1"
  ls -td ./*"${pattern}"* 2>/dev/null | head -n 1 || true
}

latest_checkpoint() {
  local run_dir="$1"
  ls -t "${run_dir}"/saves/*_best_lpips.pth 2>/dev/null | head -n 1 || \
  ls -t "${run_dir}"/saves/*_best.pth 2>/dev/null | head -n 1 || \
  ls -t "${run_dir}"/saves/*.pth 2>/dev/null | head -n 1 || true
}

require_dir() {
  local path="$1"
  if [[ -z "${path}" || ! -d "${path}" ]]; then
    echo "Missing run directory: ${path}" >&2
    exit 1
  fi
  echo "${path}"
}

require_file() {
  local path="$1"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "Missing file: ${path}" >&2
    exit 1
  fi
  echo "${path}"
}

train_lpwm_cmd() {
  local cfg="$1"
  if [[ "${USE_MULTI_GPU}" == "1" ]]; then
    "${ACCELERATE}" launch --config_file "${ACCEL_CFG}" train_lpwm_accelerate.py -d "${cfg}"
  else
    "${PYTHON}" train_lpwm.py -d "${cfg}"
  fi
}

run_one_experiment() {
  local name="$1"
  local cfg="$2"
  echo "[train:${name}] cfg=${cfg} gpus=${LPWM_ACCELERATE_GPUS}"
  train_lpwm_cmd "${cfg}"
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dst}"
  fi
}

run_video_suite() {
  local name="$1"
  local run_dir="$2"
  local ckpt="$3"
  local cond_steps="$4"
  local horizon="$5"
  local batch="$6"
  local run_ctx_eval="${7:-1}"
  local eval_dir="${run_dir}/eval"

  mkdir -p "${eval_dir}"
  echo "[eval:${name}] videos deterministic"
  "${PYTHON}" generate_lpwm_video_prediction.py -p "${run_dir}" --checkpoint "${ckpt}" -n 4 -c "${cond_steps}" --horizon "${horizon}" --prefix "det_"
  echo "[eval:${name}] videos sample"
  "${PYTHON}" generate_lpwm_video_prediction.py -p "${run_dir}" --checkpoint "${ckpt}" -n 4 -c "${cond_steps}" --horizon "${horizon}" --sample --prefix "sample_"

  echo "[eval:${name}] image metrics deterministic"
  "${PYTHON}" eval/eval_gen_metrics.py -p "${run_dir}" --checkpoint "${ckpt}" -b "${batch}" -c "${cond_steps}" --horizon "${horizon}"
  copy_if_exists "${eval_dir}/last_val_image_metrics.json" "${eval_dir}/metrics_test_det.json"

  echo "[eval:${name}] image metrics sample"
  "${PYTHON}" eval/eval_gen_metrics.py -p "${run_dir}" --checkpoint "${ckpt}" -b "${batch}" -c "${cond_steps}" --horizon "${horizon}" --sample
  copy_if_exists "${eval_dir}/last_val_image_metrics.json" "${eval_dir}/metrics_test_sample.json"

  if [[ "${run_ctx_eval}" == "1" ]]; then
    echo "[eval:${name}] image metrics sample+ctx"
    "${PYTHON}" eval/eval_gen_metrics.py -p "${run_dir}" --checkpoint "${ckpt}" -b "${batch}" -c "${cond_steps}" --horizon "${horizon}" --sample --ctx
    copy_if_exists "${eval_dir}/last_val_image_metrics.json" "${eval_dir}/metrics_test_sample_ctx.json"
  fi

  if [[ -f "${FVD_WEIGHTS}" || "${ALLOW_FVD_DOWNLOAD}" == "1" ]]; then
    echo "[eval:${name}] fvd sample"
    "${PYTHON}" eval/eval_fvd.py -p "${run_dir}" --checkpoint "${ckpt}" -b 4 -c "${cond_steps}" --horizon "${horizon}" --sample --n_videos_per_clip 1
    copy_if_exists "${eval_dir}/last_test_fvd.json" "${eval_dir}/fvd_test_sample.json"
  else
    echo "[eval:${name}] skip fvd: missing ${FVD_WEIGHTS}"
  fi
}

collect_summary() {
  echo "[summary] writing ${SUMMARY_CSV} and ${SUMMARY_MD}"
  "${PYTHON}" ./tools/collect_flow_results.py \
    --output-csv "${SUMMARY_CSV}" \
    --output-md "${SUMMARY_MD}" \
    --run "Sketchy-U=${SKETCHY_U_DIR}" \
    --run "Sketchy-A=${SKETCHY_A_DIR}" \
    --run "BAIR-U=${BAIR_U_DIR}" \
    --run "Mario-U=${MARIO_U_DIR}" \
    --run "Bridge-L=${BRIDGE_L_DIR}" \
    --run "PandaPush=${PANDAPUSH_DIR}" \
    --run "OGBench-Scene=${OGBENCH_SCENE_DIR}"

  echo "[summary] writing ${BASELINE_TEMPLATE_CSV} and ${BASELINE_TEMPLATE_MD}"
  "${PYTHON}" ./tools/make_baseline_comparison_template.py \
    --flow-summary-csv "${SUMMARY_CSV}" \
    --output-csv "${BASELINE_TEMPLATE_CSV}" \
    --output-md "${BASELINE_TEMPLATE_MD}"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  run_one_experiment "Sketchy-U" "${CFG_SKETCHY_U}"
  run_one_experiment "Sketchy-A" "${CFG_SKETCHY_A}"
  run_one_experiment "BAIR-U" "${CFG_BAIR_U}"
  run_one_experiment "Mario-U" "${CFG_MARIO_U}"
  run_one_experiment "Bridge-L" "${CFG_BRIDGE_L}"
  run_one_experiment "PandaPush" "${CFG_PANDAPUSH}"
  run_one_experiment "OGBench-Scene" "${CFG_OGBENCH_SCENE}"
fi

if [[ "${RUN_EVAL_AND_VIS}" == "1" || "${RUN_SUMMARY}" == "1" ]]; then
  SKETCHY_U_DIR="$(require_dir "$(latest_run_dir 'sketchy_flowlpwm_flow_sketchy_u_exp')")"
  SKETCHY_A_DIR="$(require_dir "$(latest_run_dir 'sketchy_flowlpwm_flow_sketchy_a_exp')")"
  BAIR_U_DIR="$(require_dir "$(latest_run_dir 'bair_flowlpwm_flow_bair_u_exp')")"
  MARIO_U_DIR="$(require_dir "$(latest_run_dir 'mario_flowlpwm_flow_mario_u_exp')")"
  BRIDGE_L_DIR="$(require_dir "$(latest_run_dir 'bridge_flowlpwm_flow_bridge_l_exp')")"
  PANDAPUSH_DIR="$(require_dir "$(latest_run_dir 'panda_flowlpwm_flow_pandapush_exp')")"
  OGBENCH_SCENE_DIR="$(require_dir "$(latest_run_dir 'ogbench_flowlpwm_flow_ogbench_scene_exp')")"
fi

if [[ "${RUN_EVAL_AND_VIS}" == "1" ]]; then
  SKETCHY_U_CKPT="$(require_file "$(latest_checkpoint "${SKETCHY_U_DIR}")")"
  SKETCHY_A_CKPT="$(require_file "$(latest_checkpoint "${SKETCHY_A_DIR}")")"
  BAIR_U_CKPT="$(require_file "$(latest_checkpoint "${BAIR_U_DIR}")")"
  MARIO_U_CKPT="$(require_file "$(latest_checkpoint "${MARIO_U_DIR}")")"
  BRIDGE_L_CKPT="$(require_file "$(latest_checkpoint "${BRIDGE_L_DIR}")")"
  PANDAPUSH_CKPT="$(require_file "$(latest_checkpoint "${PANDAPUSH_DIR}")")"
  OGBENCH_SCENE_CKPT="$(require_file "$(latest_checkpoint "${OGBENCH_SCENE_DIR}")")"

  run_video_suite "Sketchy-U" "${SKETCHY_U_DIR}" "${SKETCHY_U_CKPT}" 10 60 8 1
  run_video_suite "Sketchy-A" "${SKETCHY_A_DIR}" "${SKETCHY_A_CKPT}" 10 60 8 1
  run_video_suite "BAIR-U" "${BAIR_U_DIR}" "${BAIR_U_CKPT}" 1 30 8 1
  run_video_suite "Mario-U" "${MARIO_U_DIR}" "${MARIO_U_CKPT}" 6 60 4 1
  run_video_suite "Bridge-L" "${BRIDGE_L_DIR}" "${BRIDGE_L_CKPT}" 2 45 6 1
  run_video_suite "PandaPush" "${PANDAPUSH_DIR}" "${PANDAPUSH_CKPT}" 2 60 4 1
  run_video_suite "OGBench-Scene" "${OGBENCH_SCENE_DIR}" "${OGBENCH_SCENE_CKPT}" 2 60 8 1
  RUN_SUMMARY=1
fi

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  collect_summary
fi

echo "exp_todo.sh finished."


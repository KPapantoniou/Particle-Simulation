#!/usr/bin/env bash
set -euo pipefail

DEVICE=${DEVICE:-auto}
MODE=${MODE:-both}
BATCH_SIZE=${BATCH_SIZE:-32}
BATCH_SEED=${BATCH_SEED:-123}
DT=${DT:-1e-3}
T_MAX=${T_MAX:-20.0}
HISTORY_DEVICE=${HISTORY_DEVICE:-cpu}
HISTORY_STRIDE=${HISTORY_STRIDE:-10}
POTENTIAL_STRIDE=${POTENTIAL_STRIDE:-10}
SAVE_POTENTIAL=${SAVE_POTENTIAL:-0}
K=${K:-1.75}
GAMMA=${GAMMA:-1.0}
CURRENT_LIMIT=${CURRENT_LIMIT:-2.0}
K_GAIN_JITTER=${K_GAIN_JITTER:-0.05}
DAMPING_JITTER=${DAMPING_JITTER:-0.05}
START_MARGIN=${START_MARGIN:-0.9}
TARGET_MARGIN=${TARGET_MARGIN:-0.9}
STOP_TOLERANCE=${STOP_TOLERANCE:-1e-6}
OUTPUT_DIR=${OUTPUT_DIR:-results}

/home/kpapantoniou/MyProjects/Particle-Sim/particle_sim/bin/python run.py \
  --device "${DEVICE}" \
  --mode "${MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --batch-seed "${BATCH_SEED}" \
  --dt "${DT}" \
  --t-max "${T_MAX}" \
  --history-device "${HISTORY_DEVICE}" \
  --history-stride "${HISTORY_STRIDE}" \
  --potential-stride "${POTENTIAL_STRIDE}" \
  --k "${K}" \
  --gamma "${GAMMA}" \
  --current-limit "${CURRENT_LIMIT}" \
  --k-gain-jitter "${K_GAIN_JITTER}" \
  --damping-jitter "${DAMPING_JITTER}" \
  --start-margin "${START_MARGIN}" \
  --target-margin "${TARGET_MARGIN}" \
  --stop-tolerance "${STOP_TOLERANCE}" \
  --output-dir "${OUTPUT_DIR}" \
  $( [ "${SAVE_POTENTIAL}" = "1" ] && echo "--save-potential" ) \
  "$@"

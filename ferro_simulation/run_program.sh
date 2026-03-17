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
SAVE_POTENTIAL=${SAVE_POTENTIAL:-1}
K=${K:-1.75}
GAMMA=${GAMMA:-1.0}
VOLTAGE_LIMIT=${VOLTAGE_LIMIT:-2.0}
OPEN_LOOP_VOLTAGE=${OPEN_LOOP_VOLTAGE:-}
K_GAIN_JITTER=${K_GAIN_JITTER:-0.05}
DAMPING_JITTER=${DAMPING_JITTER:-0.05}
START_MARGIN=${START_MARGIN:-0.9}
TARGET_MARGIN=${TARGET_MARGIN:-0.9}
STOP_TOLERANCE=${STOP_TOLERANCE:-1e-6}
OUTPUT_DIR=${OUTPUT_DIR:-results}
DT_SWEEP=${DT_SWEEP:-}
DT_SWEEP_LIST=${DT_SWEEP//,/ }
MAKE_FIELD_PLOTS=${MAKE_FIELD_PLOTS:-0}
MAKE_ANIMATION=${MAKE_ANIMATION:-0}
FIELD_BATCH=${FIELD_BATCH:-0}
FIELD_FRAME=${FIELD_FRAME:--1}
FIELD_SAVE_DIR=${FIELD_SAVE_DIR:-${OUTPUT_DIR}/field_plots}
ANIMATION_FPS=${ANIMATION_FPS:-20}

EXTRA_RUN_ARGS=()
while (($# > 0)); do
  case "$1" in
    --make-field-plots)
      MAKE_FIELD_PLOTS=1
      shift
      ;;
    --make-animation)
      MAKE_ANIMATION=1
      shift
      ;;
    --field-batch)
      FIELD_BATCH="$2"
      shift 2
      ;;
    --field-frame)
      FIELD_FRAME="$2"
      shift 2
      ;;
    --field-save-dir)
      FIELD_SAVE_DIR="$2"
      shift 2
      ;;
    --animation-fps)
      ANIMATION_FPS="$2"
      shift 2
      ;;
    --)
      shift
      while (($# > 0)); do
        EXTRA_RUN_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_RUN_ARGS+=("$1")
      shift
      ;;
  esac
done

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
  --voltage-limit "${VOLTAGE_LIMIT}" \
  --k-gain-jitter "${K_GAIN_JITTER}" \
  --damping-jitter "${DAMPING_JITTER}" \
  --start-margin "${START_MARGIN}" \
  --target-margin "${TARGET_MARGIN}" \
  --stop-tolerance "${STOP_TOLERANCE}" \
  $( [ -n "${DT_SWEEP_LIST}" ] && printf -- "--dt-sweep %s" "${DT_SWEEP_LIST}" ) \
  --output-dir "${OUTPUT_DIR}" \
  $( [ -n "${OPEN_LOOP_VOLTAGE}" ] && echo "--open-loop-voltage ${OPEN_LOOP_VOLTAGE}" ) \
  $( [ "${SAVE_POTENTIAL}" = "1" ] && echo "--save-potential" ) \
  "${EXTRA_RUN_ARGS[@]}"

if [ "${MAKE_FIELD_PLOTS}" = "1" ] || [ "${MAKE_ANIMATION}" = "1" ]; then
  mkdir -p "${FIELD_SAVE_DIR}"
fi

if [ "${MAKE_FIELD_PLOTS}" = "1" ]; then
  for pt in "${OUTPUT_DIR}"/trajectories_*.pt; do
    if [ -f "${pt}" ]; then
      if ! /home/kpapantoniou/MyProjects/Particle-Sim/particle_sim/bin/python view_field_pt.py \
        "${pt}" \
        --batch "${FIELD_BATCH}" \
        --frame "${FIELD_FRAME}" \
        --save-dir "${FIELD_SAVE_DIR}"; then
        echo "Warning: field plotting failed for ${pt}" >&2
      fi
    fi
  done
fi

if [ "${MAKE_ANIMATION}" = "1" ]; then
  for pt in "${OUTPUT_DIR}"/trajectories_*.pt; do
    if [ -f "${pt}" ]; then
      stem="$(basename "${pt}" .pt)"
      if ! /home/kpapantoniou/MyProjects/Particle-Sim/particle_sim/bin/python animate_field_pt.py \
        "${pt}" \
        --batch "${FIELD_BATCH}" \
        --fps "${ANIMATION_FPS}" \
        --out "${FIELD_SAVE_DIR}/${stem}_potential_animation.mp4"; then
        echo "Warning: animation export failed for ${pt}" >&2
      fi
    fi
  done
fi

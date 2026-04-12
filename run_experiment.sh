#!/usr/bin/env bash
# ============================================================================
#  run_experiment.sh — Ejecuta TODO el experimento ITS Pipeline Comparison
#
#  Flujos:
#    A (baseline)  — sin preprocesado semantico
#    B (RF)        — Random Forest sobre 27 features geometricas
#    C (PointNet++) — PointNet++ MSG
#
#  Uso:
#    conda activate forest_its
#    bash run_experiment.sh          # todo el experimento
#    bash run_experiment.sh --from 5 # retomar desde paso 5
#
#  Los resultados quedan en:
#    output/results/comparison_table_test.csv   <-- datos para el paper
#    output/results/comparison_table_test.tex   <-- LaTeX listo para copiar
#    output/results/grid_search_best_params.csv
#    output/models/rf_feature_importance.csv
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/experiment_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

run_step() {
    local step_num=$1
    local step_name=$2
    shift 2

    log "=========================================="
    log "PASO $step_num: $step_name"
    log "=========================================="
    log "Comando: $*"

    local t_start=$SECONDS
    if "$@" 2>&1 | tee -a "$MASTER_LOG"; then
        local elapsed=$(( SECONDS - t_start ))
        log "PASO $step_num completado en ${elapsed}s"
    else
        log "ERROR: PASO $step_num fallo"
        log "Para retomar: bash run_experiment.sh --from $step_num"
        exit 1
    fi
    echo "" | tee -a "$MASTER_LOG"
}

# --- Parsear --from ---
START_FROM=1
if [[ "${1:-}" == "--from" ]]; then
    START_FROM="${2:?'--from requiere un numero de paso (1-12)'}"
fi

log "============================================================"
log "EXPERIMENTO ITS PIPELINE COMPARISON — inicio $(date)"
if (( START_FROM > 1 )); then
    log "Retomando desde paso $START_FROM"
fi
log "============================================================"

# === PASO 1: Entrenar RF ===
if (( START_FROM <= 1 )); then
    run_step 1 "Entrenar Random Forest" \
        python -m forest_its.methods.rf.train_rf
fi

# === PASO 2: Entrenar PointNet++ ===
if (( START_FROM <= 2 )); then
    run_step 2 "Entrenar PointNet++ MSG (100 epochs)" \
        python -m forest_its.methods.pointnet2.train_pointnet2
fi

# === PASO 3: Semantico RF — val ===
if (( START_FROM <= 3 )); then
    run_step 3 "RF semantico — val" \
        python -m forest_its.methods.rf.run_rf_pipeline --stage semantic --split val
fi

# === PASO 4: Semantico PointNet++ — val ===
if (( START_FROM <= 4 )); then
    run_step 4 "PointNet++ semantico — val" \
        python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage semantic --split val
fi

# === PASO 5: Grid search Watershed 3D (3 flujos, 36 combos c/u) ===
if (( START_FROM <= 5 )); then
    run_step 5 "Grid search Watershed 3D (baseline + RF + PointNet++)" \
        python -m forest_its.evaluation.grid_search --methods baseline rf pointnet2
fi

# === PASO 6: Instancias val — 3 flujos ===
if (( START_FROM <= 6 )); then
    run_step 6 "Baseline instancias — val" \
        python -m forest_its.methods.baseline.run_baseline --split val
fi

if (( START_FROM <= 7 )); then
    run_step 7 "RF instancias — val" \
        python -m forest_its.methods.rf.run_rf_pipeline --stage instance --split val
fi

if (( START_FROM <= 8 )); then
    run_step 8 "PointNet++ instancias — val" \
        python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage instance --split val
fi

# === PASO 9: Semantico RF + PN++ — test ===
if (( START_FROM <= 9 )); then
    run_step 9 "RF semantico — test" \
        python -m forest_its.methods.rf.run_rf_pipeline --stage semantic --split test
fi

if (( START_FROM <= 10 )); then
    run_step 10 "PointNet++ semantico — test" \
        python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage semantic --split test
fi

# === PASO 11: Instancias test — 3 flujos ===
if (( START_FROM <= 11 )); then
    run_step 11 "Baseline + RF + PointNet++ instancias — test" \
        bash -c '
            python -m forest_its.methods.baseline.run_baseline --split test && \
            python -m forest_its.methods.rf.run_rf_pipeline --stage instance --split test && \
            python -m forest_its.methods.pointnet2.run_pointnet2_pipeline --stage instance --split test
        '
fi

# === PASO 12: Tablas comparativas finales (baseline + rf + pointnet2) ===
if (( START_FROM <= 12 )); then
    run_step 12 "Tablas comparativas val + test" \
        bash -c '
            python -m forest_its.evaluation.run_evaluation --methods baseline rf pointnet2 --split val && \
            python -m forest_its.evaluation.run_evaluation --methods baseline rf pointnet2 --split test
        '
fi

# === PASO 13: Grid search Watershed Density (rf_density + pointnet2_density) ===
if (( START_FROM <= 13 )); then
    run_step 13 "Grid search Watershed Density (rf_density + pointnet2_density)" \
        python -m forest_its.evaluation.grid_search --methods rf_density pointnet2_density
fi

# === PASO 14: RF Density instancias — val + test ===
if (( START_FROM <= 14 )); then
    run_step 14 "RF Density instancias — val + test" \
        bash -c '
            python -m forest_its.methods.rf_density.run_rf_density_pipeline --stage instance --split val && \
            python -m forest_its.methods.rf_density.run_rf_density_pipeline --stage instance --split test
        '
fi

# === PASO 15: PointNet++ Density instancias — val + test ===
if (( START_FROM <= 15 )); then
    run_step 15 "PointNet++ Density instancias — val + test" \
        bash -c '
            python -m forest_its.methods.pointnet2_density.run_pointnet2_density_pipeline --stage instance --split val && \
            python -m forest_its.methods.pointnet2_density.run_pointnet2_density_pipeline --stage instance --split test
        '
fi

# === PASO 16: Tablas comparativas finales (todos los métodos) ===
if (( START_FROM <= 16 )); then
    run_step 16 "Tablas comparativas todos los métodos (val + test)" \
        bash -c '
            python -m forest_its.evaluation.run_evaluation --methods baseline rf pointnet2 rf_density pointnet2_density --split val && \
            python -m forest_its.evaluation.run_evaluation --methods baseline rf pointnet2 rf_density pointnet2_density --split test
        '
fi

log ""
log "============================================================"
log "EXPERIMENTO COMPLETADO — $(date)"
log "============================================================"
log ""
log "Resultados para el paper:"
log "  output/results/comparison_table_test.csv"
log "  output/results/comparison_table_test.tex"
log "  output/results/grid_search_best_params.csv"
log "  output/models/rf_feature_importance.csv"
log ""
log "Log: $MASTER_LOG"

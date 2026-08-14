#!/usr/bin/env bash

set -o pipefail
mkdir -p results/test_stage_a

TEST="/workspace/REACT_TTS_rev/data/meld_full_v2/final/manifests/test.jsonl"

run_eval () {
    GPU=$1
    COND=$2
    SEED=$3
    CFG=$4
    CKPT=$5

    OUT="results/test_stage_a/${COND}_seed${SEED}"

    if [ -f "${OUT}.json" ] && [ -f "${OUT}.npz" ]; then
        echo "[SKIP] ${COND} seed${SEED} already exists"
        return 0
    fi

    echo "===== START ${COND} seed${SEED} GPU${GPU} ====="

    CUDA_VISIBLE_DEVICES=$GPU python3 -m scripts.eval_stage_a_test \
        --config "$CFG" \
        --ckpt "$CKPT" \
        --manifest_test "$TEST" \
        --condition "$COND" \
        --batch_size 32 \
        --device cuda \
        --out_json "${OUT}.json" \
        --out_npz "${OUT}.npz" \
        > "${OUT}.log" 2>&1

    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "!!!!! FAILED ${COND} seed${SEED} !!!!!"
        tail -n 30 "${OUT}.log"
        return $STATUS
    fi

    echo "===== DONE ${COND} seed${SEED} ====="
}


# ============================================================
# GPU 0
# Temporal No-Delta: seeds 42-51
# seed42 will automatically SKIP because it is already done.
# ============================================================
(
    for s in $(seq 42 51); do

        if [ "$s" = "42" ]; then
            CFG="configs/stage_a_no_delta_final.yaml"
            CKPT="checkpoints/stage_a_no_delta_final/best.pt"
        else
            CFG="configs/stage_a_no_delta_seed${s}.yaml"
            CKPT="checkpoints/stage_a_no_delta_seed${s}/best.pt"
        fi

        run_eval 0 temporal "$s" "$CFG" "$CKPT" || exit 1
    done
) &

PID0=$!


# ============================================================
# GPU 1
# Text-only: seeds 42-51
# ============================================================
(
    for s in $(seq 42 51); do

        if [ "$s" = "42" ]; then
            CFG="configs/stage_a_text_only_final.yaml"
            CKPT="checkpoints/stage_a_text_only_final/best.pt"
        else
            CFG="configs/stage_a_text_only_seed${s}.yaml"
            CKPT="checkpoints/stage_a_text_only_seed${s}/best.pt"
        fi

        run_eval 1 text_only "$s" "$CFG" "$CKPT" || exit 1
    done
) &

PID1=$!


# ============================================================
# GPU 2
# Static 42-46, then Full+Delta 42-46
# ============================================================
(
    for s in $(seq 42 46); do

        if [ "$s" = "42" ]; then
            CFG="configs/stage_a_static_final.yaml"
            CKPT="checkpoints/stage_a_static_final/best.pt"
        else
            CFG="configs/stage_a_static_seed${s}.yaml"
            CKPT="checkpoints/stage_a_static_seed${s}/best.pt"
        fi

        run_eval 2 static "$s" "$CFG" "$CKPT" || exit 1
    done

    for s in $(seq 42 46); do

        if [ "$s" = "42" ]; then
            CFG="configs/stage_a_full_final.yaml"
            CKPT="checkpoints/stage_a_full_final/best.pt"
        else
            CFG="configs/stage_a_full_seed${s}.yaml"
            CKPT="checkpoints/stage_a_full_seed${s}/best.pt"
        fi

        run_eval 2 full "$s" "$CFG" "$CKPT" || exit 1
    done
) &

PID2=$!


echo "GPU0 PID=$PID0"
echo "GPU1 PID=$PID1"
echo "GPU2 PID=$PID2"
echo "All Test queues launched."

wait $PID0
S0=$?

wait $PID1
S1=$?

wait $PID2
S2=$?

echo
echo "========== ALL QUEUES FINISHED =========="
echo "GPU0 status=$S0"
echo "GPU1 status=$S1"
echo "GPU2 status=$S2"

if [ $S0 -eq 0 ] && [ $S1 -eq 0 ] && [ $S2 -eq 0 ]; then
    echo "ALL TEST EVALUATIONS COMPLETED SUCCESSFULLY"
else
    echo "ONE OR MORE QUEUES FAILED — CHECK LOGS"
    exit 1
fi

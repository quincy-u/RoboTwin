GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
mkdir -p eval_logs
export SIMPLE_GRASP_ROOT="$HOME/projects/simple-grasp"

# echo "start easy SR eval"

# xargs -r -n 1 \
#   -P "$GPU_COUNT" \
#   --process-slot-var=GPU_ID \
#   bash -c '
#     task=$1
#     log="eval_logs/${task}.log"

#     echo "[GPU ${GPU_ID}] starting ${task}"

#     if bash policy/heuristic_baseline/eval.sh \
#         "$task" demo_clean 0 "$GPU_ID" auto 100 \
#         >"$log" 2>&1
#     then
#         echo "[GPU ${GPU_ID}] finished ${task}"
#     else
#         echo "[GPU ${GPU_ID}] crashed ${task}; see ${log}"
#     fi
#   ' _ < heuristic_tasks.txt

echo "start hard SR eval"

xargs -r -n 1 \
  -P "$GPU_COUNT" \
  --process-slot-var=GPU_ID \
  bash -c '
    task=$1
    log="eval_logs/${task}.log"

    echo "[GPU ${GPU_ID}] starting ${task}"

    if bash policy/heuristic_baseline/eval.sh \
        "$task" demo_randomized 0 "$GPU_ID" auto 100 \
        >"$log" 2>&1
    then
        echo "[GPU ${GPU_ID}] finished ${task}"
    else
        echo "[GPU ${GPU_ID}] crashed ${task}; see ${log}"
    fi
  ' _ < heuristic_tasks.txt
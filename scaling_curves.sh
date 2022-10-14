#!/bin/bash
set -e

# Constants
DIR="scaling"
MIN_NUM_RECORDS="10000"
GEN_TEST_INTERV="10"
N_EDITS="1,56,100,316,562,1000,1778,3162,5623,10000"

# Run configurations
MODEL_NAME="EleutherAI/gpt-j-6B"
ALG_NAMES=("FT" "MEND" "ROME" "MEMIT")
HPARAMS_FNAMES=("EleutherAI_gpt-j-6B_wd.json" "EleutherAI_gpt-j-6B.json" "EleutherAI_gpt-j-6B.json" "EleutherAI_gpt-j-6B.json")

# Execute
for i in ${!ALG_NAMES[@]}
do
    alg_name=${ALG_NAMES[$i]}
    hparams_fname=${HPARAMS_FNAMES[$i]}

    echo "Running evals for $alg_name..."
    sweep_dir="$DIR/$alg_name"
    
    if [ -d "results/$sweep_dir" ]; then
        echo "Note: results/$sweep_dir already exists! Continuing from previous run..."
    fi

    echo "Dumping results at results/$sweep_dir"
    mkdir -p results/$sweep_dir
    echo "{}" > results/$sweep_dir/config.json

    python3 -m experiments.sweep --alg_name=$alg_name --model_name=$MODEL_NAME --hparams_fname=$hparams_fname --sweep_dir=$sweep_dir --min_num_records=$MIN_NUM_RECORDS --num_edits=$N_EDITS --generation_test_interval=$GEN_TEST_INTERV --use_cache
done

exit 0

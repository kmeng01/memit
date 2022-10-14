#!/bin/bash
set -e

# Constants
N_EDITS="10000"

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

    python3 -m experiments.evaluate --alg_name=$alg_name --model_name=$MODEL_NAME --hparams_fname=$hparams_fname --num_edits=$N_EDITS --use_cache --dataset_size_limit=$N_EDITS --ds_name=zsre
done

exit 0

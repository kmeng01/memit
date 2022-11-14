import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.evaluate import HPARAMS_DIR
from experiments.evaluate import main as eval_main

TMP_PARAMS_TEMPLATE = "sweep_params_tmp_{}_.json"


def exec_sweep(
    alg_name: str,
    model_tok: Tuple[AutoModelForCausalLM, AutoTokenizer],
    hparams_fname: str,
    ds_name: str,
    sweep_dir: Path,
    num_records: int,
    generation_test_interval: bool,
    num_edits: int,
    use_cache: bool,
):
    # Configure hparams
    with open(HPARAMS_DIR / alg_name / hparams_fname, "r") as f:
        hparams_orig = json.load(f)
    with open(Path("results") / sweep_dir / "config.json", "r") as f:
        sweep_config = json.load(f)
        sweep_keys = list(sweep_config.keys())

    # Sweep
    for s_i, state in enumerate(get_states([], sweep_config, sweep_keys)):
        # Set dirs
        tmp_params_name = TMP_PARAMS_TEMPLATE.format(time.time_ns())
        tmp_params_path = HPARAMS_DIR / alg_name / tmp_params_name

        # Set new hparams
        hparams_new = deepcopy(hparams_orig)
        for key_num, state_num in enumerate(state):
            k = sweep_keys[key_num]
            hparams_new[k] = sweep_config[k][state_num]
            print(f"Sweep {s_i}: Setting {k} = {hparams_new[k]}")

        with open(tmp_params_path, "w") as f:
            json.dump(hparams_new, f)

        # Execute
        eval_main(
            alg_name,
            model_name=model_tok,
            hparams_fname=tmp_params_name,
            ds_name=ds_name,
            dataset_size_limit=num_records,
            continue_from_run="run_000",
            skip_generation_tests=(generation_test_interval == -1),
            generation_test_interval=generation_test_interval,
            conserve_memory=False,
            dir_name=sweep_dir / f"{num_edits}_edits_setting_{s_i}",
            num_edits=num_edits,
            use_cache=use_cache,
        )

        # Clean up
        os.remove(tmp_params_path)


def get_states(
    state: List,
    sweep_config: Dict,
    sweep_keys: List,
):
    """
    Standard recursive procedure for generating all possible configurations.
    """

    ans = []
    if len(state) < len(sweep_config):
        for i in range(len(sweep_config[sweep_keys[len(state)]])):
            for s in get_states(state + [i], sweep_config, sweep_keys):
                ans.append(s)
    else:
        ans.append(state)
    return ans


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name", choices=["MEMIT", "FT", "ROME", "MEND"], required=True
    )
    parser.add_argument(
        "--model_name", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"], required=True
    )
    parser.add_argument("--hparams_fname", type=str, required=True)
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument("--min_records", type=int, default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument(
        "--num_edits",
        type=str,
        default="1",
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=-1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument("--sweep_dir", type=str)
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs (MEMIT and ROME only)",
    )

    args = parser.parse_args()
    assert args.sweep_dir is not None, f"Must specify a sweep_dir."

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token

    for cur_num_edits in list(map(int, args.num_edits.split(","))):
        torch.cuda.empty_cache()

        num_records = (
            None if args.max_records is None
            else min(args.max_records, cur_num_edits)
        )
        if args.min_records is not None:
            num_records = max(args.min_records, cur_num_edits)

        exec_sweep(
            args.alg_name,
            (model, tok),
            args.hparams_fname,
            args.ds_name,
            Path(args.sweep_dir),
            num_records,
            args.generation_test_interval,
            cur_num_edits,
            args.use_cache,
        )

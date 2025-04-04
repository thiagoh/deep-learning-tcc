from tabnanny import verbose
import dotenv

dotenv.load_dotenv()

import argparse
from typing import List
from metrics_main import combine_model_results, load_results, results_to_dataframe
from display import export_html
import pandas as pd

from logging import Logger

logger = Logger("compare_models")


def compare_models(
    *,
    results_filename_prefix: str,
    model_results_file_prefix: str,
    model_results_file_suffix: str,
    verbose: bool = False,
):
    results_dfs = []

    model_results_map = {}
    for model_results_entry in load_results(prefix=model_results_file_prefix, suffix=model_results_file_suffix, verbose=verbose):
        model_name = model_results_entry["name"]
        model_results_map[model_name] = model_results_entry["results"]

    grouped_results_map = {}
    for model_name, model_results_entry in model_results_map.items():
        is_rag = model_name.find("-RAG") >= 0
        model_name = model_name.replace("-RAG", "")
        grouped_results_map[model_name] = grouped_results_map[model_name] if model_name in grouped_results_map else {"name": model_name}
        grouped_results_map[model_name][("modelrag" if is_rag else "model")] = model_results_entry
        for entry in model_results_entry:
          entry["IsRag"] = is_rag

    for model_name, grouped_results in grouped_results_map.items():
        if "modelrag" not in grouped_results:
            logger.warning(f"No RAG Model available for {model_name}")
            continue

        grouped_results_map[model_name] = []
        for model_result, modelrag_result in zip(grouped_results["model"], grouped_results["modelrag"]):
            grouped_results_map[model_name].append(model_result)
            grouped_results_map[model_name].append(modelrag_result)

        results_dfs.append((model_name, results_to_dataframe(grouped_results_map[model_name])))
    # print(results_dfs[0][1].columns)
    # exit(0)

    print(f"Comparing models...")
    comparison_df = combine_model_results(model_results_map)

    print(f"Building reports...")
    export_html(
        prefix=results_filename_prefix,
        results_dfs=results_dfs,
        comparison_df=comparison_df,
    )
    print(f"Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_filename_prefix",
        type=str,
        help="Prefix of file on where results will be stored.",
        required=True,
    )
    parser.add_argument(
        "--model_results_file_prefix",
        type=str,
        help="Which models we want to compare.",
        required=True,
    )
    parser.add_argument(
        "--model_results_file_suffix",
        type=str,
        help="Which models we want to compare.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    compare_models(
        results_filename_prefix=args.results_filename_prefix,
        model_results_file_prefix=args.model_results_file_prefix,
        model_results_file_suffix=args.model_results_file_suffix,
        verbose=args.verbose,
    )

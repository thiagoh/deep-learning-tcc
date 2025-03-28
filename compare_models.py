import dotenv

dotenv.load_dotenv()

import argparse
from typing import List
from metrics_main import combine_model_results, load_results, results_to_dataframe
from display import export_html
import pandas as pd


def compare_models(
    *, results_filename_prefix: str, model_results_file_prefix: List[str]
):
    model_results_map = {}
    results_dfs = []
    for model_file_prefix in model_results_file_prefix:
        model_results_entry = load_results(model_file_prefix)
        model_name = model_results_entry["name"]
        model_results_map[model_name] = model_results_entry["results"]
        results_dfs.append(
            (model_name, results_to_dataframe(model_results_entry["results"]))
        )

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
        nargs="+",
        type=str,
        help="Which models we want to compare.",
        required=True,
    )
    args = parser.parse_args()
    print(type(args.model_results_file_prefix))
    print(args.model_results_file_prefix)
    compare_models(
        results_filename_prefix=args.results_filename_prefix,
        model_results_file_prefix=args.model_results_file_prefix,
    )

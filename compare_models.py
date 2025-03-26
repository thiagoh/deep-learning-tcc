import dotenv

dotenv.load_dotenv()

import argparse
from typing import List
from metrics_main import combine_model_results, load_results, results_to_dataframe
from display import export_html
import pandas as pd


# def process_all_models(*, dataset: str):
#     if dataset not in ["demo", "full"]:
#         raise ValueError("Invalid dataset")

#     data = pd.read_csv(f"./data/{dataset}.csv")
#     questions_with_ground_truth = data[["question", "answer"]].values.tolist()

#     print(f'Processing questions with "LLama2"...')
#     baseline_llama2_results = compute(
#         model_name="LLama2",
#         llm=ChatOllama(model="llama2", temperature=0, max_retries=3),
#         questions_with_ground_truth=questions_with_ground_truth,
#     )

#     print(f'Processing questions with "GPT 4o mini"...')
#     baseline_gpt4o_mini_results = compute(
#         model_name="Chat GPT 4o mini",
#         llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3),
#         questions_with_ground_truth=questions_with_ground_truth,
#     )

#     model_results = {
#         "BL-LLama2": baseline_llama2_results,
#         "BL-GPT-4o-mini": baseline_gpt4o_mini_results,
#     }

#     print(f"Comparing models...")
#     comparison_df = compare_models(model_results)

#     print(f"Building reports...")
#     export_html(
#         name="test",
#         results_dfs=[
#             ("BL-LLama2", results_to_dataframe(baseline_llama2_results)),
#             ("BL-GPT-4o-mini", results_to_dataframe(baseline_gpt4o_mini_results)),
#         ],
#         comparison_df=comparison_df,
#     )
#     print(f"Done.")


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

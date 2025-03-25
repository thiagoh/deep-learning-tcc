import dotenv

dotenv.load_dotenv()

import argparse
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from metrics_main import compare_models, evaluate, results_to_dataframe
from llm import query_llm
from display import export_html
import pandas as pd

MODELS = {
    "llama2": ("llama2", "Llama2"),
    "llama3.2-3b": ("llama3.2", "Llama3.2-3b"),
    "gpt-4o-mini": ("gpt-4o-mini", "GPT-4o-mini"),
}


def compute(
    *,
    model_name: str,
    llm: BaseChatModel,
    questions_with_ground_truth: List[Tuple[str, str]],
    verbose=False,
    save_data=True,
    data_filename_prefix: str = "",
):
    questions, ground_truths, predictions = query_llm(
        llm=llm,
        questions_with_ground_truth=questions_with_ground_truth,
        verbose=verbose,
    )

    results = evaluate(
        model_name=model_name,
        ground_truths=ground_truths,
        predictions=predictions,
        questions=questions,
        save_data=save_data,
        data_filename_prefix=data_filename_prefix,
    )

    return results


def process_all_models(*, dataset: str):
    if dataset not in ["demo", "full"]:
        raise ValueError("Invalid dataset")

    data = pd.read_csv(f"./data/{dataset}.csv")
    questions_with_ground_truth = data[["question", "answer"]].values.tolist()

    print(f'Processing questions with "LLama2"...')
    baseline_llama2_results = compute(
        model_name="LLama2",
        llm=ChatOllama(model="llama2", temperature=0, max_retries=3),
        questions_with_ground_truth=questions_with_ground_truth,
    )

    print(f'Processing questions with "GPT 4o mini"...')
    baseline_gpt4o_mini_results = compute(
        model_name="Chat GPT 4o mini",
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3),
        questions_with_ground_truth=questions_with_ground_truth,
    )

    model_results = {
        "BL-LLama2": baseline_llama2_results,
        "BL-GPT-4o-mini": baseline_gpt4o_mini_results,
    }

    print(f"Comparing models...")
    comparison_df = compare_models(model_results)

    print(f"Building reports...")
    export_html(
        name="test",
        results_dfs=[
            ("BL-LLama2", results_to_dataframe(baseline_llama2_results)),
            ("BL-GPT-4o-mini", results_to_dataframe(baseline_gpt4o_mini_results)),
        ],
        comparison_df=comparison_df,
    )
    print(f"Done.")


def process_model(*, dataset: str, model_id: str, data_filename_prefix: str = ""):
    if dataset not in ["demo", "full"]:
        raise ValueError("Invalid dataset")

    if model_id not in MODELS.keys():
        raise ValueError("Invalid model")

    (model_id, model_name) = MODELS[args.model]

    data = pd.read_csv(f"./data/{dataset}.csv")
    questions_with_ground_truth = data[["question", "answer"]].values.tolist()

    print(f'Processing questions with "{model_name}"...')
    baseline_llama2_results = compute(
        model_name=model_name,
        llm=ChatOllama(model=model_id, temperature=0, max_retries=3),
        questions_with_ground_truth=questions_with_ground_truth,
        save_data=True,
        data_filename_prefix=data_filename_prefix,
    )

    print(f"Done.")
    return baseline_llama2_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=["llama2", "llama3.2-3b", "gpt-4o-mini"],
        help="Which models we want to run.",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--data_filename_prefix",
        help="Which prefix to use in files when saving results from processing the model.",
        type=str,
        default="",
    )
    args = parser.parse_args()
    print(args)
    process_model(
        dataset="full",
        model_id=args.model,
        data_filename_prefix=args.data_filename_prefix,
    )

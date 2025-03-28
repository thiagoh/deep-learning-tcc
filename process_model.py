import dotenv
from openai import vector_stores

from vectordb import get_store

dotenv.load_dotenv()

import argparse
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from metrics_main import evaluate
from llm import query_llm
from display import export_html
import pandas as pd

# fmt: off
MODELS = {
    "llama2":       ("llama2",      "Llama2",       ChatOllama,),
    "llama3.2-3b":  ("llama3.2",    "Llama3.2-3b",  ChatOllama,),
    "gpt-4o-mini":  ("gpt-4o-mini", "GPT-4o-mini",  ChatOpenAI,),
}
# fmt: on


def compute(
    *,
    model_name: str,
    llm: BaseChatModel,
    questions_with_ground_truth: List[Tuple[str, str]],
    vector_store_config_name=None,
    verbose=False,
    save_data=True,
    data_filename_prefix: str = "",
):
    questions, ground_truths, predictions = query_llm(
        llm=llm,
        questions_with_ground_truth=questions_with_ground_truth,
        verbose=verbose,
    )

    if vector_store_config_name:
        vector_store = get_store(config_name="")

    results = evaluate(
        model_name=model_name,
        ground_truths=ground_truths,
        predictions=predictions,
        questions=questions,
        save_data=save_data,
        data_filename_prefix=data_filename_prefix,
    )

    return results


def process_model(*, dataset: str, model_id: str, data_filename_prefix: str = ""):
    if dataset not in ["demo", "full"]:
        raise ValueError("Invalid dataset")

    if model_id not in MODELS.keys():
        raise ValueError("Invalid model")

    (model_id, model_name, ModelClass) = MODELS[args.model]

    data = pd.read_csv(f"./data/{dataset}.csv")
    questions_with_ground_truth = data[["question", "answer"]].values.tolist()

    print(f'Processing questions with "{model_name}"...')
    model_results = compute(
        model_name=model_name,
        llm=ModelClass(model=model_id, temperature=0, max_retries=3),
        questions_with_ground_truth=questions_with_ground_truth,
        save_data=True,
        data_filename_prefix=data_filename_prefix,
    )

    print(f"Done.")
    return model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["demo", "full"],
        help="Which dataset to use.",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["llama2", "llama3.2-3b", "gpt-4o-mini"],
        help="Which models to run.",
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
        dataset=args.dataset,
        model_id=args.model,
        data_filename_prefix=args.data_filename_prefix,
    )

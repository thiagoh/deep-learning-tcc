import dotenv
from openai import vector_stores

from models import MODELS
from vectordb import get_vector_store

dotenv.load_dotenv()

import argparse
from typing import List, Tuple

from langchain_core.language_models import BaseChatModel
from metrics_main import evaluate
from llm import query_llm
import pandas as pd


def compute(
    *,
    model_name: str,
    llm: BaseChatModel,
    questions: List[str],
    ground_truths: List[str],
    vector_store_config_name=None,
    verbose=False,
    save_data=True,
    data_filename_prefix: str = "",
):
    answers = query_llm(
        llm=llm,
        vector_store_config_name=vector_store_config_name,
        questions=questions,
        verbose=verbose,
    )

    results = evaluate(
        model_name=model_name + ("-RAG" if vector_store_config_name else ""),
        ground_truths=ground_truths,
        predictions=answers,
        questions=questions,
        save_data=save_data,
        data_filename_prefix=data_filename_prefix,
    )

    return results


def process_model(
    *,
    dataset: str,
    model_id: str,
    data_filename_prefix: str = "",
    vector_store_config_name: str = None,
):
    if dataset not in ["demo", "full"]:
        raise ValueError("Invalid dataset")

    if model_id not in MODELS.keys():
        raise ValueError("Invalid model")

    (model_id, model_name, ModelClass) = MODELS[args.model]

    data = pd.read_csv(f"./data/{dataset}.csv")
    questions = data["question"].values.tolist()
    ground_truths = data["answer"].values.tolist()

    print(f'Processing questions with "{model_name}"...')
    model_results = compute(
        model_name=model_name,
        llm=ModelClass(model=model_id, temperature=0, max_retries=3),
        vector_store_config_name=vector_store_config_name,
        questions=questions,
        ground_truths=ground_truths,
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
        choices=MODELS.keys(),
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
    parser.add_argument(
        "--vector_store_config_name",
        help="Which name this configuration has.",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    print(args)
    process_model(
        dataset=args.dataset,
        model_id=args.model,
        data_filename_prefix=args.data_filename_prefix,
        vector_store_config_name=args.vector_store_config_name,
    )

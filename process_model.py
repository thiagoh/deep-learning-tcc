from tabnanny import verbose
import dotenv

dotenv.load_dotenv()

from utils import setup_log
import argparse
from models import MODELS
from typing import List, Literal, Tuple
from langchain_core.language_models import BaseChatModel
from evaluation import evaluate
from llm import query_llm
import pandas as pd

setup_log()


def compute(
    *,
    model_name: str,
    llm: BaseChatModel,
    questions: List[str],
    ground_truths: List[str],
    data_filename_suffix: str,
    vector_store_config_name=None,
    k: int = None,
    score_threshold: float = None,
    save_data=True,
    data_filename_prefix: str = "",
    verbose=False,
):
    answers = query_llm(
        llm=llm,
        vector_store_config_name=vector_store_config_name,
        questions=questions,
        k=k,
        score_threshold=score_threshold,
        verbose=verbose,
    )

    results = evaluate(
        model_name=model_name,
        ground_truths=ground_truths,
        predictions=answers,
        questions=questions,
        save_data=save_data,
        data_filename_prefix=data_filename_prefix,
        data_filename_suffix=data_filename_suffix,
        verbose=verbose,
    )

    return results


def process_model(
    *,
    dataset: str,
    model_id: str,
    data_filename_prefix: str = "",
    data_filename_suffix: str = "",
    vector_store_config_name: str = None,
    k: int = None,
    score_threshold: float = None,
    run_type: Literal["model", "model_and_modelrag", "modelrag"],
    verbose=False,
):
    if dataset not in ["demo", "full"]:
        raise ValueError("Invalid dataset")

    if model_id not in MODELS.keys():
        raise ValueError("Invalid model")

    (model_id, model_name, ModelClass) = MODELS[args.model]

    data = pd.read_csv(f"./data/{dataset}.csv")
    questions = data["question"].values.tolist()
    ground_truths = data["answer"].values.tolist()

    if run_type == "model" or run_type == "model_and_modelrag":
        print(f'Processing questions with "{model_name}"...')
        model_results = compute(
            model_name=model_name,
            llm=ModelClass(model=model_id, temperature=0, max_retries=3),
            vector_store_config_name=None,  # Model with no RAG
            questions=questions,
            ground_truths=ground_truths,
            save_data=True,
            data_filename_prefix=data_filename_prefix,
            data_filename_suffix=data_filename_suffix,
            verbose=verbose,
        )

    if run_type == "modelrag" or run_type == "model_and_modelrag":
        print(f'Processing questions with RAG "{model_name}"...')
        model_results = compute(
            model_name=model_name + "-RAG",
            llm=ModelClass(model=model_id, temperature=0, max_retries=3),
            vector_store_config_name=vector_store_config_name,  # Model with RAG
            questions=questions,
            ground_truths=ground_truths,
            k=k,
            score_threshold=score_threshold,
            save_data=True,
            data_filename_prefix=data_filename_prefix,
            data_filename_suffix=data_filename_suffix,
            verbose=verbose,
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
        "--model",
        choices=MODELS.keys(),
        help="Which models to run.",
        type=str,
    )
    parser.add_argument(
        "--data_filename_prefix",
        help="Which prefix to use in files when saving results from processing the model.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--data_filename_suffix",
        help="Which suffix to use in files when saving results from processing the model.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--run_type",
        choices=["model", "model_and_modelrag", "modelrag"],
        help="Which .",
        type=str,
    )
    parser.add_argument(
        "--vector_store_config_name",
        help="Which name this configuration has.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--k",
        help="Max number of documents to add do the context",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--score_threshold",
        help="Score threshold to use when searching for context in vector database",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--verbose",
        help="Verbose",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    process_model(
        dataset=args.dataset,
        model_id=args.model,
        data_filename_prefix=args.data_filename_prefix,
        data_filename_suffix=args.data_filename_suffix,
        vector_store_config_name=args.vector_store_config_name,
        k=args.k,
        score_threshold=args.score_threshold,
        run_type=args.run_type,
        verbose=args.verbose,
    )

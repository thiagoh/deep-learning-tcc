import dotenv

dotenv.load_dotenv()

from typing import Dict, Optional, List, Tuple, Union
from tqdm import tqdm
import json
import glob
import os
import uuid
import warnings
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.evaluation import (
    StringEvaluator,
    EmbeddingDistanceEvalChain,
    StringDistance,
    StringDistanceEvalChain,
    LabeledScoreStringEvalChain,
)


def evaluate(
    *,
    model_name: str,
    ground_truths: List[str],
    predictions: List[str],
    questions: List[str] = None,
    save_data=True,
    data_filename_prefix: str = "",
):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3)

    SYSTEM_MESSAGE = "You are a helpful assistant."
    SCORING_TEMPLATE_WITH_REFERENCE = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            (
                "human",
                "[Instruction]\nPlease act as an impartial judge \
    and evaluate the quality of the response provided by an AI \
    assistant to the user question displayed below in a trivia game. \
    The question is about something in the world the Harry Potter book series. \
    Your evaluation should consider factors such as the helpfulness, relevance, \
    accuracy, depth, creativity, and specially how direct the response was."
                '[Ground truth]\n{reference}\nBegin your evaluation \
    by providing a short explanation. Be as objective as possible. \
    After providing your explanation, you must rate the response on a scale of 1 to 10 \
    by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n\
    [Question]\n{input}\n\n[The Start of Assistant\'s Answer]\n{prediction}\n\
    [The End of Assistant\'s Answer]',
            ),
        ]
    )

    # fmt: off
    METRICS: List[Tuple[str, StringEvaluator]] = [
        ('StringDst(JARO)', StringDistanceEvalChain(distance=StringDistance.JARO),),
        ('StringDst(JARO_WINKLER)', StringDistanceEvalChain(distance=StringDistance.JARO_WINKLER),),
        ('StringDst(INDEL)', StringDistanceEvalChain(distance=StringDistance.INDEL),),
        ('StringDst(LEVENSHTEIN)', StringDistanceEvalChain(distance=StringDistance.LEVENSHTEIN),),
        ('EmbeddingDst(nomic-embed-text)', EmbeddingDistanceEvalChain(embeddings=embeddings),),
        ('LabeledScoreString(GPT4o-mini)', LabeledScoreStringEvalChain(llm=llm, prompt=SCORING_TEMPLATE_WITH_REFERENCE, criterion_name="Customized-criteria"),),
    ]
    # fmt: on

    results = []

    ground_truths_and_predictions = list(zip(ground_truths, predictions))
    progress_ground_truths_and_predictions = tqdm(
        ground_truths_and_predictions,
        total=len(ground_truths_and_predictions),
        desc="Evaluating Responses",
        unit="response",
    )
    try:
        warnings.filterwarnings("ignore", category=UserWarning)
        for i, (ground_truth, answer) in enumerate(
            progress_ground_truths_and_predictions
        ):
            progress_ground_truths_and_predictions.set_description(f"Evaluating...")
            row_data = {
                "Ground_Truth": ground_truth,
                "Answer": answer,
            }
            if questions and i < len(questions):
                row_data["Question"] = questions[i]

            for evaluator_name, evaluator in METRICS:
                try:
                    question = questions[i]
                    comparison_result = evaluator.evaluate_strings(
                        prediction=answer,
                        reference=ground_truth,
                        input=question,
                    )
                    score = comparison_result["score"]
                    row_data[evaluator_name] = score
                except Exception as e:
                    print(f"Error in {evaluator_name}: {e}")

            results.append(row_data)
        progress_ground_truths_and_predictions.set_description_str(
            "Done evaluating responses."
        )
    finally:
        warnings.resetwarnings()

    # Save the data to a JSON file when save_data is True
    if save_data:
        os.makedirs("outputs", exist_ok=True)
        output_file = f"outputs/{data_filename_prefix}{str(uuid.uuid4())}.json"

        # Prepare the data to be saved
        data_to_save = {
            "name": model_name,
            "questions": questions,
            "ground_truths": ground_truths,
            "predictions": predictions,
            "results": results,
        }
        with open(output_file, "w") as f:
            json.dump(data_to_save, f, indent=2)

    return results


def load_results(file_prefix: str) -> Dict:
    directory: str = "outputs"
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None

    matching_files = glob.glob(os.path.join(directory, f"{file_prefix}*.json"))
    if not matching_files:
        print(f"No files found with prefix '{file_prefix}' in {directory}")
        return None

    target_file = sorted(matching_files)[0]
    try:
        with open(target_file, "r") as f:
            results = json.load(f)

        print(f"Loaded results from: {target_file}")
        return results

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading file {target_file}: {e}")
        raise e


def results_to_dataframe(evaluation_results):
    """
    Convert the list of dictionaries from evaluate() into a pandas DataFrame.

    Args:
        evaluation_results (list): List of dictionaries containing evaluation results

    Returns:
        pd.DataFrame: A DataFrame containing all evaluation results
    """

    df = pd.DataFrame(evaluation_results)
    standard_cols = ["Question", "Ground_Truth", "Answer"]
    available_cols = [col for col in standard_cols if col in df.columns]
    metric_cols = [col for col in df.columns if col not in standard_cols]
    return df[available_cols + metric_cols]


def compare_models(model_results_dict):
    """
    Compare evaluation results across multiple models.

    Args:
        model_results_dict (dict): Dictionary mapping model names to their evaluation results
                                   (output from evaluate() function)

    Returns:
        pd.DataFrame: A DataFrame with rows for each model and columns for metric statistics
    """

    # Dictionary to store the aggregated results
    comparison_data = {}

    # Process each model's results
    for model_name, results in model_results_dict.items():
        # Convert list of dicts to DataFrame for easier calculation
        results_df = results_to_dataframe(results)

        # Identify metric columns (skip Question, Ground_Truth, Answer)
        standard_cols = ["Question", "Ground_Truth", "Answer"]
        metric_cols = [col for col in results_df.columns if col not in standard_cols]

        # Calculate statistics for each metric
        model_stats = {}
        for metric in metric_cols:
            metric_values = results_df[metric]

            # Calculate statistics
            model_stats[f"{metric} Mean / StdDev"] = (
                f"{metric_values.mean():.4f} / {metric_values.std():.4f}"
            )
            # model_stats[f"{metric}_mean"] = metric_values.mean()
            # model_stats[f"{metric}_std"] = metric_values.std()
            # model_stats[f"{metric}_min"] = metric_values.min()
            # model_stats[f"{metric}_max"] = metric_values.max()

        # Store this model's stats
        comparison_data[model_name] = model_stats

    # Convert to DataFrame
    comparison_df = pd.DataFrame.from_dict(comparison_data, orient="index")

    return comparison_df

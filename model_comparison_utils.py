from typing import Dict, List, Tuple
import json
import glob
import os
import uuid
import numpy as np
import pandas as pd
from display import export_csv, export_html
import sys
import logging
from utils import is_rag


# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compare_models(
    *,
    results_filename_prefix: str,
    model_results_file_prefix: str,
    model_results_file_suffix: str,
    verbose: bool = False,
) -> None:
    # Get model data
    grouped_results, model_results_map = get_model_data(
        model_results_file_prefix=model_results_file_prefix, model_results_file_suffix=model_results_file_suffix, verbose=verbose
    )

    # Generate results DataFrame
    grouped_results = [(model_name, _results_to_dataframe(grouped_results_entry['combined_results'])) for model_name, grouped_results_entry in grouped_results]

    # Generate comparison DataFrame
    logger.info(f"Comparing models...")
    comparison_df = _combine_model_results(model_results_map)

    # Export to HTML
    logger.info(f"Building reports...")
    export_html(
        prefix=results_filename_prefix,
        results_dfs=grouped_results,
        comparison_df=comparison_df,
    )
    logger.info(f"Done.")


def export_model_data(
    *,
    results_filename_prefix: str,
    model_results_file_prefix: str,
    model_results_file_suffix: str,
    verbose: bool = False,
) -> str:
    # Get model data
    grouped_results, model_results_map = get_model_data(
        model_results_file_prefix=model_results_file_prefix, model_results_file_suffix=model_results_file_suffix, verbose=verbose
    )

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for export
    export_data = {"model_results_map": model_results_map, "grouped_results": grouped_results}

    # Export to JSON
    output_filename = f"{output_dir}/{results_filename_prefix}-full-data-{str(uuid.uuid4())}.json"

    with open(output_filename, "w") as f:
        json.dump(export_data, f, indent=2)
        logger.info(f"Exported full data to: {output_filename}")

    logger.info(f"Done.")

    return output_filename


def export_model_csv_data(
    *,
    results_filename_prefix: str,
    model_results_file_prefix: str,
    model_results_file_suffix: str,
    verbose: bool = False,
) -> str:
    # Get model data
    grouped_results, _ = get_model_data(
        model_results_file_prefix=model_results_file_prefix, model_results_file_suffix=model_results_file_suffix, verbose=verbose
    )

    # Generate comparison DataFrame
    logger.info(f"Comparing models...")
    comparison_df = _combine_model_results_csv(grouped_results)

    # Export to CSV
    logger.info(f"Building reports...")
    export_csv(
        prefix=results_filename_prefix,
        results=comparison_df,
    )
    logger.info(f"Done.")


def get_model_data(
    *,
    model_results_file_prefix: str,
    model_results_file_suffix: str,
    verbose: bool = False,
) -> Tuple[List[Tuple[str, Dict]], Dict]:
    grouped_results = []

    model_results_map = {}
    for model_results_entry in load_results(prefix=model_results_file_prefix, suffix=model_results_file_suffix, verbose=verbose):
        model_name = model_results_entry["name"]
        model_results_map[model_name] = model_results_entry["results"]

    grouped_results_map = {}
    for model_name, model_results in model_results_map.items():
        is_rag_value = is_rag(model_name)
        model_name = model_name.replace("-RAG", "")
        grouped_results_map[model_name] = grouped_results_map[model_name] if model_name in grouped_results_map else {"name": model_name}
        grouped_results_map[model_name][("rag_model_results" if is_rag_value else "model_results")] = model_results
        grouped_results_map[model_name]["is_rag"] = is_rag_value
        for entry in model_results:
            entry["IsRag"] = is_rag_value

    for model_name, grouped_results_entry in grouped_results_map.items():
        if "rag_model_results" not in grouped_results_entry:
            logger.warning(f"No RAG Model available for {model_name}")
            continue

        grouped_results_map[model_name] = {
            "model_results": grouped_results_entry["model_results"],
            "rag_model_results": grouped_results_entry["rag_model_results"],
            "combined_results": [],
        }
        for model_result, rag_model_result in zip(grouped_results_entry["model_results"], grouped_results_entry["rag_model_results"]):
            grouped_results_map[model_name]["combined_results"].append(model_result)
            grouped_results_map[model_name]["combined_results"].append(rag_model_result)

        grouped_results.append((model_name, grouped_results_map[model_name]))

    return grouped_results, model_results_map


def load_results(*, prefix: str, suffix: str, verbose=False) -> List[Dict]:
    directory: str = "outputs"
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None

    matching_files = glob.glob(os.path.join(directory, f"{prefix}*{suffix}.json"))
    if not matching_files:
        print(f"No files found with pattern '{prefix}*{suffix}.json' in {directory}")
        return None

    output = []
    for target_file in sorted(matching_files):
        try:
            with open(target_file, "r") as f:
                output.append(json.load(f))

            if verbose:
                print(f"Loaded results from: {target_file}")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading file {target_file}: {e}")
            raise e

    return output


def _results_to_dataframe(evaluation_results):
    df = pd.DataFrame(evaluation_results)
    standard_cols = ["Question", "Ground_Truth", "Answer", "IsRag"]
    available_cols = [col for col in standard_cols if col in df.columns]
    metric_cols = [col for col in df.columns if col not in standard_cols]
    return df[available_cols + metric_cols]


def _combine_model_results(model_results_dict: dict) -> pd.DataFrame:
    sorted_model_results = dict(sorted(model_results_dict.items(), key=lambda item: item[0].replace("RAG-", "")))

    comparison_data = {}
    for model_name, results in sorted_model_results.items():
        print(f"model_name: {model_name}")
        results_df = _results_to_dataframe(results)

        standard_cols = ["Question", "Ground_Truth", "Answer", "IsRag"]
        metric_cols = [col for col in results_df.columns if col not in standard_cols]

        model_stats = {"is_rag": is_rag(model_name)}
        for metric in metric_cols:
            metric_values = results_df[metric]
            model_stats[metric] = {
                "mean": metric_values.mean(),
                "std": metric_values.std(),
                "median": metric_values.median(),
                "min": metric_values.min(),
                "max": metric_values.max(),
            }

        comparison_data[model_name] = model_stats

    return pd.DataFrame.from_dict(comparison_data, orient="index")


def _combine_model_results_csv(
    model_results_dict: List[Tuple[str, Dict]], input_metric: str = "LabeledScoreString(GPT4o-mini)", output_metric: str = "mean"
) -> pd.DataFrame:
    sorted_model_results = dict(sorted(model_results_dict, key=lambda item: item[0].replace("RAG-", "")))

    comparison_data = {}
    for model_name, entry in sorted_model_results.items():
        print(f"model_name: {model_name}", type(entry["rag_model_results"]))
        comparison_data[model_name] = {
            "Model": model_name,
            "NoRAG": getattr(np, output_metric)(list(map(lambda item: item[input_metric], entry["model_results"]))),
            "RAG": getattr(np, output_metric)(list(map(lambda item: item[input_metric], entry["rag_model_results"]))),
        }

    return pd.DataFrame.from_dict(comparison_data, orient="index")

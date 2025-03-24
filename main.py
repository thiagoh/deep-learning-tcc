from typing import List, Tuple
import dotenv

dotenv.load_dotenv()

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from metrics_main import compare_models, evaluate, results_to_dataframe
from llm import query_llm
from display import export_html
import pandas as pd


def compute(
    *,
    name,
    llm: BaseChatModel,
    questions_with_ground_truth: List[Tuple[str, str]],
    verbose=False,
    save_data=True,
):
    questions, ground_truths, predictions = query_llm(
        llm=llm,
        questions_with_ground_truth=questions_with_ground_truth,
        verbose=verbose,
    )

    results = evaluate(
        ground_truths=ground_truths,
        predictions=predictions,
        questions=questions,
        name=name,
        save_data=save_data,
    )

    return results


def main(dataset: str):
    if dataset not in ["demo", "full"]:
        raise ValueError("Invalid dataset")

    data = pd.read_csv(f"./data/{dataset}.csv")
    questions_with_ground_truth = data[["question", "answer"]].values.tolist()

    print(f'Processing questions with "LLama2"...')
    baseline_llama2_results = compute(
        name="LLama2",
        llm=ChatOllama(model="llama2", temperature=0, max_retries=3),
        questions_with_ground_truth=questions_with_ground_truth,
    )

    print(f'Processing questions with "GPT 4o mini"...')
    baseline_gpt4o_mini_results = compute(
        name="Chat GPT 4o mini",
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


if __name__ == "__main__":
    main('full')

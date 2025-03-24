import dotenv

dotenv.load_dotenv()

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from metrics_main import compare_models, evaluate, results_to_dataframe
from llm import query_llm
from display import export_html
import pandas as pd


def compute(*, name, llm, verbose=False, save_data=True):
    questions, ground_truths, predictions = query_llm(llm=llm, verbose=verbose)

    results = evaluate(
        ground_truths=ground_truths,
        predictions=predictions,
        questions=questions,
        name=name,
        save_data=True,
    )

    return results


def main():
    # # Set maximum columns and rows to display
    # pd.set_option("display.max_columns", None)  # Show all columns
    # pd.set_option("display.max_rows", 20)  # Limit rows shown
    # pd.set_option("display.width", 1000)  # Increase width
    # pd.set_option("display.precision", 2)  # Round decimals
    # pd.set_option("display.colheader_justify", "center")  # Center column headers

    baseline_llama2_results = compute(name="LLama2", llm=ChatOllama(model="llama2"))
    baseline_gpt4o_mini_results = compute(
        name="Chat GPT 4o mini", llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )

    # print("Baseline LLama2 Results:")
    # print(results_to_dataframe(baseline_llama2_results))
    # print("Baseline GPT-4o-mini Results:")
    # print(results_to_dataframe(baseline_gpt4o_mini_results))

    model_results = {
        "BL-LLama2": baseline_llama2_results,
        "BL-GPT-4o-mini": baseline_gpt4o_mini_results,
    }

    comparison_df = compare_models(model_results)

    # print("\n======= MODEL COMPARISON =======")
    # print(comparison_df)

    export_html(
        name="test",
        results_dfs=[
            ("BL-LLama2", results_to_dataframe(baseline_llama2_results)),
            ("BL-GPT-4o-mini", results_to_dataframe(baseline_gpt4o_mini_results)),
        ],
        comparison_df=comparison_df,
    )


if __name__ == "__main__":
    main()

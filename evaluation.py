import dotenv

from utils import setup_log, time_function

dotenv.load_dotenv()

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Ignoring input in .*EvalChain")


import logging
from langchain_ollama import OllamaEmbeddings
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import os
import warnings
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain.evaluation import (
    StringEvaluator,
    EmbeddingDistanceEvalChain,
    StringDistance,
    StringDistanceEvalChain,
    LabeledScoreStringEvalChain,
)

setup_log()

logger = logging.getLogger(__name__)


def get_embeddings() -> Embeddings:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings


def get_llm() -> BaseChatModel:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3)
    # llm = ChatOpenAI(model="o3-mini-2025-01-31", max_retries=3)
    return llm


@time_function()
def evaluate(
    *,
    model_name: str,
    ground_truths: List[str],
    predictions: List[str],
    questions: List[str] = None,
    save_data=True,
    data_filename_prefix: str = "",
    data_filename_suffix: str = "",
    verbose=False,
):
    embeddings = get_embeddings()
    llm = get_llm()

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

    class NormalizedStringEvalChain:

        def __init__(self, instance: StringEvaluator):
            self._instance = instance

        def evaluate_strings(
            self,
            **kwargs,
        ) -> dict:
            result = self._instance.evaluate_strings(**kwargs)
            result["score"] = float(result["score"]) / 10.0
            return result

    labeledScoreStringEvalChain: StringDistanceEvalChain = NormalizedStringEvalChain(
        LabeledScoreStringEvalChain(llm=llm, prompt=SCORING_TEMPLATE_WITH_REFERENCE, criterion_name="Customized-criteria")
    )

    # fmt: off
    EVALUATORS: List[Tuple[str, StringEvaluator]] = [
        ('StringDst(JARO)', StringDistanceEvalChain(distance=StringDistance.JARO),),
        ('StringDst(JARO_WINKLER)', StringDistanceEvalChain(distance=StringDistance.JARO_WINKLER),),
        ('StringDst(INDEL)', StringDistanceEvalChain(distance=StringDistance.INDEL),),
        ('StringDst(LEVENSHTEIN)', StringDistanceEvalChain(distance=StringDistance.LEVENSHTEIN),),
        ('EmbeddingDst(nomic-embed-text)', EmbeddingDistanceEvalChain(embeddings=embeddings),),
        (f'LabeledScoreString({llm.model_name})', labeledScoreStringEvalChain,),
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

    for i, (ground_truth, answer) in enumerate(progress_ground_truths_and_predictions):
        progress_ground_truths_and_predictions.set_description(f"Evaluating...")
        row_data = {
            "Ground_Truth": ground_truth,
            "Answer": answer,
        }
        question = None
        if questions and i < len(questions):
            row_data["Question"] = questions[i]
            question = questions[i]
        _evaluate_answer(
            row_data=row_data,
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            evaluators=EVALUATORS,
        )
        results.append(row_data)
    progress_ground_truths_and_predictions.set_description_str("Done evaluating responses.")

    # Save the data to a JSON file when save_data is True
    if save_data:
        os.makedirs("outputs", exist_ok=True)
        output_file = f"outputs/{data_filename_prefix}-{model_name}-{data_filename_suffix}.json"

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
        if verbose:
            logger.info(f"Saved results into: {output_file}")

    return results


@time_function()
def _evaluate_answer(
    *,
    row_data: dict,
    question: str,
    ground_truth: str,
    answer: str,
    evaluators: List[Tuple[str, StringEvaluator]],
):
    for evaluator_name, evaluator in evaluators:
        try:
            comparison_result = evaluator.evaluate_strings(
                prediction=answer,
                reference=ground_truth,
                input=question,
            )
            score = comparison_result["score"]
            row_data[evaluator_name] = score
        except Exception as e:
            logger.error(f"Error in {evaluator_name}: {e}")

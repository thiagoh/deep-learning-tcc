import dotenv

dotenv.load_dotenv()

from typing import List, Tuple
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel


def query_llm(
    *,
    llm: BaseChatModel,
    questions_with_ground_truth: List[Tuple[str, str]],
    verbose=False,
):
    template = """
    You are a helpful assistant to someone that's answering trivia questions about the Harry Potter books. \
    Answer the question below directly. Do not include any unnecessary words in the answer.
        
    <question>
    {question}
    </question>
    """
    # print("template", ChatPromptTemplate.from_template(template=template))

    # fmt: off
    # questions_with_ground_truth = [
    #   ("What are the three European wizarding schools that participate in the Triwizard Tournament?", "Hogwarts, Beauxbatons, and Durmstrang."),
    #   ("When was Hogwarts founded?", "10th century"),
    #   ("What is a Wronski Feint?", "Pretending to dive for the Snitch"),
    #   ("You wouldn't know anything about this. Name a method to make your broom go faster.", "Using polish"),
    #   ("Harry first took the Knight Bus in The Prisoner of Azkaban. How much does a ticket cost if it includes hot chocolate?", "14 sickles."),
    # ]
    # fmt: on

    chain = ChatPromptTemplate.from_template(template=template) | llm
    ground_truths = []
    predictions = []
    questions = []

    progress_questions_with_ground_truth = tqdm(
        questions_with_ground_truth, desc="Processing Questions", unit="question"
    )

    for question, ground_truth in progress_questions_with_ground_truth:
        progress_questions_with_ground_truth.set_description(f"Processing...")
        result = chain.invoke(input={"question": question})
        answer = result.content
        ground_truths.append(ground_truth)
        questions.append(question)
        predictions.append(answer)

        if verbose:
            print("####################################")
            print(f"# Question: {question}")
            print(f"# Ground Truth:\t{ground_truth}")
            print(f"# Answer:\t\t{answer}")
    progress_questions_with_ground_truth.set_description_str("Done processing questions.")
    return questions, ground_truths, predictions

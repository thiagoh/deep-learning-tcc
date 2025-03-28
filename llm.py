import dotenv

dotenv.load_dotenv()


from typing import List, Tuple
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

DEFAULT_TEMPLATE = """
You are a helpful assistant to someone that's answering trivia questions about the Harry Potter books. \
Answer the question below directly. Do not include any unnecessary words in the answer.
    
<question>
{question}
</question>
"""


def query_llm(
    *,
    llm: BaseChatModel,
    questions_with_ground_truth: List[Tuple[str, str]],
    verbose=False,
):

    chain = ChatPromptTemplate.from_template(template=DEFAULT_TEMPLATE) | llm
    ground_truths = []
    predictions = []
    questions = []

    progress_questions_with_ground_truth = tqdm(
        questions_with_ground_truth, desc="Processing Questions", unit="question"
    )

    for question, ground_truth in progress_questions_with_ground_truth:
        try:
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
        except Exception as e:
            print(f"Error querying question {question}: {e}")

    progress_questions_with_ground_truth.set_description_str(
        "Done processing questions."
    )

    return questions, ground_truths, predictions

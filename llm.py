from tabnanny import verbose
import dotenv


dotenv.load_dotenv()


from models import MODELS
import traceback
from vectordb import get_vector_store
import argparse
from typing import List, Tuple
from tqdm import tqdm
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_core.language_models import BaseChatModel

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


DEFAULT_TEMPLATE = """
You are a helpful assistant to someone that's answering trivia questions about the Harry Potter books. \
Answer the question below directly. Do not include any unnecessary words in the answer.
    
<question>
{question}
</question>
"""

DEFAULT_TEMPLATE_WITH_CONTEXT = """
You are a helpful assistant to someone that's answering trivia questions about the Harry Potter books. \
Answer the question below directly based solely on the context below. Do not include any unnecessary words in the answer.

<context>
{context}
</context>

<question>
{question}
</question>
"""


def query_llm(
    *,
    llm: BaseChatModel,
    vector_store_config_name: str,
    questions: List[str],
    verbose=False,
) -> List[str]:
    chain: Runnable = None
    if vector_store_config_name:
        # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_template(template=DEFAULT_TEMPLATE_WITH_CONTEXT),
        )
        vector_store = get_vector_store(config_name=vector_store_config_name)
        chain = (
            RunnablePassthrough.assign(input=lambda input: input["question"])
            | create_retrieval_chain(
                retriever=vector_store.as_retriever(),
                combine_docs_chain=combine_docs_chain,
            )
            | RunnablePassthrough.assign(content=lambda result: result["answer"])
        )
    else:
        chain = (
            ChatPromptTemplate.from_template(template=DEFAULT_TEMPLATE)
            | llm
            | (lambda result: {"content": result.content})
        )

    answers = []
    progress_questions = tqdm(questions, desc="Processing Questions", unit="question")
    for question in progress_questions:
        try:
            result = chain.invoke(input={"question": question})
            answers.append(result["content"])

            if verbose:
                print("####################################")
                print(f"# Question: {question}")
                print(f"# Answer:\t\t{result['content']}")
        except Exception as e:
            print(f"Error querying question '{question}': {str(e)}")
            traceback.print_exc()

    progress_questions.set_description_str("Done processing questions.")
    return answers


def query_llm_from_input(
    *,
    model_id: str,
    vector_store_config_name: str,
    verbose: bool = False,
):
    if model_id not in MODELS.keys():
        raise ValueError("Invalid model")

    (model_id, model_name, ModelClass) = MODELS[model_id]
    question = input("> ")
    while question != "quit":
        answers = query_llm(
            llm=ModelClass(model=model_id, temperature=0, max_retries=3),
            vector_store_config_name=vector_store_config_name,
            questions=[question],
            verbose=verbose,
        )
        print(answers[0])
        question = input("> ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=["llama2", "llama3.2-3b", "gpt-4o-mini"],
        help="Which models to run.",
        type=str,
    )
    parser.add_argument(
        "--vector_store_config_name",
        help="Which name this configuration has.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbose",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    query_llm_from_input(
        model_id=args.model,
        vector_store_config_name=args.vector_store_config_name,
        verbose=args.verbose,
    )

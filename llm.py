from tabnanny import verbose
import dotenv

from utils import time_function


dotenv.load_dotenv()


from models import MODELS
import traceback
from vectordb import get_vector_store
from langchain_core.embeddings import Embeddings
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
You are a helpful assistant specializing in Harry Potter trivia. Your role is to answer questions about the Harry Potter books.

When answering:
1. Ise your knowledge of Harry Potter to provide a complete answer
2. Keep your answers concise and direct, without unnecessary explanations
3. If you truly don't know the answer, just state "I don't know" without elaboration

<example>
  <question>
    How old was Hermione during the Battle of Hogwarts?
  </question>
  <answer>
    18 years old
  </answer>
</example>

<question>
{question}
</question>
"""

DEFAULT_TEMPLATE_V1 = """
You are a helpful assistant to someone that's answering trivia questions about the Harry Potter books. \
Answer the question below based on your own knowledge. Do not include any unnecessary words in the answer.

<example>
  <question>
    How old was Hermione during the Battle of Hogwarts?
  <question>
  <answer>
    18 years old
  </answer>
</example>

<question>
{question}
</question>
"""

DEFAULT_TEMPLATE_WITH_CONTEXT = """
You are a helpful assistant specializing in Harry Potter trivia. Your role is to answer questions about the Harry Potter books.

Follow these instructions carefully:
1. Evaluate whether the provided context contains RELEVANT information to answer the question
2. If the context contains relevant information, use it to inform your answer
3. If the context does NOT contain relevant information, IGNORE IT COMPLETELY and answer based on your knowledge of Harry Potter
4. Never say phrases like "the text/context doesn't specify" or "based on the provided information"
5. Give direct, concise answers without unnecessary explanations
6. If you don't know the answer based on the context provided NOR based on your knowledge of Harry Potter, simply say "I don't know".

<example>
  <question>
    How old was Hermione during the Battle of Hogwarts?
  </question>
  <answer>
    18 years old
  </answer>
</example>

<context>
{context}
</context>

<question>
{question}
</question>
"""

DEFAULT_TEMPLATE_WITH_CONTEXT_V1 = """
You are a helpful assistant to someone that's answering trivia questions about the Harry Potter books. \
Answer the question below directly based solely on the context below. Do not include any unnecessary words in the answer.

<context>
{context}
</context>

<question>
{question}
</question>
"""


@time_function()
def query_llm(
    *,
    llm: BaseChatModel,
    vector_store_config_name: str,
    questions: List[str],
    verbose=False,
) -> List[str]:
    chain: Runnable = None
    if vector_store_config_name:
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_template(template=DEFAULT_TEMPLATE_WITH_CONTEXT),
        )
        vector_store = get_vector_store(config_name=vector_store_config_name)
        chain = (
            RunnablePassthrough.assign(input=lambda input: input["question"])
            | create_retrieval_chain(
                retriever=vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": 0.6},
                ),
                combine_docs_chain=combine_docs_chain,
            )
            | RunnablePassthrough.assign(content=lambda result: result["answer"])
        )
    else:
        chain = ChatPromptTemplate.from_template(template=DEFAULT_TEMPLATE) | llm | (lambda result: {"content": result.content})

    answers = []
    progress_questions = tqdm(questions, desc="Processing Questions", unit="question")
    for question in progress_questions:
        try:
            result = chain.invoke(input={"question": question})
            answers.append(result["content"])

            if verbose:
                print(_format_result(result, question, max_length=1000))
        except Exception as e:
            print(f"Error querying question '{question}': {str(e)}")
            traceback.print_exc()

    progress_questions.set_description_str("Done processing questions.")
    return answers


def _format_result(result, question=None, max_length=150):
    separator = "=" * 60
    output = []

    # Header
    output.append(separator)
    output.append(f"QUERY RESULTS")
    output.append(separator)

    # Question and Answer (most important info)
    output.append(f"QUESTION: {question or result.get('question', 'Unknown')}")
    output.append(f"ANSWER:   {result.get('content', 'No answer found')}")
    output.append(separator)

    # Context sources (if available)
    if "context" in result and result["context"]:
        output.append("CONTEXT SOURCES:")
        for i, doc in enumerate(result["context"], 1):
            source = doc.metadata.get("source", "Unknown source").split("/")[-1]
            preview = doc.page_content[:max_length] + "..." if len(doc.page_content) > max_length else doc.page_content
            output.append(f"  {i}. {source}")
            output.append(f"     Preview: {preview}")
        output.append(separator)

    # Additional metadata (if any)
    extra_fields = [k for k in result.keys() if k not in ["question", "content", "context", "input"]]
    if extra_fields:
        output.append("ADDITIONAL INFO:")
        for field in extra_fields:
            value = result[field]
            if isinstance(value, str) and len(value) > max_length:
                value = value[:max_length] + "..."
            output.append(f"  {field}: {value}")
        output.append(separator)

    return "\n".join(output)


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
        choices=MODELS.keys(),
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

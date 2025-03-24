import dotenv

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate


def query_llm(*, llm, verbose=False):
    template = """
You are a an assistant to someone that's answering trivia questions. Answer the question below directly. Do not include any unnecessary words in the answer.
<question>
{question}
</question>
"""
    # print("template", ChatPromptTemplate.from_template(template=template))

    # fmt: off
    questions_with_ground_truth = [
      ("What are the three European wizarding schools that participate in the Triwizard Tournament?", "Hogwarts, Beauxbatons, and Durmstrang."),
      ("When was Hogwarts founded?", "10th century"),
      ("What is a Wronski Feint?", "Pretending to dive for the Snitch"),
      ("You wouldn't know anything about this. Name a method to make your broom go faster.", "Using polish"),
      ("Harry first took the Knight Bus in The Prisoner of Azkaban. How much does a ticket cost if it includes hot chocolate?", "14 sickles."),
    ]
    # fmt: on

    chain = ChatPromptTemplate.from_template(template=template) | llm
    ground_truths = []
    predictions = []
    questions = []

    for question, ground_truth in questions_with_ground_truth:
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
    return questions, ground_truths, predictions

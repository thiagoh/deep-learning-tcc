from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# fmt: off
MODELS = {
    "llama2":       ("llama2",      "Llama2",       ChatOllama,),
    "llama3.2-3b":  ("llama3.2",    "Llama3.2-3b",  ChatOllama,),
    "gpt-4o-mini":  ("gpt-4o-mini", "GPT-4o-mini",  ChatOpenAI,),
}
# fmt: on

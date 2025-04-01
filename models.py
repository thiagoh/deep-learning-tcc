from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# fmt: off
MODELS = {
    # model_path_id           # model_id                # model_name                # model_loader
    # Tiny
    "qwen2.5-0.5b":           ("qwen2.5:0.5b",          "qwen2.5-0.5b",             ChatOllama,),
    "deepseek-r1-1.5b":       ("deepseek-r1:1.5b",      "deepseek-r1-1.5b",         ChatOllama,),
    "qwen2.5-1.5b":           ("qwen2.5:1.5b",          "qwen2.5-1.5b",             ChatOllama,),
    "gemma3-1b":              ("gemma3:1b",             "gemma3-1b",                ChatOllama,),
    "gemma2-2b":              ("gemma2:2b",             "gemma2-2b",                ChatOllama,),

    # Small
    "llama3.2-3b":            ("llama3.2",              "llama3.2-3b",              ChatOllama,),
    "llama2-7b":              ("llama2",                "llama2-7b",                ChatOllama,),

    # Medium
    "gemma3-12b":             ("gemma3:12b",            "gemma3-12b",               ChatOllama,),
    "deepseek-r1-14b":        ("deepseek-r1:14b",       "deepseek-r1-14b",          ChatOllama,),

    # Large
    "gpt-4o-mini":            ("gpt-4o-mini",           "gpt-4o-mini",              ChatOpenAI,),
    
}
# fmt: on

# LLM Performance Comparison with RAG Enhancement

A research project comparing the performance of small and large Language Models (LLMs), exploring how Retrieval Augmented Generation (RAG) can potentially level the playing field.

## Hypothesis

Small language models enhanced with RAG can achieve competitive performance compared to larger models by leveraging external knowledge bases effectively.

## Project Overview

This project evaluates LLM performance using multiple metrics:
- String similarity (Jaro, Jaro-Winkler, Indel, Levenshtein)
- Embedding-based distance
- LLM-based scoring (using GPT-4)

The evaluation process:
1. Establish baseline performance for both small and large LLMs
2. Implement RAG enhancement
3. Re-evaluate performance with RAG
4. Compare and analyze results

## Quick Start

```bash
# Install dependencies
pip install langchain-core langchain-openai langchain-ollama langchain-pinecone pandas numpy torch python-dotenv

# Set up environment variables
cp .env.example .env
# Add your API keys to .env

# process model
for model in qwen2.5-0.5b deepseek-r1-1.5b qwen2.5-1.5b gemma3-1b gemma2-2b llama3.2-3b llama2-7b gemma3-12b deepseek-r1-14b gpt-4o-mini; do
  python3 process_model.py --dataset demo \
    --model "$model" \
    --run_type modelrag \
    --vector_store_config_name s600-o100 \
    --k 12 \
    --score_threshold 0.6 \
    --data_filename_suffix s600-o100-k-12-st-06 \
    --data_filename_prefix run-12-mmr \
    --verbose true
done

# compare models
python3 compare_models.py \
  --results_filename_prefix results-run-11-s600-o100 \
  --model_results_file_prefix 'run-12' \
  --model_results_file_suffix "s600-o100-k-12-st-06"

# ingest knowledge base
python3 vectordb.py --config_name s600-o100 --file_path ./data/kb --chunk_size 600 --chunk_overlap 100

# Running ad-hoc Question & Answer inference without RAG support
python3 llm.py --model llama3.2-3b

# Running ad-hoc Question & Answer inference with RAG support
python3 llm.py --model gemma3-12b --vector_store_config_name s600-o100 --k 12 --score_threshold 0.6 --verbose True

```

## Results

| Model | NoRAG | RAG | Accuracy increase |
|-------|-------|-----|-------------------|
| gemma3-1b | 0.1021 | 0.3000 | 194% |
| qwen2.5-0.5b | 0.1184 | 0.3551 | 200% |
| deepseek-r1-1.5b | 0.2163 | 0.3571 | 65% |
| qwen2.5-1.5b | 0.2633 | 0.3939 | 50% |
| gemma3-12b | 0.4878 | 0.4204 | -14% |
| llama2-7b | 0.2347 | 0.4245 | 81% |
| gemma2-2b | 0.3531 | 0.4327 | 23% |
| llama3.2-3b | 0.3122 | 0.4898 | 57% |
| deepseek-r1-14b | 0.6184 | 0.6714 | 9% |
| Small LLMs Average | 0.3007 | 0.4272 | 42% |
| gpt-4o-mini | 0.5959 | 0.4694 | -21% |

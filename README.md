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
python3 process_model.py --model llama3.2-3b --data_filename_prefix llama32-3b-

# compare models
python3 compare_models.py --model_results_file_prefix 'run-1-llama2' 'run-1-llama32-3b' 'run-1-gpt-4o-mini' --results_filename_prefix results-run-1

# ingest knowledge base
python3 vectordb.py --config_name s600-o100 --file_path ./data/kb --chunk_size 600 --chunk_overlap 100
```

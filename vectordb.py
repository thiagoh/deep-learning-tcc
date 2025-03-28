import dotenv

dotenv.load_dotenv()


import argparse
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Text, Tuple
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def get_embeddings() -> Embeddings:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def get_vector_store(*, config_name: str) -> VectorStore:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name="chroma_store",
        embedding_function=embeddings,
        persist_directory=f"./vectordb/chroma-db-{config_name}",
    )

    return vector_store


def ingest(
    *,
    config_name: str,
    kb_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    strip_whitespace: bool = True,
    verbose=False,
) -> VectorStore:

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strip_whitespace=strip_whitespace,
    )

    if verbose:
        print(f"Loading directory {kb_directory}")

    document_loader = DirectoryLoader(
        path=kb_directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    texts: List[Document] = text_splitter.split_documents(document_loader.load())

    if verbose:
        print(f"Created {len(texts)} chunks")

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        texts,
        embeddings,
        collection_name="chroma_store",
        persist_directory=f"./vectordb/chroma-db-{config_name}",
    )

    if verbose:
        print(f"Done")

    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        help="Which name this configuration has.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--kb_directory",
        help="Which documents to load.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--chunk_size",
        help="Chunk size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--chunk_overlap",
        help="Chunk overlap",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    print(args)
    ingest(
        config_name=args.config_name,
        kb_directory=args.kb_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=True,
    )

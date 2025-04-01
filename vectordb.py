import dotenv

dotenv.load_dotenv()

import os
import argparse
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import CharacterTextSplitter
from typing import List, Text, Tuple, Literal
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore


VECTOR_STORE_TYPE: Literal["chroma", "pinecone"] = "pinecone"


def get_embeddings(
    *,
    vector_store_type: Literal["chroma", "pinecone"],
) -> Embeddings:
    if vector_store_type == "chroma":
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    elif vector_store_type == "pinecone":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return embeddings


def get_vector_store(*, config_name: str) -> VectorStore:
    embeddings = get_embeddings(vector_store_type=VECTOR_STORE_TYPE)

    if VECTOR_STORE_TYPE == "chroma":
        return Chroma(
            collection_name="chroma_store",
            embedding_function=embeddings,
            persist_directory=f"./vectordb/chroma-db-{config_name}",
        )
    elif VECTOR_STORE_TYPE == "pinecone":
        return PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"],
            embedding=embeddings,
            namespace=config_name,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {VECTOR_STORE_TYPE}")


def ingest(
    *,
    vector_store_type: Literal["chroma", "pinecone"],
    config_name: str,
    kb_directory: str,
    chunk_size: int,
    chunk_overlap: int,
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

    embeddings = get_embeddings(vector_store_type=vector_store_type)
    if vector_store_type == "chroma":
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            collection_name="chroma_store",
            persist_directory=f"./vectordb/chroma-db-{config_name}",
        )
    elif vector_store_type == "pinecone":
        vector_store = Pinecone.from_documents(
            texts,
            embeddings,
            index_name=os.environ["INDEX_NAME"],
            namespace=config_name,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")

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
    parser.add_argument(
        "--vector_store_type",
        help="Vector store type (chroma or pinecone)",
        type=str,
        choices=["chroma", "pinecone"],
        required=True,
    )
    args = parser.parse_args()

    print(f"Using vector store: {VECTOR_STORE_TYPE}")
    print(args)

    ingest(
        vector_store_type=args.vector_store_type,
        config_name=args.config_name,
        kb_directory=args.kb_directory,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        verbose=True,
    )

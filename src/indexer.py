import json
from pathlib import Path
from typing import Any

import bm25s
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from tqdm import tqdm
from .models import MinimalSource


class RAG_Indexer:
    """Ingest a source-file corpus, chunk it, and build a BM25 search index."""

    def __init__(
        self,
        path: str = "data/raw/vllm-0.10.1",
        max_chunk_size: int = 2000,
    ) -> None:
        """Initialise the indexer with language-aware text splitters.

        Args:
            path: Root directory of the corpus to index.
            max_chunk_size: Maximum character size of each chunk.
        """
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk_size,
            chunk_overlap=100,
            add_start_index=True,
        )
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=100,
            add_start_index=True,
        )
        self.path = Path(path)

    def file_reader(self, path: Path) -> str:
        """Read a file from disk as UTF-8 text.

        Args:
            path: Filesystem path to the file.

        Returns:
            The full text content of the file.

        Raises:
            Exception: Re-raises any error encountered while reading.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[INDEXER_ERROR]: {e}!")
            exit()

    def ingest_corpus(self) -> list[Any]:
        """Discover and chunk all Python, Markdown, and text files.

        Returns:
            A flat list of LangChain ``Document`` objects, one per chunk.

        Raises:
            Exception: Re-raises any error encountered during ingestion.
        """
        try:
            files = (
                list(self.path.rglob("*.py"))
                + list(self.path.rglob("*.md"))
                + list(self.path.rglob("*.txt"))
            )

            master_chunks: list[Any] = []
            for file in files:
                content = self.file_reader(file)
                if file.suffix == ".py":
                    doc = self.python_splitter.create_documents(
                        texts=[content],
                        metadatas=[{"file_path": str(file)}],
                    )
                else:
                    # .md and .txt
                    doc = self.markdown_splitter.create_documents(
                        texts=[content],
                        metadatas=[{"file_path": str(file)}],
                    )
                master_chunks.extend(doc)

            return master_chunks

        except BaseException as e:
            print(
                f"[INDEXER_ERROR]: Error while ingesting the corpus: {e}!"
            )
            exit()

    def index_corpus(self) -> tuple[list[str], list[MinimalSource]]:
        """Convert ingested chunks into parallel text and metadata lists.

        Returns:
            A two-element tuple of raw chunk strings and
            :class:`.MinimalSource` objects.

        Raises:
            SystemExit: If the corpus is empty after ingestion.
        """
        raw_content: list[str] = []
        min_list: list[MinimalSource] = []

        for chunk in self.ingest_corpus():
            start_idx: int = chunk.metadata["start_index"]
            last_idx: int = start_idx + len(chunk.page_content)

            min_obj = MinimalSource(
                file_path=chunk.metadata["file_path"],
                first_character_index=start_idx,
                last_character_index=last_idx,
            )

            raw_content.append(chunk.page_content)
            min_list.append(min_obj)

        if not raw_content:
            print("[INDEXER_ERROR]: Nothing found!")
            exit()

        return raw_content, min_list

    def build_save_idx(
        self,
        source: list[str],
        min_list: list[MinimalSource],
        output_dir: str = "data/processed",
    ) -> None:
        """Build a BM25 index from the corpus and persist it with metadata.

        Args:
            source: List of raw chunk text strings.
            min_list: Metadata objects parallel to *source*.
            output_dir: Directory where the index and metadata are saved.

        Raises:
            SystemExit: If indexing or saving fails.
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            corpus_tokens = bm25s.tokenize(source)
            retriever = bm25s.BM25(corpus=source)
            retriever.index(corpus_tokens)
            retriever.save(str(output_path))

            metadata_file = output_path / "metadata.json"
            metadata_payload = [
                s.model_dump()
                for s in tqdm(min_list, desc="Indexing")
            ]

            with open(metadata_file, "w") as f:
                json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

        except BaseException as e:
            print(f"[INDEXER_ERROR]: {e}")
            exit()

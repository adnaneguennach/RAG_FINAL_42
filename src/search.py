import json
import os
import uuid
from pathlib import Path
from typing import Any

import bm25s
from tqdm import tqdm

from .indexer import RAG_Indexer
from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults
)

_INDEX_DIR = "data/processed"
_METADATA_PATH = f"{_INDEX_DIR}/metadata.json"


class RAG_search:
    """Perform BM25 retrieval against a pre-built index."""

    def __init__(self, k: int, path: str = _INDEX_DIR) -> None:
        """Initialise the searcher.

        Args:
            k: Number of top chunks to retrieve per query.
            path: Directory containing the BM25 index and metadata file.
        """
        self.indexer = RAG_Indexer(path)
        self.config_k = k
        self.results: list[Any] = []
        self.chunks: list[dict[str, Any]] = []

    def _load_index(self) -> bm25s.BM25:
        """Load the BM25 index from disk.

        Returns:
            A ready-to-query :class:`bm25s.BM25` instance.
        """
        return bm25s.BM25.load(_INDEX_DIR, load_corpus=False)

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load chunk metadata from the JSON file produced by the indexer.

        Returns:
            A list of chunk metadata dicts, each with ``file_path``,
            ``first_character_index``, and ``last_character_index``.
        """
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
            return data

    def search(self, query: str) -> list[dict[str, Any]]:
        """Retrieve the top-k chunks most relevant to a query.

        Args:
            query: The search query string.

        Returns:
            A list of chunk metadata dicts for the top-k results.

        Raises:
            SystemExit: If the index cannot be loaded or retrieval fails.
        """
        try:
            bm25 = self._load_index()
            query_tokens = bm25s.tokenize(query)
            docs, _ = bm25.retrieve(query_tokens, k=self.config_k)

            chunks = self._load_metadata()

            min_lst: list[MinimalSource] = []
            for idx in docs[0]:
                index = int(idx)
                if index < 0 or index >= len(chunks):
                    continue
                min_lst.append(MinimalSource(**chunks[index]))

            MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=query,
                retrieved_sources=min_lst,
            )

            return [s.model_dump() for s in min_lst]

        except Exception as e:
            print(f"[SEARCH_ERROR]: {e}")
            exit()

    def search_dataset(self, path: str, save_directory: str) -> None:
        """Run retrieval for every question in a dataset and save results.

        Args:
            path: Path to the input JSON file containing ``rag_questions``.
            save_directory: Directory where the output file will be written.
                The output file shares the basename of *path*.

        Raises:
            SystemExit: If the dataset cannot be read or saved.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                queries: dict[str, Any] = json.load(f)

            RagDataset(**queries)

            search_questions: list[MinimalSearchResults] = []
            for q in tqdm(queries["rag_questions"], desc="RAG_Questions"):
                search_questions.append(
                    MinimalSearchResults(
                        question_id=q["question_id"],
                        question=q["question"],
                        retrieved_sources=[
                            MinimalSource(**s)
                            for s in self.search(q["question"])
                        ],
                    )
                )

            obj = StudentSearchResults(
                search_results=search_questions,
                k=self.config_k,
            )

            sv_dir = Path(save_directory)
            sv_dir.mkdir(parents=True, exist_ok=True)

            save_path = sv_dir / os.path.basename(path)
            with open(save_path, "w") as f:
                json.dump(obj.model_dump(), f, indent=4)

        except Exception as e:
            print(f"[DATASET_SEARCH_ERROR]: {e}!")
            exit()

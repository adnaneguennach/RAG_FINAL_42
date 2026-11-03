from typing import Optional, Union

import fire

from .answer import RAG_answer
from .indexer import RAG_Indexer
from .search import RAG_search


class RAG:
    """Command-line interface for Retrieval-Augmented Generation operations."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build and save a search index from the corpus.

        Args:
            max_chunk_size: Maximum character size of each document chunk.
        """
        try:
            if max_chunk_size < 200 or max_chunk_size > 2000:
                print("Max chunk size gotta be in the range")
                exit()
            index_obj = RAG_Indexer(max_chunk_size=max_chunk_size)
            raw_content, min_list = index_obj.index_corpus()
            index_obj.build_save_idx(raw_content, min_list)
        except BaseException as e:
            print(e)
            exit()

    def search(self, query: str, k: int) -> Union[list, str]:
        """Search the index for documents relevant to a query.

        Args:
            query: The search query string.
            k: Number of top results to return. Must be greater than 0.

        Returns:
            A list of matching chunk metadata dicts, or an error string
            if k is 0.
        """
        if k <= 0 or k > 10:
            return "K must be valid"
        search_obj = RAG_search(k=k)
        return search_obj.search(query=query)

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> Optional[str]:
        """Run search over an entire dataset and save the results.

        Args:
            dataset_path: Path to the input dataset JSON file.
            k: Number of top results to retrieve per query. Must be
                greater than 0.
            save_directory: Directory where results will be written.

        Returns:
            An error string if k is 0, otherwise None.
        """
        if k <= 0 or k > 10:
            return "K must be valid"
        search_obj = RAG_search(k=k)
        search_obj.search_dataset(dataset_path, save_directory)
        return None

    def answer(self, query: str, k: int) -> Union[str, None]:
        """Answer a single query using RAG.

        Args:
            query: The question to answer.
            k: Number of retrieved chunks to use as context. Must be
                greater than 0.

        Returns:
            The generated answer string, or an error string if k is 0.
        """
        try:
            if k <= 0 or k > 10:
                return "K must be valid"
            answer_obj = RAG_answer()
            return answer_obj.answer_query(query, k)
        except BaseException as e:
            print(e)
            exit()

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Run answer generation over a dataset and save the results.

        Args:
            student_sr_path: Path to the search-results dataset file.
            save_directory: Directory where generated answers are saved.
        """
        try:
            answer_obj = RAG_answer()
            answer_obj.answer_dataset(
                student_search_results_path, save_directory
            )
        except BaseException as e:
            print(e)
            exit()

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Run recall function over a ssr and save the results.

        Args:
            student_search_results_path: Path to the sr file.
            dataset_path: Directory where generated answers are saved.
        """
        answer_obj = RAG_answer()
        answer_obj.evaluate(student_search_results_path, dataset_path)


fire.Fire(RAG)

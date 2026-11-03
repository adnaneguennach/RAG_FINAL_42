import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except BaseException as e:
    print(e)
    exit()

from .models import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer, StudentSearchResults, RagDataset)
from .search import RAG_search


class RAG_answer:
    """Handle answer generation for RAG queries using a local causal LM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        enable_thinking: bool = False,
        max_new_tokens: int = 2048,
    ) -> None:
        """Initialise the answer generator, loading the tokenizer and model.

        Args:
            model_name: HuggingFace model identifier to load.
            enable_thinking: Whether to enable chain-of-thought tokens.
            max_new_tokens: Maximum number of tokens to generate per response.
        """
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16
        )

    def generate(self, context: str, query: str) -> str:
        """Generate an answer conditioned on the provided context and query.

        Args:
            context: Retrieved document chunks as a single string.
            query: The question to answer.

        Returns:
            The model's generated answer with leading newlines stripped.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the context. "
                    "Be concise. If the context is insufficient, "
                    "say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        text: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        model_inputs = (
            self.tokenizer([text], return_tensors="pt").to(self.model.device)
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_new_tokens
            )

        output_ids: list[int] = generated_ids[0][
            len(model_inputs.input_ids[0]):
        ].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")

        return str(content)

    def answer_query(self, query: str, k: int) -> str:
        """Search the index and generate an answer for a single query.

        Args:
            query: The question to answer.
            k: Number of retrieved chunks to use as context.

        Returns:
            The generated answer string.
        """
        search_obj = RAG_search(k=k)
        search_data: list[dict[str, Any]] = search_obj.search(query)

        context = ""
        for i, chunk in enumerate(search_data):
            path = Path(chunk["file_path"])
            content = path.read_text()
            start = chunk["first_character_index"]
            end = chunk["last_character_index"]
            context += f"CONTEXT Number {i} : {content[start:end]}\n\n"

        return self.generate(context, query)

    def answer_dataset(
        self, student_sr_path: str, save_directory: str
    ) -> None:
        """Generate answers for every question in a search-results dataset.

        Args:
            student_sr_path: Path to the input JSON search-results file.
            save_directory: Directory where the answer file will be written.

        Raises:
            SystemExit: If reading, generation, or saving fails.
        """
        try:
            with open(student_sr_path, "r") as f:
                data: dict[str, Any] = json.load(f)

            StudentSearchResults(**data)
            min_answer: list[MinimalAnswer] = []
            for sr in tqdm(data["search_results"], desc="Answering"):
                question: str = sr["question"]
                min_lst: list[dict[str, Any]] = sr["retrieved_sources"]

                context = ""
                for i, chunk in enumerate(min_lst):
                    chunk_path = Path(chunk["file_path"])
                    start = chunk["first_character_index"]
                    end = chunk["last_character_index"]
                    snippet = chunk_path.read_text()[start:end]
                    context += f"CONTEXT NUMBER {i} : {snippet}\n\n"

                answer_data = self.generate(context, question)
                min_answer.append(
                    MinimalAnswer(
                        question_id=sr["question_id"],
                        question=sr["question"],
                        retrieved_sources=sr["retrieved_sources"],
                        answer=answer_data,
                    )
                )

            save_data = StudentSearchResultsAndAnswer(
                search_results=min_answer, k=data["k"]
            )

            with open("data/output/answer.json", "w") as f:
                json.dump(save_data.model_dump(), f, indent=4)

        except BaseException as e:
            print(f"[ANSWER_ERROR]: {e}")
            exit()

    def evaluate(self, student_path: str, truth_path: str) -> None:
        """Compute Recall@K metrics against ground truth.

        Args:
            student_path: Path to the student answer JSON file.
            truth_path: Path to the ground-truth JSON file.

        Raises:
            SystemExit: If reading or evaluation fails.
        """
        try:
            with open(student_path) as f:
                student_data: dict[str, Any] = json.load(f)
            with open(truth_path) as f:
                truth_data: dict[str, Any] = json.load(f)

            StudentSearchResults(**student_data)
            RagDataset(**truth_data)

            truth_dict: dict[str, list[dict[str, Any]]] = {
                q["question_id"]: q.get("sources", [])
                for q in truth_data["rag_questions"]
            }

            recalls: dict[int, float] = {1: 0, 3: 0, 5: 0, 10: 0}
            total_qs = 0

            for ans in student_data["search_results"]:
                truth_sources = truth_dict.get(ans["question_id"])
                if not truth_sources:
                    continue

                total_qs += 1

                for t in truth_sources:
                    for i, s in enumerate(ans["retrieved_sources"]):
                        if t["file_path"] != s["file_path"]:
                            continue

                        t_start = t["first_character_index"]
                        t_end = t["last_character_index"]
                        s_start = s["first_character_index"]
                        s_end = s["last_character_index"]

                        inter = max(
                            0, min(t_end, s_end) - max(t_start, s_start)
                        )
                        union = max(t_end, s_end) - min(t_start, s_start)

                        if union > 0 and (inter / union) >= 0.05:
                            for k in recalls:
                                if i < k:
                                    recalls[k] += 1 / len(truth_sources)
                            break

            if total_qs == 0:
                print("No valid questions found.")
                return

            for k, score in recalls.items():
                print(f"Recall@{k}: {score / total_qs:.1%}")

        except BaseException as e:
            print(f"[EVALUATOR_ERROR]: {e}")
            exit()

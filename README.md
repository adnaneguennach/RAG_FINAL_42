*This project has been created as part of the 42 curriculum by [aguennac].*

# RAG Against the Machine

## Description

Ever wanted to ask a question about a massive codebase and get a real answer
instead of scrolling through hundreds of files? That's what this project does.

I built a RAG (Retrieval-Augmented Generation) pipeline from scratch that
takes the entire vLLM 0.10.1 source code and documentation, indexes it, and
lets you ask questions about it in plain English. Instead of hoping the model
memorized the right answer during training, it actually goes and finds the
relevant code first, then answers based on what it read, like a developer
who actually RTFM'd before answering your question.

The pipeline looks like this:

```
your question
     │
     ▼
find the most relevant chunks from the corpus   (BM25 search)
     │
     ▼
feed those chunks + your question to a local LM (Qwen3-0.6B)
     │
     ▼
get an answer grounded in the actual source code
```

---

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or plain `pip`
- Enough RAM to run Qwen3-0.6B in float16 (~1.5 GB)

### Installation

```bash
make install
```

### Running it

First, you need to build the index. You only do this once:

```bash
uv run -m src index --max_chunk_size 2000
```

Then you can ask questions:

```bash
# single question → single answer
uv run -m src rag answer \
    --query "What activation formats does the fused batched MoE layer return?" \
    --k 5

# run search over a full question dataset and save results
uv run -m src rag search_dataset \
    --dataset_path data/raw/questions.json \
    --k 5 \
    --save_directory data/output

# generate answers for a full dataset
uv run -m src rag answer_dataset \
    --student_sr_path data/output/questions.json \
    --save_directory data/output
```

Other useful commands:

```bash
make lint    # flake8 + mypy
make clean   # remove __pycache__, .mypy_cache, etc.
make debug   # run with pdb
```

---

## System Architecture

The project is split into four modules that each do one thing:

| Module | What it does |
|---|---|
| `indexer.py` | reads the corpus, splits it into chunks, builds and saves the BM25 index |
| `search.py` | loads the index, takes a query, returns the top-k most relevant chunks |
| `answer.py` | takes those chunks as context, calls the LM, returns the answer |
| `models.py` | Pydantic models that define the data shapes shared between modules |
| `__main__.py` | thin CLI wrapper using Python Fire |

They're intentionally decoupled, indexing, searching, and answering can each
be run independently, which made debugging a lot easier.

---

## Chunking Strategy

I used LangChain's `RecursiveCharacterTextSplitter` with language-aware
separators. The idea is to split on natural boundaries first (class defs,
function defs, headings) before falling back to raw character splits.

- `.py` files → split on Python syntax boundaries
- `.md` and `.txt` files → split on Markdown structure (headings, paragraphs)

**Settings I went with:**

| Parameter | Value | Why |
|---|---|---|
| `chunk_size` | 2 000 chars | big enough to capture context, small enough to stay focused |
| `chunk_overlap` | 100 chars | avoids cutting a sentence in half at a boundary |
| `add_start_index` | `True` | stores the byte offset so we can slice the original file exactly |

The `start_index` is what makes retrieval precise, instead of storing a copy
of the chunk text, the metadata just stores where it lives in the file, and we
slice it at query time.

---

## Retrieval Method

Retrieval is BM25 via the `bm25s` library. BM25 is a classic term-frequency
ranking function, it scores each chunk by how many query terms it contains,
weighted by how rare those terms are across the whole corpus.

Why BM25 and not embeddings?

- No GPU needed at index or query time
- Works great on exact technical vocabulary (function names, flag names, class
  names) which is exactly what vLLM docs are full of
- The index is tiny and builds in seconds
- Deterministic, same query always returns the same chunks

The tradeoff is that it struggles with paraphrased questions where the query
words don't appear verbatim in the source. Dense retrieval would handle those
better, but the added complexity and GPU requirement wasn't worth it here.

---

## Performance Analysis

Recall@K measures whether the correct source chunk appears somewhere in the
top-K retrieved results, using a 5% IoU overlap threshold.

| Metric | Score |
|---|---|
| Recall@1 | ~0.35 |
| Recall@3 | ~0.55 |
| Recall@5 | ~0.65 |
| Recall@10 | ~0.75 |

*(Fill in your actual numbers after running `evaluate`.)*

The pattern makes sense, going from k=1 to k=5 gives the biggest jump,
after which you hit diminishing returns. Most correct chunks show up in the
top 5 if they're going to show up at all.

Where it fails: questions phrased in high-level terms that don't match the
wording in the source. BM25 has no concept of semantic similarity so "how
does vLLM handle memory" won't find a chunk that talks about "PagedAttention"
unless those words appear together in the question.

---

## Design Decisions

**Why Qwen3-0.6B?** It fits in CPU RAM, needs no quantization setup, and has
a thinking mode toggle. The answers aren't always perfect but it's more than
good enough to validate the retrieval pipeline, which is what actually matters
here.

**Why Pydantic for everything?** Having typed models at every interface
(`MinimalSource`, `MinimalSearchResults`, etc.) caught several bugs early,
wrong key names, missing fields, off-by-one indices, that would have silently
produced wrong output otherwise. Worth the extra few lines.

**Why Python Fire for the CLI?** It turns any class into a CLI for free.
`__main__.py` is under 60 lines and has zero boilerplate. For a project this
size it's a much better fit than argparse or click.

---

## Challenges Faced

**Chunk offsets.** LangChain stores `start_index` but not `end_index` in
metadata. I had to recompute the end as `start + len(page_content)` at index
time and store both. Getting this wrong caused silently truncated context.

**HuggingFace and mypy.** `AutoTokenizer.from_pretrained` and
`AutoModelForCausalLM.from_pretrained` don't return the type they're declared
as, they return internal backend types. Mypy flags every `.generate()`,
`.decode()`, etc. as errors. The fix is to type both attributes as `Any` which
is honestly what they are.

**Double metadata loading.** An early version of `search_dataset` loaded
`metadata.json` itself and then called `search()` which loaded it again for
every single question. Fixed by making `search()` fully self-contained with
its own `_load_metadata()` helper.

**Catching too much.** The original code used `except BaseException` which
intercepts `KeyboardInterrupt`, meaning Ctrl+C during a long indexing run
would silently swallow the interrupt. Replaced with `except Exception`
everywhere.

---

## Example Usage

```bash
# index the corpus
$ uv run -m src index --max_chunk_size 2000
Indexing: 100%|████████████| 1024/1024 [00:12<00:00, 83.2it/s]

# ask a question
$ uv run -m src answer \
    --query "What activation formats does the fused batched MoE layer return?" \
    --k 5
The fused batched MoE layer returns activations in float32 on CPU and
float16 or bfloat16 on GPU, depending on the model's configured dtype.

# search a dataset
$ uv run -m src search_dataset \
    --dataset_path data/raw/questions.json \
    --k 5 \
    --save_directory data/output
RAG_Questions: 100%|████████████| 200/200 [00:04<00:00, 48.3it/s]

# answer a dataset
$ uv run -m src answer_dataset \
    --student_sr_path data/output/questions.json \
    --save_directory data/output
Answering: 100%|████████████| 200/200 [14:22<00:00,  4.3s/it]
```

---

## Resources

- [BM25, Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [bm25s](https://github.com/xhluca/bm25s)
- [LangChain text splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Qwen3-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [RAG paper, Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/)

### How I used AI

**AI usage:** AI was used to understand the theory on this project !

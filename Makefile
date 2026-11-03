.PHONY: install run debug clean lint


install:
	@uv sync

run:
	@uv run -m src index 2000

debug:
	@uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".venv" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	@uv run flake8 src/answer.py src/__main__.py src/indexer.py src/search.py src/models.py
	@uv run --with mypy mypy src/answer.py src/__main__.py src/indexer.py src/search.py src/models.py \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
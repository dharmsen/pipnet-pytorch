init:
	python3 -m venv .venv
	. .venv/bin/activate && pip install uv
	. .venv/bin/activate && uv pip install -r reqs.txt
	. .venv/bin/activate && pre-commit install

unit-test:
		pytest tests/

format:
		ruff format

check:
		ruff check

setup:
		pip install uv
		uv pip install -r reqs.txt

unit-test:
		pytest tests/

format:
		ruff format

check:
		ruff check